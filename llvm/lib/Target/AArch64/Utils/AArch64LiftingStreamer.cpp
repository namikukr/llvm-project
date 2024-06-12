//===-- AArch64LiftingStreamer.cpp - Assembly to IR lifter ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64LiftingStreamer.h"
#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCLabel.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCWinCOFFStreamer.h"
#include <memory>
#include <stack>
#include <utility>

// TODO: add LLVM_DEBUG to relevant debug outputs, delete rest
#define DEBUG_TYPE "ARM64ECStreamer"

using namespace llvm;

namespace {

// Helper to tell whether an MCInst is a branch.
// Returns the index of the target.

// FIXME: these should probably be STATISTICs. 
static unsigned InstructionsRewritten = 0;
static unsigned LoadsInserted = 0;
static unsigned StoresInserted = 0;
static unsigned TotalInstructions = 0;
static unsigned InstructionsAnalyzed = 0;

static constexpr unsigned VirtualVectorFlag = MCRegister::VirtualRegFlag >> 1;
static inline unsigned isBranch(const MCInst &Inst) {
  switch (Inst.getOpcode()) {
  case AArch64::B:
    return 0;
  case AArch64::Bcc:
  case AArch64::CBZW:
  case AArch64::CBZX:
  case AArch64::CBNZW:
  case AArch64::CBNZX:
    return 1;
  case AArch64::TBZW:
  case AArch64::TBZX:
  case AArch64::TBNZW:
  case AArch64::TBNZX:
    return 2;
  default:
    return 0xDEAD;
  }
}

// Returns true if an MCRegUnit is a GPR, false otherwise.
static inline bool isGPR(MCRegUnit MCRU, const MCRegisterInfo *MRI) {
  bool Ret = false;
  MCRegUnit B = *MRI->regunits(AArch64::X0).begin();
  MCRegUnit E = *MRI->regunits(AArch64::X28).begin();
  Ret = MCRU >= B && MCRU <= E;
  B = *MRI->regunits(AArch64::W0).begin();
  E = *MRI->regunits(AArch64::W30).begin();
  Ret |= MCRU >= B && MCRU <= E;
  return Ret;
}

// Returns true if an MCRegUnit is a vector register, false otherwise.
static inline bool isVectorReg(MCRegUnit MCRU, const MCRegisterInfo *MRI) {
  MCRegUnit B = *MRI->regunits(AArch64::Q0).begin();
  MCRegUnit E = *MRI->regunits(AArch64::Q31).begin();
  return MCRU >= B && MCRU <= E;
}

// Obtains the name of the branch target of an instruction.
static StringRef getTarget(const MCInst &Inst) {
  unsigned OpNum = isBranch(Inst);
  assert(OpNum != 0xDEAD && "Not a branch instruction!");
  assert(Inst.getOperand(OpNum).isExpr() && "Branch target not an expression!");
  const MCExpr *Expr = Inst.getOperand(OpNum).getExpr();
  const MCSymbol &Sym = cast<MCSymbolRefExpr>(Expr)->getSymbol();
  return Sym.getName();
}

static inline bool isRet(const MCInst &Inst) {
  return Inst.getOpcode() == AArch64::RET;
}

static inline bool isVirtual(MCRegister Reg) {
  return Reg & MCRegister::VirtualRegFlag;
}

static inline bool isVirtualGPR(MCRegister Reg) {
  return isVirtual(Reg) && !(Reg & VirtualVectorFlag);
}

static inline bool isVirtualVector(MCRegister Reg) {
  return isVirtual(Reg) && (Reg & VirtualVectorFlag);
}

static inline bool isSpecialReg(MCRegister Reg) {
  return Reg == AArch64::WZR || Reg == AArch64::XZR ||
         Reg == AArch64::W30 || Reg == AArch64::LR ||
         Reg == AArch64::SP;
}

static inline Twine getRegName(MCRegister Reg, const MCRegisterInfo *MRI) {
  if (isVirtual(Reg)) {
    if (isVirtualVector(Reg))
      return Twine("VV") + Twine(Reg ^ (MCRegister::VirtualRegFlag | (MCRegister::VirtualRegFlag >> 1)));
    
    return Twine("VGPR") + Twine(Reg ^ MCRegister::VirtualRegFlag);
  }
  return MRI->getName(Reg);
}

static inline bool regsEqual(MCRegister Reg1, MCRegister Reg2, const MCRegisterInfo *MRI) {
  if (isVirtual(Reg1) && isVirtual(Reg2)) {
    return Reg1 == Reg2;
  }

  if (!isVirtual(Reg1) && !isVirtual(Reg2)) {
    return MRI->regsOverlap(Reg1, Reg2);
  }

  return false;
}

// Gets the symbolic name of an MCRegUnit.
static inline Twine getRegUnitName(MCRegUnit Reg,
                                         const MCRegisterInfo *MRI) {
  if (isVirtual(Reg)) {
    if (isVirtualVector(Reg))
      return Twine("VV") + Twine(Reg ^ (MCRegister::VirtualRegFlag | (MCRegister::VirtualRegFlag >> 1)));
    
    return Twine("VGPR") + Twine(Reg ^ MCRegister::VirtualRegFlag);
  }
  MCRegUnitRootIterator Roots(Reg, MRI);
  assert(Roots.isValid() && "Unit has no roots.");
  return MRI->getName(*Roots);
}

static inline const MCRegister getRegFromUnit(MCRegUnit Reg,
                                         const MCRegisterInfo *MRI) {
  MCRegUnitRootIterator Roots(Reg, MRI);
  assert(Roots.isValid() && "Unit has no roots.");
  return *Roots;
}
static inline MCRegUnit getRegUnit(MCRegister Reg, const MCRegisterInfo *MRI) {
  if (isVirtual(Reg)) return Reg;
  auto RegUnits = MRI->regunits(Reg);
  return *RegUnits.begin();
}

static inline bool isGPR(MCRegister Reg) {
  return (Reg >= AArch64::X0 && Reg <= AArch64::X28) || 
         (Reg >= AArch64::W0 && Reg <= AArch64::W28);
}

static inline bool isVectorReg(MCRegister Reg) {
  return (Reg >= AArch64::Q0 && Reg <= AArch64::Q31) ||
         (Reg >= AArch64::D0 && Reg <= AArch64::D31) ||
         (Reg >= AArch64::S0 && Reg <= AArch64::S31) ||
         (Reg >= AArch64::H0 && Reg <= AArch64::H31) ||
         (Reg >= AArch64::B0 && Reg <= AArch64::B31);
}

static void loadGPR(MCInst &Inst, MCRegister To, MCRegister From) {
  assert(isVirtualGPR(To) || isGPR(To));
  Inst.setOpcode(AArch64::LDURXi);
  Inst.clear();
  Inst.addOperand(MCOperand::createReg(To));
  Inst.addOperand(MCOperand::createReg(From));
  Inst.addOperand(MCOperand::createImm(16));
}

static void storeGPR(MCInst &Inst, MCRegister From, MCRegister To) {
  assert(isVirtualGPR(From) || isGPR(From));
  Inst.setOpcode(AArch64::STURXi);
  Inst.clear();
  Inst.addOperand(MCOperand::createReg(From));
  Inst.addOperand(MCOperand::createReg(To));
  Inst.addOperand(MCOperand::createImm(16));
}

static void loadVec(MCInst &Inst, MCRegister To, MCRegister From) {
  Inst.setOpcode(AArch64::LDURQi);
  Inst.clear();
  Inst.addOperand(MCOperand::createReg(To));
  Inst.addOperand(MCOperand::createReg(From));
  Inst.addOperand(MCOperand::createImm(16));
}
static void storeVec(MCInst &Inst, MCRegister From, MCRegister To) {
  assert(isVirtualVector(From) || isVectorReg(From));
  Inst.setOpcode(AArch64::STURQi);
  Inst.clear();
  Inst.addOperand(MCOperand::createReg(From));
  Inst.addOperand(MCOperand::createReg(To));
  Inst.addOperand(MCOperand::createImm(16));
}
// static inline std::optional<unsigned> defines(MCInst &Inst, MCRegister R, const MCInstrInfo &MII) {
//   unsigned NumDefs = MII.get(Inst.getOpcode()).getNumDefs();
//   for (unsigned i = 0; i < NumDefs; ++i) {
//     MCOperand &O = Inst.getOperand(i);
//     if (O.isReg()) {
//       if (O.getReg() == R) {
//         return i;
//       }
//     }
//   }
//   return std::nullopt;
// }

// static inline std::optional<unsigned> uses(MCInst &Inst, MCRegister R, const MCInstrInfo &MII) {
//   unsigned NumDefs = MII.get(Inst.getOpcode()).getNumDefs();
//   for (unsigned i = NumDefs; i < Inst.getNumOperands(); ++i) {
//     MCOperand &O = Inst.getOperand(i);
//     if (O.isReg()) {
//       if (O.getReg() == R) {
//         return i;
//       }
//     }
//   }
//   return std::nullopt;
// }
} // end anonymous namespace

// Prints a basic block's information.
void MCBasicBlock::dump(raw_ostream &OS) {
  OS << "Basic block" << BlockNum << ": \n";
  for (auto &I : Objects) {
    OS << "\t\t" << *I << "\n";
  }
}

MCFunction::MCFunction() { }

// Creates an initial MCFunction for the streamer.
AArch64LiftingStreamer::AArch64LiftingStreamer(MCStreamer *S)
    : MCStreamer(S->getContext()), Streamer(S), MCBBB(&S->getContext()) {
  std::unique_ptr<MCFunction> Function = std::make_unique<MCFunction>();
  CurrentFunction = Function.get();
  Functions.push_back(std::move(Function));
  // the first instruction of the function is a leader.
  MCBBB.NextInstrIsFuncStart = true;
}

void AArch64LiftingStreamer::emitLabel(MCSymbol *Symbol, SMLoc Loc) {
  // If we are emitting to an object file, register the symbol with 
  // the assembler so we can parse directional labels.
  
  // Hack: dummy define the labels so that we can buffer them without error.
  Symbol->setFragment(&getCurrentSectionOnly()->getDummyFragment());

  // The instruction immediately after this label is its "location".
  MCBBB.LabelLocs[Symbol->getName()] = MCBBB.Objects.size();
  std::unique_ptr<MCLabelContainer> LabelContainer = std::make_unique<MCLabelContainer>(Symbol, Loc, getCurrentSection());
  if (MCBBB.NextInstrIsFuncStart) {
    LabelContainer->IsLeader = 1;
    LabelContainer->IsFuncStart = 1;
    MCBBB.NextInstrIsFuncStart = false;
    // debugging
    LLVM_DEBUG(dbgs() << "Instruction " << MCBBB.Objects.size()
           << " is a leader. (start of function) \n");
  }
  MCBBB.Labels.insert(LabelContainer.get());
  MCBBB.Objects.push_back(std::move(LabelContainer));
}

// Catches each instruction and processes it into the internal state of the
// streamer.
void AArch64LiftingStreamer::emitInstruction(const MCInst &Inst,
                                             const MCSubtargetInfo &STI) {
  ++InstructionsAnalyzed;
  // Handles cases in which an asm source file does not
  // start with a .text directive.
  if (FirstInstrNotEmitted) {
    FirstInstrNotEmitted = false;
  }
  std::unique_ptr<MCInstContainer> InstContainer = std::make_unique<MCInstContainer>(Inst, getCurrentSection());
  // Make sure we record if the next instruction is a function start.
  if (MCBBB.NextInstrIsFuncStart) {
    InstContainer->IsLeader = 1;
    InstContainer->IsFuncStart = 1;
    MCBBB.NextInstrIsFuncStart = false;
    // debugging
    LLVM_DEBUG(dbgs() << "Instruction " << MCBBB.Objects.size()
           << " is a leader. (start of function) \n");
  }

  MCBBB.Objects.push_back(std::move(InstContainer));
}

void AArch64LiftingStreamer::emitBytes(StringRef Data) {
  MCBBB.Objects.push_back(std::make_unique<MCDataContainer>(Data, getCurrentSection()));
}

void AArch64LiftingStreamer::emitValueImpl(const MCExpr *Value, unsigned Size, SMLoc Loc) {
  std::function<void()> F = [=] () {
    Streamer->emitValueImpl(Value, Size, Loc);
  };
  MCBBB.Objects.push_back(std::make_unique<MCCallback>(F, getCurrentSection()));
}
void AArch64LiftingStreamer::emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                              Align ByteAlignment) {
  Streamer->emitCommonSymbol(Symbol, Size, ByteAlignment);
}

bool AArch64LiftingStreamer::emitSymbolAttribute(MCSymbol *Symbol,
                                                 MCSymbolAttr Attribute) {
  return Streamer->emitSymbolAttribute(Symbol, Attribute);
}

void AArch64LiftingStreamer::emitZerofill(MCSection *Section, MCSymbol *Symbol,
                                          uint64_t Size, Align ByteAlignment,
                                          SMLoc Loc) {
  Streamer->emitZerofill(Section, Symbol, Size, ByteAlignment, Loc);
}

void AArch64LiftingStreamer::emitDataRegion(MCDataRegionType Kind) { 
  std::function<void()> F = [=] () {
    Streamer->emitDataRegion(Kind);
  };
  MCBBB.Objects.push_back(std::make_unique<MCCallback>(F, getCurrentSection()));
}

void AArch64LiftingStreamer::emitValueToAlignment(Align Alignment, int64_t Value, unsigned ValueSize, unsigned MaxBytesToEmit) {
  std::function<void()> F = [=] () {
    Streamer->emitValueToAlignment(Alignment, Value, ValueSize, MaxBytesToEmit);
  };
  MCBBB.Objects.push_back(std::make_unique<MCCallback>(F, getCurrentSection()));
}

void AArch64LiftingStreamer::emitCodeAlignment(Align Alignment, const MCSubtargetInfo *STI, unsigned MaxBytesToEmit) {
  std::function<void()> F = [=] () {
    Streamer->emitCodeAlignment(Alignment, STI, MaxBytesToEmit);
  };
  MCBBB.Objects.push_back(std::make_unique<MCCallback>(F, getCurrentSection()));
}

MCBasicBlock &AArch64LiftingStreamer::getCurrentBasicBlock() {
  return *CurrentFunction->CurrentBlock;
}

void AArch64LiftingStreamer::initSections(bool NoExecStack, const MCSubtargetInfo &STI) {
  MCStreamer::initSections(NoExecStack, STI);
  Streamer->initSections(NoExecStack, STI);
}

void AArch64LiftingStreamer::switchSection(MCSection *Section, uint32_t Subsection) {
  if (Section->isText()) {
    MCBBB.NextInstrIsFuncStart = true;
  }
  MCStreamer::switchSection(Section, Subsection);
  Streamer->switchSection(Section, Subsection);
}

void AArch64LiftingStreamer::finishImpl() {
  // We call finish() and not finishImpl() on the streamer because
  // MCFunction has things to do during its finish(). We want the other streamer
  // to properly finish emitting, so we call finish() and not finishImpl().
  finalize(*MII);
  Streamer->finish();
}

// Determines the basic block leaders.
void MCBasicBlockBuilder::determineLeaders() {
  // Go through all instructions to determine leaders.
  for (unsigned i = 0; i < Objects.size(); ++i) {
    if (!Objects[i]->isInst()) 
      continue;
    
    MCInstContainer &MCIC = cast<MCInstContainer>(*Objects[i]);
    unsigned OpNum = isBranch(MCIC.Inst);
    if (OpNum != 0xDEAD) {
      // TODO: This could break
      // Insert the statement immediately succeeding the branch.
      // Make sure there is actually an instruction after.
      bool InstPresentAfter = false;
      for (unsigned j = i + 1; j < Objects.size() && !InstPresentAfter; ++j) {
        if (Objects[j]->isInst()) {
          InstPresentAfter = true;
        }
      }
      // Note: there is a case where the last instruction of a "function"
      // ends up being a branch. This actually works out, because the next
      // instruction (if existing) would be the start of a function, which
      // is already a leader.
      if (InstPresentAfter) {
        Objects[i + 1]->IsLeader = true;
      }


      // Insert the target of the branch.
      // Note: assumption is that we are branching to a label. It may
      // be possible to branch to an absolute value?
      StringRef Target = getTarget(MCIC.Inst);
      Objects[LabelLocs[Target]]->IsLeader = true;
    }
  }
}

void MCBasicBlockBuilder::createBasicBlocks() {
  // Go from one leader to another, creating basic blocks
  // The first object is a leader.
  unsigned BlockNum = 0;
  std::unique_ptr<MCBasicBlock> CurBlock = std::make_unique<MCBasicBlock>(BlockNum);
  // Edge case: include everything up to the first instruction in this basic block.
  //            This is because there may be data and labels that are not marked as
  //            "leaders".
  unsigned i = 0;
  while (i < Objects.size() && !Objects[i]->IsLeader) {
    Objects[i]->BlockNum = BlockNum;
    if (Objects[i]->isInst() && Objects[i]->IsFuncStart) {
      CurBlock->IsStartOfFunc = true;
    } else if (Objects[i]->isLabel()) {
      StringRef Name = cast<MCLabelContainer>(*Objects[i]).Symbol->getName();
      LabelLocs[Name] = BlockNum;
    }
    CurBlock->Objects.push_back(std::move(Objects[i++]));
  }
  // Add the leading instruction itself.
  if (i < Objects.size()) {
    if (Objects[i]->isInst() && Objects[i]->IsFuncStart) {
      CurBlock->IsStartOfFunc = true;
    } else if (Objects[i]->isLabel()) {
      StringRef Name = cast<MCLabelContainer>(*Objects[i]).Symbol->getName();
      LabelLocs[Name] = BlockNum;
    }
    CurBlock->Objects.push_back(std::move(Objects[i++]));
  }

  // Continue as needed.
  for (; i < Objects.size(); ++i) {
    // Do we need to change basic blocks?
    if (Objects[i]->IsLeader) {
      Blocks.push_back(std::move(CurBlock));
      CurBlock = std::make_unique<MCBasicBlock>(++BlockNum);
    }
    if (Objects[i]->isInst() && Objects[i]->IsFuncStart) {
      CurBlock->IsStartOfFunc = true;
    } else if (Objects[i]->isLabel()) {
      StringRef Name = cast<MCLabelContainer>(*Objects[i]).Symbol->getName();
      LabelLocs[Name] = BlockNum;
    }
    Objects[i]->BlockNum = BlockNum;
    CurBlock->Objects.push_back(std::move(Objects[i]));
  }
  // Push the last block on.
  Blocks.push_back(std::move(CurBlock));

  // debugging
  LLVM_DEBUG(dbgs() << "Blocks: \n");
  for (auto &B : Blocks) {
    LLVM_DEBUG(B->dump(dbgs()));
  }
}

// fills out ancestor and successor information for basic blocks.
void MCBasicBlockBuilder::controlFlowAnalysis() {
  // The last instruction of a basic block must be either a change of flow
  // (branch) or implies sequential flow to the next block.
  for (unsigned i = 0; i < Blocks.size(); ++i) {
    MCBasicBlock &CurrentBlock = *Blocks[i];

    // Check if we have an unconditional branch
    // Find the last instruction. There could be data blocks and labels in the way.
    auto Itr = CurrentBlock.Objects.rbegin();
    while (Itr != CurrentBlock.Objects.rend() && !(*Itr)->isInst()) {
      LLVM_DEBUG(dbgs() << "\t" << **Itr << "\n");
      ++Itr;
    }
    if (Itr == CurrentBlock.Objects.rend())
      // No instructions in this. Could be data blocks.
      continue;
    MCInst &LastInst = cast<MCInstContainer>(**Itr).Inst;
    unsigned OpNum = isBranch(LastInst);
    if (OpNum != 0xDEAD) {
      // Link to target
      StringRef TargetLabel = getTarget(LastInst);
      unsigned TargetBlockIdx = LabelLocs[TargetLabel];
      auto &TargetBlock = Blocks[TargetBlockIdx];
      CurrentBlock.Successors.insert(TargetBlock.get());
      TargetBlock->Ancestors.insert(&CurrentBlock);

      // If conditional branch, we also link to its fallthrough.
      if (isCondBranchOpcode(LastInst.getOpcode())) {
        if (!isRet(LastInst) && i + 1 < Blocks.size() &&
            !Blocks[i + 1]->IsStartOfFunc) {
          auto &Fallthrough = Blocks[i + 1];
          CurrentBlock.Successors.insert(Fallthrough.get());
          Fallthrough->Ancestors.insert(&CurrentBlock);
        }
      }
    } else {
      // This must fall through to the next block.
      if (!isRet(LastInst) && i + 1 < Blocks.size() &&
          !Blocks[i + 1]->IsStartOfFunc) {
        auto &Fallthrough = Blocks[i + 1];
        CurrentBlock.Successors.insert(Fallthrough.get());
        Fallthrough->Ancestors.insert(&CurrentBlock);
      }
    }
  }
}

// TODO: rename this to something better. Def/use analysis?
// TODO: could this be moved to createBasicBlocks?
// TODO: X30 - will have interferences either way,
//             do we still want to disallow explicitly?
void MCBasicBlockBuilder::defUseAnalysis(const MCInstrInfo &MII) {
  const MCRegisterInfo *MRI = Ctx->getRegisterInfo();

  for (auto &MCBB : Blocks) {
    // Reset the uses and defs.
    MCBB->Defs = std::set<MCRegUnit>();
    MCBB->Uses = std::set<MCRegUnit>();
    for (const auto &I : MCBB->Objects) {
      if (!I->isInst()) 
        continue;
      const MCInst &Inst = cast<MCInstContainer>(*I).Inst;
      unsigned NumDefs = MII.get(Inst.getOpcode()).getNumDefs();
      for (unsigned i = NumDefs; i < Inst.getNumOperands(); ++i) {
        const MCOperand &MCO = Inst.getOperand(i);
        if (MCO.isReg()) {
          MCRegister Reg = MCO.getReg();

          // Don't falsely tag XZR.
          if (isSpecialReg(Reg)) continue;

          // If there is a def, this is not a true use.
          if (isVirtual(Reg)) {
            if (MCBB->Defs.find(Reg) == MCBB->Defs.end()) {
              MCBB->Uses.insert(Reg);
            }
          } else {
            auto RegUnits = MRI->regunits(Reg);
            auto B = RegUnits.begin(), E = RegUnits.end();
            while (B != E) {
              ++RegFrequencies[*B];
              if (MCBB->Defs.find(*B) == MCBB->Defs.end()) {
                MCBB->Uses.insert(*B);
              }
              ++B;
            }
          }
        }
      }
      for (unsigned i = 0; i < NumDefs; ++i) {
        const MCOperand &MCO = Inst.getOperand(i);
        if (MCO.isReg()) {
          MCRegister Reg = MCO.getReg();

          // Don't falsely tag XZR.
          if (isSpecialReg(Reg)) continue;

          if (isVirtual(Reg)) {
            MCBB->Defs.insert(Reg);
          } else {
            auto RegUnits = MRI->regunits(Reg);
            MCBB->Defs.insert(RegUnits.begin(), RegUnits.end());
            for (auto I = RegUnits.begin(); I != RegUnits.end(); ++I) {
              ++RegFrequencies[*I];
            }
          }
        }
      }
    }
  }
}

// void MCBasicBlock::replaceReg(MCRegister Old, MCRegister New, unsigned Pos, const MCInstrInfo &MII, const MCRegisterInfo *MRI, bool Defined) {
//   bool Redefined = false;
//   for (unsigned i = Pos; i < Objects.size() && !Redefined; ++i) {
//     if (!Objects[i]->isInst())
//       continue;
    
//     MCInstContainer *IC = cast<MCInstContainer>(Objects[i].get());
//     if (uses(IC->Inst, Old, MII)) {
//       unsigned NumDefs = MII.get(IC->Inst.getOpcode()).getNumDefs();
//       for (unsigned j = NumDefs; j < IC->Inst.getNumOperands(); ++j) {
//         MCOperand &O = IC->Inst.getOperand(j);
//         if (O.isReg() && regsEqual(O.getReg(), Old, MRI))
//           *(IC->Inst.begin() + j) = MCOperand::createReg(New);
//       }
//     }

//     Redefined = defines(IC->Inst, Old, MII).has_value();
//   }
  
//   if (!Redefined) {
//     for (auto &B : Successors) {
//       // Avoid circular dependencies.
//       if (B->BlockNum != BlockNum)
//         B->replaceReg(Old, New, 0, MII, MRI, false);
//     }
//   } else {
//     // this is the death of this variable. Traverse upward and correct as well
//     if (!Defined) {
//       for (auto &B : Ancestors) {
//         if (B->BlockNum != BlockNum)
//           B->backwardsReplaceReg(Old, New, MII, MRI);
//       }
//     }
//   }
// }


// void MCBasicBlock::backwardsReplaceReg(MCRegister Old, MCRegister New, const MCInstrInfo &MII, const MCRegisterInfo *MRI) {
//   // Go all the way through, unless the register is defined.
//   for (auto &O : Objects) {
//     dbgs() << *O << "\n";
//   }
//   for (auto Itr = Objects.rbegin(); Itr != Objects.rend(); ++Itr) {
//     auto &Object = *Itr;
//     if (!Object->isInst())
//       continue;
    
//     MCInstContainer *IC = cast<MCInstContainer>(Object.get());
    
//     if (defines(IC->Inst, Old, MII)) {
//       return;
//     }

//     if (uses(IC->Inst, Old, MII)) {
//       unsigned NumDefs = MII.get(IC->Inst.getOpcode()).getNumDefs();
//       for (unsigned j = NumDefs; j < IC->Inst.getNumOperands(); ++j) {
//         MCOperand &O = IC->Inst.getOperand(j);
//         if (O.isReg() && regsEqual(O.getReg(), Old, MRI))
//           *(IC->Inst.begin() + j) = MCOperand::createReg(New);
//       }
//     }
//   }
//   // If we reach here, there could be more defs in previous blocks..
//   for (auto &B : Ancestors) {
//     if (B->BlockNum != BlockNum)
//       B->backwardsReplaceReg(Old, New, MII, MRI);
//   }
  
// }

// void MCBasicBlockBuilder::temporarify(const MCInstrInfo &MII) {
//   // replace all vector registers with virtual temporaries.
//   const MCRegisterInfo *MRI = Ctx->getRegisterInfo();
//   for (auto &B: Blocks) {
//     for (unsigned i = 0; i < B->Objects.size(); ++i) {
//       MCObject *O = B->Objects[i].get();
//       if (!O->isInst())
//         continue;

//       MCInstContainer *InstContainer = cast<MCInstContainer>(O);
//       MCInst *Inst = &InstContainer->Inst;
//       unsigned NumDefs = MII.get(Inst->getOpcode()).getNumDefs();
//       for (unsigned j = 0; j < NumDefs; ++j) {
//         MCOperand O = Inst->getOperand(j);
//         if (!O.isReg())
//           continue;

//         MCRegister R = O.getReg();
//         if (isVectorReg(R)) {
//           *(Inst->begin() + j) = MCOperand::createReg(NextVirtualVectorReg);
//           B->replaceReg(R, NextVirtualVectorReg, i + 1, MII, MRI);
//           NextVirtualVectorReg++;
//         }
//       }
//     }
//   }
//   dbgs() << "code after temporarifying: \n";
//   for (auto &B : Blocks) {
//     for (auto &I : B->Objects) {
//       if (!I->isInst())
//         continue;

//       MCInst &Inst = cast<MCInstContainer>(*I).Inst;
//       dbgs() << MII.getName(Inst.getOpcode()) << " ";
//       for (unsigned i = 0; i < Inst.getNumOperands() - 1; ++i) {
//         if (Inst.getOperand(i).isReg()) {
//           dbgs() << getRegName(Inst.getOperand(i).getReg(), MRI) << ", ";
//         } else {
//           dbgs() << Inst.getOperand(i) << ", ";
//         }
//       }
//       if (Inst.getOperand(Inst.getNumOperands() - 1).isReg()) {
//         dbgs() << getRegName(Inst.getOperand(Inst.getNumOperands() - 1).getReg(), MRI) << ", ";
//       } else {
//         dbgs() << Inst.getOperand(Inst.getNumOperands() - 1) << ", ";
//       }
//       dbgs() << "\n";
//     }
//   }
// }

void MCBasicBlockBuilder::livenessAnalysis() {
  // Reset sets to null.
  for (auto &MCBB : Blocks) {
    MCBB->LiveIn = std::set<MCRegUnit>();
    MCBB->LiveOut = std::set<MCRegUnit>();
  }
  bool Changed = false; // by default, assume no change.
  // keep going until the sets are the same.
  do {
    Changed = false;
    for (auto &MCBB : Blocks) {
      const std::set<MCRegUnit> PrevIn = MCBB->LiveIn;
      const std::set<MCRegUnit> PrevOut = MCBB->LiveOut;

      MCBB->LiveOut = std::set<MCRegUnit>();

      // liveout = U (all successors) livein[s]
      for (auto *Successor : MCBB->Successors) {
        MCBB->LiveOut.insert(Successor->LiveIn.begin(), Successor->LiveIn.end());
      }

      MCBB->LiveIn = std::set<MCRegUnit>();

      // livein = liveout - def U uses
      for (MCRegUnit R : PrevOut) {
        if (MCBB->Defs.find(R) == MCBB->Defs.end()) {
          MCBB->LiveIn.insert(R);
        }
      }
      // add all uses
      for (MCRegUnit R : MCBB->Uses) {
        MCBB->LiveIn.insert(R);
      }

      // if livein or liveout changed, mark Changed as true.
      Changed |= PrevIn != MCBB->LiveIn || PrevOut != MCBB->LiveOut;
    }
  } while (Changed);

  // debugging
  LLVM_DEBUG(dbgs() << "Liveness information:\n");
  for (auto &MCBB : Blocks) {
    LLVM_DEBUG(dbgs() << "Basic Block " << MCBB->BlockNum << ":\n");
    LLVM_DEBUG(dbgs() << "\tLive In: ");
    for (MCRegUnit R : MCBB->LiveIn) {
      LLVM_DEBUG(dbgs() << getRegUnitName(R, Ctx->getRegisterInfo()) << " ");
    }

    LLVM_DEBUG(dbgs() << "\n\tLive Out: ");
    for (MCRegUnit R : MCBB->LiveOut) {
      LLVM_DEBUG(dbgs() << getRegUnitName(R, Ctx->getRegisterInfo()) << " ");
    }
    LLVM_DEBUG(dbgs() << "\n");
  }
}
void InterferenceGraph::dump(raw_ostream &OS, const MCRegisterInfo *MRI) {
  // debugging
  OS << "Map: " << "\n";
  for (auto Node : Nodes) {
    OS << "\t";
    OS << getRegUnitName(Node.first, MRI);
    OS << "\n\t\tConnected to:";
    for (auto OtherUnit : Node.second) {
      OS << getRegUnitName(OtherUnit, MRI) << " ";
    }
    OS << "\n";
  }
  OS << "\n";
}

void MCBasicBlockBuilder::constructInterferenceGraph(const MCInstrInfo &MII) {
  const MCRegisterInfo *MRI = Ctx->getRegisterInfo();

  // First, reset both graphs.
  GPRGraph = InterferenceGraph();
  VectorGraph = InterferenceGraph();

  // for each instruction in a block, granularly determine interference
  for (auto &MCBB : Blocks) {
    if (MCBB->Objects.size() == 0)
      continue;

    // bottom up - start with the current interference list
    std::list<std::set<MCRegUnit>> LiveVariables;
    // first, pass through and generate the live variable sets.
    LiveVariables.push_front(
        std::set<MCRegUnit>(MCBB->LiveOut.begin(), MCBB->LiveOut.end()));
    for (auto Itr = MCBB->Objects.rbegin(); Itr != MCBB->Objects.rend();
         ++Itr) {
      if (!(*Itr)->isInst())
        continue;
      std::set<MCRegUnit> NewLiveVariables = LiveVariables.front();
      // remove all defs that this instruction has.
      MCInst &Inst = cast<MCInstContainer>(**Itr).Inst;
      unsigned NumDefs = MII.get(Inst.getOpcode()).getNumDefs();
      for (unsigned i = 0; i < NumDefs; ++i) {
        const MCOperand &Operand = Inst.getOperand(i);
        if (Operand.isReg()) {
          MCRegister Reg = Operand.getReg();
          // XZR can be falsely tagged as a use.
          if (isSpecialReg(Reg)) continue;

          if (isVirtual(Reg)) {
            NewLiveVariables.erase(Reg);
          } else {
            auto RegUnits = MRI->regunits(Reg);
            for (auto Itr = RegUnits.begin(); Itr != RegUnits.end(); ++Itr) {
              NewLiveVariables.erase(*Itr);
            }
          }
        }
      }

      // add all uses that this instruction has.
      for (unsigned i = NumDefs; i < Inst.getNumOperands(); ++i) {
        const MCOperand &Operand = Inst.getOperand(i);
        if (Operand.isReg()) {
          MCRegister Reg = Operand.getReg();
          // XZR can be falsely tagged as a use.
          if (isSpecialReg(Reg)) continue;

          if (isVirtual(Reg)) {
            NewLiveVariables.insert(Reg);
          } else {
            auto RegUnits = MRI->regunits(Reg);
            for (auto Itr = RegUnits.begin(); Itr != RegUnits.end(); ++Itr) {
              NewLiveVariables.insert(*Itr);
            }
          }
        }
      }
      LiveVariables.push_front(NewLiveVariables);
    }

    // now, we simply add edges to our graph per the liveness info.
    // TODO: handle move instructions

    // we need to maintain 2 interference graphs: one for vector registers,
    // one for general purpose registers.
    auto OutSetItr = LiveVariables.begin();
    // Connect the variables in the in set.
    for (MCRegUnit Reg : *OutSetItr) {
      bool ThisIsVecReg = isVectorReg(Reg, MRI) || isVirtualVector(Reg);
      bool ThisIsGPR = isGPR(Reg, MRI) || isVirtualGPR(Reg);
      if (ThisIsGPR)
        GPRGraph.addNode(Reg);

      if (ThisIsVecReg)
        VectorGraph.addNode(Reg);

      for (MCRegUnit Other : *OutSetItr) {
        bool OtherIsVecReg = isVectorReg(Other, MRI) || isVirtualVector(Other);
        bool OtherIsGPR = isGPR(Other, MRI) || isVirtualGPR(Other);

        if (OtherIsGPR)
          GPRGraph.addNode(Other);
          
        if (OtherIsVecReg)
          VectorGraph.addNode(Other);

        if (ThisIsVecReg && OtherIsVecReg)
          VectorGraph.addEdge(Reg, Other);
        else if (ThisIsGPR && OtherIsGPR)
          GPRGraph.addEdge(Reg, Other);
        // else, no interference
      }
    }

    ++OutSetItr;
    unsigned i = 0;
    for (auto &I : MCBB->Objects) {
      if (!I->isInst())
        continue;

      MCInst &Inst = cast<MCInstContainer>(*I).Inst;
      // Calculate the dead set for this instruction.
      MCBB->Dead.push_back(std::set<MCRegUnit>());
      std::set<MCRegUnit> &CurrentDeadSet = MCBB->Dead[i++]; 
      std::set<MCRegUnit> &PreviousOut = *(--OutSetItr);
      std::set<MCRegUnit> &CurrentOut = *(++OutSetItr);
      // Calculate the dead set for this instruction.
      for (MCRegUnit R : PreviousOut) {
        if (!CurrentOut.count(R)) {
          CurrentDeadSet.insert(R);
        }
      }

      // for all definitions, we add an edge to the out of the instrution.
      unsigned NumDefs = MII.get(Inst.getOpcode()).getNumDefs();
      for (unsigned i = 0; i < NumDefs; ++i) {
        const MCOperand &Operand = Inst.getOperand(i);
        if (Operand.isReg()) {
          MCRegister Reg = Operand.getReg();
          // XZR can be falsely tagged as a use.
          if (isSpecialReg(Reg)) continue;

          // add an edge between each regunit and the out set.
          if (isVirtual(Reg)) {
            bool ThisIsVecReg = isVirtualVector(Reg);
            bool ThisIsGPR = isVirtualGPR(Reg);
            if (ThisIsGPR)
              GPRGraph.addNode(Reg);

            if (ThisIsVecReg)
              VectorGraph.addNode(Reg);

            for (MCRegUnit Other : CurrentOut) {
              bool OtherIsVecReg = isVectorReg(Other, MRI) || isVirtualVector(Other);
              bool OtherIsGPR = isGPR(Other, MRI) || isVirtualGPR(Other);

              if (OtherIsGPR)
                GPRGraph.addNode(Other);
                
              if (OtherIsVecReg)
                VectorGraph.addNode(Other);

              if (ThisIsVecReg && OtherIsVecReg)
                VectorGraph.addEdge(Reg, Other);
              else if (ThisIsGPR && OtherIsGPR)
                GPRGraph.addEdge(Reg, Other);
              // else, no interference
            }
          } else {
              auto RegUnits = MRI->regunits(Reg);
              for (auto Itr = RegUnits.begin(); Itr != RegUnits.end(); ++Itr) {
                MCRegUnit MCRU = *Itr;
                bool ThisIsVecReg = isVectorReg(MCRU, MRI);
                bool ThisIsGPR = isGPR(MCRU, MRI);

                if (ThisIsGPR)
                  GPRGraph.addNode(MCRU);

                if (ThisIsVecReg)
                  VectorGraph.addNode(MCRU);

                for (MCRegUnit Other : *OutSetItr) {
                  bool OtherIsVecReg = isVectorReg(Other, MRI) || isVirtualVector(Other);
                  bool OtherIsGPR = isGPR(Other, MRI) || isVirtualGPR(Other);

                  if (OtherIsGPR)
                    GPRGraph.addNode(Other);
                    
                  if (OtherIsVecReg)
                    VectorGraph.addNode(Other);

                  if (ThisIsVecReg && OtherIsVecReg)
                    VectorGraph.addEdge(MCRU, Other);
                  else if (ThisIsGPR && OtherIsGPR)
                    GPRGraph.addEdge(MCRU, Other);
                  // else, no interference
                }
              }
          }
        }
      }
      ++OutSetItr;
    }
    --OutSetItr;
    // Connect the variables in the in set.
    for (MCRegUnit Reg : *OutSetItr) {
      bool ThisIsVecReg = isVectorReg(Reg, MRI) || isVirtualVector(Reg);
      bool ThisIsGPR = isGPR(Reg, MRI) || isVirtualGPR(Reg);
      if (ThisIsGPR)
        GPRGraph.addNode(Reg);

      if (ThisIsVecReg)
        VectorGraph.addNode(Reg);

      for (MCRegUnit Other : *OutSetItr) {
        bool OtherIsVecReg = isVectorReg(Other, MRI) || isVirtualVector(Reg);
        bool OtherIsGPR = isGPR(Other, MRI) || isVirtualGPR(Reg);

        if (OtherIsGPR)
          GPRGraph.addNode(Other);
          
        if (OtherIsVecReg)
          VectorGraph.addNode(Other);

        if (ThisIsVecReg && OtherIsVecReg)
          VectorGraph.addEdge(Reg, Other);
        else if (ThisIsGPR && OtherIsGPR)
          GPRGraph.addEdge(Reg, Other);
        // else, no interference
      }
    }
  }
  // dbgs() << "Dead Sets:\n";
  // for (auto &B : Blocks) {
  //   for (unsigned i = 0; i < B->Objects.size(); ++i) {
  //     if (!B->Objects[i]->isInst())
  //       continue;
  //     MCInst &Inst = cast<MCInstContainer>(*B->Objects[i]).Inst;
  //     dbgs() << "Instruction " << Inst << ": \n";
  //     dbgs() << "\tDead: ";
  //     for (auto D : B->Dead[i]) {
  //       dbgs() << getRegUnitName(D, MRI) << " ";
  //     }
  //     dbgs() << "\n";
  //   }
  // }
  LLVM_DEBUG(dbgs() << "GPR: \n");
  LLVM_DEBUG(GPRGraph.dump(dbgs(), MRI));
  LLVM_DEBUG(dbgs() << "Vector: \n");
  LLVM_DEBUG(VectorGraph.dump(dbgs(), MRI));
}

// If there are any pre-colored registers, they must be colored using
// InterferenceGraph::assignColor() before this, otherwise they will be
// treated as uncolored.
MCRegUnit InterferenceGraph::color(std::set<unsigned> &AvailableColors,
                                   std::map<MCRegUnit, uint64_t> &RegFrequencies,
                              const MCRegisterInfo *MRI) {
  // debugging
  LLVM_DEBUG(dbgs() << "Initial Colors:\n");
  for (auto R : Colors) {
    LLVM_DEBUG(dbgs() << "\t" << getRegUnitName(R.first, MRI) << ": " << getRegUnitName(R.second, MRI) << "\n");
  }

  unsigned K = AvailableColors.size();
  std::stack<Node> NodesToColor;

  bool Changed = true;
  while (Changed) {
    SmallVector<MCRegUnit> ToRemove;
    Changed = false;
    // Remove a node with no color.
    for (Node N : Nodes) {
      MCRegUnit R = N.first;
      if (!Colors.count(R) && N.second.size() < K) {
        NodesToColor.push(N);
        ToRemove.push_back(R);
        Changed = true;
      }
    }
    // To avoid co-modification.
    for (auto R : ToRemove) {
      removeNode(R);
    }
  }

  // Attempt to color the graph.
  while (!NodesToColor.empty()) {
    Node N = NodesToColor.top();
    NodesToColor.pop();
    // Restore edges.
    for (MCRegUnit Other : N.second) {
      addEdge(N.first, Other);
    }
    // Assign a color that is not present.
    for (unsigned Reg : AvailableColors) {
      unsigned Color = getRegUnit(Reg, MRI);
      bool ColorAvailable = true;
      for (MCRegUnit Other : N.second) {
        if (Colors.count(Other) && Colors[Other] == Color) {
          ColorAvailable = false;
          break;
        }
      }

      if (ColorAvailable) {
        assignColor(N.first, Color);
        break;
      }
    }
  }



  // If a register in the graph still has no color,
  // we must spill.
  for (Node N : Nodes) {
    MCRegUnit R = N.first;
    if (!Colors.count(R)) {
      // Try one last time to find a color.
      for (unsigned Reg : AvailableColors) {
        unsigned Color = getRegUnit(Reg, MRI);
        bool ColorAvailable = true;
        for (MCRegUnit Other : N.second) {
          if (Colors.count(Other) && Colors[Other] == Color) {
            ColorAvailable = false;
            break;
          }
        }

        if (ColorAvailable) {
          assignColor(N.first, Color);
          continue;
        }
      }
      
      // If we reach here, we must spill.
      LLVM_DEBUG(dbgs() << "Could not find an allocation.\n");
      // Restore the interference graph.
      while (!NodesToColor.empty()) {
        Node Cur = NodesToColor.top();
        NodesToColor.pop();
        // Restore the edges in each list.
        for (MCRegUnit Other : Cur.second) {
          addEdge(Cur.first, Other);
        }
      }
      // Decide on a register to spill.
      const double DblMax = std::numeric_limits<double>::max();
      std::pair<double, MCRegUnit> Max = std::make_pair(DblMax, 0);
      for (Node N : Nodes) {
        // Do not attempt to spill virtual registers.
        if (isVirtual(N.first) || getRegFromUnit(N.first, MRI) == AArch64::X27) continue;
        double Cost = RegFrequencies[N.first] / (double) N.second.size();
        if (Cost < Max.first) {
          Max = std::make_pair(Cost, N.first);
        }
      }
      assert(Max.first != DblMax && "Could not find a used register!");
      return Max.second;
    }
  }

  // debugging
  LLVM_DEBUG(dbgs() << "Colors:\n");
  for (auto R : Colors) {
    LLVM_DEBUG(dbgs() << "\t" << getRegUnitName(R.first, MRI) << ": " << getRegUnitName(R.second, MRI) << "\n");
  }

  // don't bother keeping information about registers that haven't changed.
  for (auto Itr = Colors.begin(); Itr != Colors.end();) {
    auto ColorPair = *Itr;
    if (ColorPair.first == ColorPair.second) {
      Colors.erase(Itr++);
    } else {
      ++Itr;
    }
  }
  return AArch64::FFR;
}

void MCBasicBlockBuilder::dumpCode(const MCInstrInfo &MII, const MCRegisterInfo *MRI) {
  for (auto &B : Blocks) {
    for (auto &I : B->Objects) {
      if (!I->isInst())
        continue;
      MCInst &Inst = cast<MCInstContainer>(*I).Inst;
      dbgs() << MII.getName(Inst.getOpcode()) << " ";
      for (unsigned i = 0; i < Inst.getNumOperands() - 1; ++i) {
        if (Inst.getOperand(i).isReg()) {
          dbgs() << getRegName(Inst.getOperand(i).getReg(), MRI) << ", ";
        } else {
          dbgs() << Inst.getOperand(i) << ", ";
        }
      }
      if (Inst.getOperand(Inst.getNumOperands() - 1).isReg()) {
        dbgs() << getRegName(Inst.getOperand(Inst.getNumOperands() - 1).getReg(), MRI) << ", ";
      } else {
        dbgs() << Inst.getOperand(Inst.getNumOperands() - 1);
      }
      dbgs() << "\n";
    }
  }
}
void MCBasicBlockBuilder::spill(MCRegUnit MCRU, const MCInstrInfo &MII, std::set<MCRegister> RegsToPreserve) {
  ++Spills;
  bool Preserved = RegsToPreserve.count(getRegFromUnit(MCRU, Ctx->getRegisterInfo()));
  bool Redefined = false;
  bool Stored = false;

  // assign each load/use a new temporary. 
  const MCRegisterInfo *MRI = Ctx->getRegisterInfo();
  MCRegister Target = isVirtual(MCRU) ? static_cast<MCRegister>(MCRU) : getRegFromUnit(MCRU, MRI);
  bool Loaded = false;
  for (auto &MCBB : Blocks) {
    for (unsigned i = 0; i < MCBB->Objects.size(); ++i) {
      if (!MCBB->Objects[i]->isInst())
        continue;

      MCInstContainer *InstContainer = cast<MCInstContainer>(MCBB->Objects[i].get());
      MCInst &Inst = InstContainer->Inst;

      unsigned NumDefs = MII.get(Inst.getOpcode()).getNumDefs();
      unsigned NumOperands = Inst.getNumOperands();
      for (unsigned j = NumDefs; j < NumOperands; ++j) {
        MCInstContainer *InstContainer = cast<MCInstContainer>(MCBB->Objects[i].get());
        Inst = cast<MCInstContainer>(*MCBB->Objects[i]).Inst;
        MCOperand O = Inst.getOperand(j);
        if (O.isReg() && regsEqual(Target, O.getReg(), MRI)) {
          // If this is the first load, make sure to store it into its reg slot.
          if (!Stored) {
            // TODO: store
            Stored = true;
          }
          // If this is a function start, skip this load if we haven't 
          // redefined this preserved register. This must be a parameter 
          // and the value is there.
          // FIXME: a much better solution is to follow the control flow 
          //        from the start of the function, keeping the register
          //        live until it truly dies.
          if (Preserved && MCBB->IsStartOfFunc && !Redefined)
            continue;

          unsigned NextVReg = 0;
          bool ThisIsGPR = isGPR(O.getReg());
          bool ThisIsVector = isVectorReg(O.getReg());
          // TODO: fix
          // TODO: clean up
          if (!(ThisIsGPR || ThisIsVector)) {
            LLVM_DEBUG(dbgs() << "WARNING: IGNORING UNIQUE REGTYPE\n");
          }
          MCInst Load;
          if (!Loaded) {
            if (ThisIsGPR) {
              loadGPR(Load, ++NextVirtualGPR, AArch64::X27);
            } else {
              loadVec(Load, ++NextVirtualVectorReg, AArch64::X27);
            }
            // Insert a load before the use.
            MCBB->Objects.insert(MCBB->Objects.begin() + i, std::make_unique<MCInstContainer>(Load, InstContainer->SectionPair, true));
            // insert a placebo empty set in dead sets.
            MCBB->Dead.insert(MCBB->Dead.begin() + i, std::set<MCRegUnit>());
            ++LoadsInserted; 
            // Skip the old instruction.
            ++i;
            Loaded = false;
          }
          if (isGPR(O.getReg())) {
            NextVReg = NextVirtualGPR;
          }
          else {
            NextVReg = NextVirtualVectorReg;
          }

          Inst = cast<MCInstContainer>(*MCBB->Objects[i]).Inst;
          // Make the operand the same as the temporary.
          *(Inst.begin() + j) = MCOperand::createReg(NextVReg);
        }
      }
      for (unsigned j = 0; j < NumDefs; ++j) {
        MCInstContainer *InstContainer = cast<MCInstContainer>(MCBB->Objects[i].get());
        Inst = InstContainer->Inst;
        MCOperand O = Inst.getOperand(j);
        if (O.isReg() && regsEqual(O.getReg(), Target, MRI)) {
          // Mark this reg as redefined.
          if (Preserved)
            Redefined = true;

          MCInst Store;
          unsigned NextVReg;
          if (isGPR(O.getReg())) {
            NextVReg = ++NextVirtualGPR;
            storeGPR(Store, NextVReg, AArch64::SP);
          }
          else {
            NextVReg = ++NextVirtualVectorReg;
            storeVec(Store, NextVReg, AArch64::SP);
          }
          // Make the operand the same as the temporary.
          *(Inst.begin() + j) = MCOperand::createReg(NextVReg);

          // Insert a store after the def.
          MCBB->Objects.insert(MCBB->Objects.begin() + i + 1, std::make_unique<MCInstContainer>(Store, InstContainer->SectionPair, true));
          MCBB->Dead.insert(MCBB->Dead.begin() + i, std::set<MCRegUnit>());
          ++StoresInserted;
          // skip the new instruction.
          ++i;
          // if we've stored, this register is no longer in use.
          // indicate that the value needs to be loaded again.
          Loaded = false;
        }
      }
    }
  }
  LLVM_DEBUG(dumpCode(MII, MRI));
}

void InterferenceGraph::precolor(std::set<MCRegUnit> Disallowed, const MCRegisterInfo *MRI) {
  for (Node N : Nodes) {
    if (!Disallowed.count(N.first) && !isVirtual(N.first)) {
      assignColor(N.first, N.first);
    }
  }
}

// Attempt to color the interference graph. Two possible approaches:
// 1. leave allowed registers as their existing color, and attempt
//    to color the graph after assigning colors to viable registers
// 2. clear all colors and recolor the graph ourselves
void MCBasicBlockBuilder::colorGraphs(const MCInstrInfo &MII) {
  const MCRegisterInfo *MRI = Ctx->getRegisterInfo();
  // All allowed registers are already in the graph.
  // Assign a color to them.

  // for the GPR graph, K is 25
  // for the vector graph, K is 16

  // For GPRs, we will only consider the disallowed registers.
  // 1. remove disallowed registers from the graph and make sure
  //    the graph is still 25-colorable
  // 2. assign non-interfering colors to the disallowed registers
  //    and give up if we have to spill.
  // First, attempt to remove a node with degree less than K
  std::set<MCRegUnit> Disallowed;

  // The graph coloring algorithm expects these to be as registers,
  // not register units.
  std::set<unsigned> GPRColors = {
    AArch64::X0,
    AArch64::X1,
    AArch64::X2,
    AArch64::X3,
    AArch64::X4,
    AArch64::X5,
    AArch64::X6,
    AArch64::X7,
    AArch64::X8,
    AArch64::X9,
    AArch64::X10,
    AArch64::X11,
    AArch64::X12,
    AArch64::X15,
    AArch64::X16,
    AArch64::X17,
    AArch64::X18,
    AArch64::X19,
    AArch64::X20,
    AArch64::X21,
    AArch64::X22,
    AArch64::X25,
    AArch64::X26
  };

  std::set<unsigned> VecColors = {
    AArch64::Q0,
    AArch64::Q1,
    AArch64::Q2,
    AArch64::Q3,
    AArch64::Q4,
    AArch64::Q5,
    AArch64::Q6,
    AArch64::Q7,
    AArch64::Q8,
    AArch64::Q9,
    AArch64::Q10,
    AArch64::Q11,
    AArch64::Q12,
    AArch64::Q13,
    AArch64::Q14,
    AArch64::Q15
  };

  std::set<MCRegister> GPRParameterRegs = {
    AArch64::X0,
    AArch64::X1, 
    AArch64::X2,
    AArch64::X3,
    AArch64::X4,
    AArch64::X5,
    AArch64::X6,
    AArch64::X7
  };

  Disallowed.insert(getRegUnit(AArch64::X13, MRI));
  Disallowed.insert(getRegUnit(AArch64::X14, MRI));
  Disallowed.insert(getRegUnit(AArch64::X23, MRI));
  Disallowed.insert(getRegUnit(AArch64::X24, MRI));
  Disallowed.insert(getRegUnit(AArch64::X28, MRI));
  if (Spills == 0) {
    // If we haven't spilled, we should use X27 as our scratchpad register.
    // We must rewrite instructions with X27.
    Disallowed.insert(getRegUnit(AArch64::X27, MRI));
  }

  // Pre-color the existing nodes.
  // FIXME: this is a little messy, relying on the regunit of x0 to
  //        assign the color. Maybe have a formal mapping
  GPRGraph.precolor(Disallowed, MRI);

  MCRegister Res;
  LLVM_DEBUG(dbgs() << "GPR:\n");
  Res = GPRGraph.color(GPRColors, RegFrequencies, MRI);

  // Keep rebuilding and recoloring until we have an acceptable one.
  while (Res != AArch64::FFR) { // spill whatever res was and rebuild liveness + recolor.
    // Pre-color the existing nodes.
    // FIXME: this is a little messy, relying on the regunit of x0 to
    //        assign the color. Maybe have a formal mapping
    LLVM_DEBUG(dbgs() << "spilling: " << getRegUnitName(Res, MRI) << "\n");

    spill(Res, MII, GPRParameterRegs);

    LLVM_DEBUG(dbgs() << "code after spill: \n");
    LLVM_DEBUG(dumpCode(MII, MRI));

    defUseAnalysis(MII);
    livenessAnalysis();
    constructInterferenceGraph(MII);
    GPRGraph.precolor(Disallowed, MRI);
    Res = GPRGraph.color(GPRColors, RegFrequencies, MRI);
  }

  LLVM_DEBUG(dbgs() << "Vector:\n");
  Res = VectorGraph.color(VecColors, RegFrequencies, MRI);
  while (Res != AArch64::FFR) { // spill whatever res was and rebuild liveness + recolor.
    // Pre-color the existing nodes.
    // FIXME: this is a little messy, relying on the regunit of x0 to
    //        assign the color. Maybe have a formal mapping
    LLVM_DEBUG(dbgs() << "spilling: " << getRegUnitName(Res, MRI) << "\n");
    spill(Res, MII, std::set<MCRegister>());

    LLVM_DEBUG(dbgs() << "code after spill: \n");
    LLVM_DEBUG(dumpCode(MII, MRI));

    defUseAnalysis(MII);
    livenessAnalysis();
    constructInterferenceGraph(MII);
    Res = VectorGraph.color(VecColors, RegFrequencies, MRI);
  }
}

void MCBasicBlockBuilder::rewriteRegs(const MCInstrInfo &MII) {
  const MCRegisterInfo *MRI = Ctx->getRegisterInfo();
  for (auto &Block : Blocks) {
    for (auto &I : Block->Objects) {
      if (!I->isInst())
        continue;
      MCInst &Inst = cast<MCInstContainer>(*I).Inst;
      ++TotalInstructions;
      bool Counted = false;
      for (auto *I = Inst.begin(); I != Inst.end(); ++I) {
        MCOperand Operand = *I;
        if (Operand.isReg()) {
          MCRegister Reg = Operand.getReg();
          MCRegister Rewritten = 0;
          if ((isGPR(Reg) || isVirtualGPR(Reg)) && GPRGraph.Colors.count(getRegUnit(Reg, MRI))) {
            Rewritten = GPRGraph.Colors[getRegUnit(Reg, MRI)];
          } else if ((isVectorReg(Reg) || isVirtualVector(Reg)) && VectorGraph.Colors.count(getRegUnit(Reg, MRI))) {
            Rewritten = VectorGraph.Colors[getRegUnit(Reg, MRI)];
          } else {
            continue;
          }
          Rewritten = getRegFromUnit(Rewritten, MRI);
          assert(Rewritten != 1 && "Unexpected empty color!");
          LLVM_DEBUG(dbgs() << "Rewriting " << getRegName(Reg, MRI) << " to " << getRegName(Rewritten, MRI) << "\n");
          *I = MCOperand::createReg(Rewritten);
          if (!Counted && !isVirtual(Reg)) {
            ++InstructionsRewritten;
            Counted = true;
          }
        }
      }
    }
  }
  LLVM_DEBUG(dbgs() << "code after rewrite: \n");
  LLVM_DEBUG(dumpCode(MII, MRI));
}

void MCBasicBlockBuilder::mapAndPatchSpills(const MCInstrInfo &MII) {
  if (Spills == 0)
    return;
  const MCRegisterInfo *MRI = Ctx->getRegisterInfo();
  MCSymbol *ScratchpadSym = Ctx->getOrCreateSymbol(Twine("reg_scratchpad"));
  const MCExpr *CPExpr = MCSymbolRefExpr::create(ScratchpadSym, *Ctx);
  // TODO: figure out how to emit constant pool things..
  // Allocate space for the labels.
  int64_t Offset = 16;
  // Now we have offsets. Patch them within basic blocks
  for (auto &B : Blocks) {
    // Insert the prologue at function starts.
    std::pair<MCSection *, uint32_t> SectionPair = std::make_pair(nullptr, 0);
    if (B->IsStartOfFunc) {
      // first, look for any instruction within the block to get the section.
      for (auto Itr = B->Objects.begin(); Itr != B->Objects.end() && SectionPair.first == nullptr; ++Itr) {
        if ((*Itr)->isInst()) {
          SectionPair = (*Itr)->SectionPair;
        }
      }
      if (SectionPair.first != nullptr) {
        MCInst Prologue;
        Prologue.setOpcode(AArch64::LDRXl);
        Prologue.addOperand(MCOperand::createReg(AArch64::X27));
        Prologue.addOperand(MCOperand::createExpr(CPExpr));
        B->Objects.insert(B->Objects.begin(), std::make_unique<MCInstContainer>(Prologue, SectionPair));
      }
    }
    for (unsigned InstNum = 0; InstNum < B->Objects.size(); ++InstNum) {
      auto &I = B->Objects[InstNum];
      if (!I->isInst())
        continue;
      MCInstContainer &MCI = cast<MCInstContainer>(*B->Objects[InstNum]);
      if (MCI.needsPatch()) {
        MCInst &Inst = MCI.Inst;
        assert(Inst.getOperand(0).isReg() && "unexpect non-reg operand!");
        MCRegister Reg = Inst.getOperand(0).getReg();
        if (!SpillsToOffsets.count(Reg)) {
          LLVM_DEBUG(dbgs() << "Assigning offset " << Offset << " to reg " << getRegName(Reg, MRI) << "\n");
          SpillsToOffsets[Reg] = Offset;
          Offset += 16; // TODO change if needed
        }
        int64_t Spill = SpillsToOffsets[Reg];
        // TODO might need to fix
        *(Inst.begin() + 2) = MCOperand::createImm(Spill);
        LLVM_DEBUG(dbgs() << "Patched Inst : " << Inst << "\n");
      }
    }
  }
  Blocks.back()->Objects.push_back(
    std::make_unique<MCLabelContainer>(ScratchpadSym, SMLoc(), 
      Blocks.back()->Objects.back()->SectionPair));
  // Emit a zero block for each spilled register.
  LLVM_DEBUG(dbgs() << "spills: " << Spills << "\n");
  for (unsigned i = 0; i < Spills; ++i) {
    const MCExpr *Expr = MCConstantExpr::create(0, *Ctx);
    Blocks.back()->Objects.push_back(
      std::make_unique<MCDataContainer>(Expr, 4,
        Blocks.back()->Objects.back()->SectionPair));
  }
}

void AArch64LiftingStreamer::createMCObjects() {
  // First, undef the labels, since they will all be redefined.
  for (auto &Label : MCBBB.Labels) {
    Label->Symbol->setUndefined();
  }

  const MCSubtargetInfo &STI = *Streamer->getContext().getSubtargetInfo();
  for (auto &Block : MCBBB.Blocks) {
    for (auto &I : Block->Objects) {
      if (SectionPair != I->SectionPair) {
        SectionPair = I->SectionPair;
        Streamer->switchSection(SectionPair.first, SectionPair.second);
      }
      if (I->isInst()) {
        MCInstContainer *InstContainer = cast<MCInstContainer>(I.get());
        Streamer->emitInstruction(InstContainer->Inst, STI);
      } else if (I->isLabel()) {
        MCLabelContainer *LabelContainer = cast<MCLabelContainer>(I.get());
        Streamer->emitLabel(LabelContainer->Symbol, LabelContainer->Loc);
      } else if (I->isData()) {
        MCDataContainer *DataContainer = cast<MCDataContainer>(I.get());
        if (DataContainer->HasMCExpr)
          Streamer->emitValue(DataContainer->Expr, DataContainer->Size);
        else
          Streamer->emitBytes(DataContainer->Data);
      } else if (I->isDirective()) {
        MCCallback *Callback = cast<MCCallback>(I.get());
        Callback->F();
      }
    }
  }
}

void AArch64LiftingStreamer::finalize(const MCInstrInfo &MII) {
  MCBBB.determineLeaders();
  MCBBB.createBasicBlocks();
  MCBBB.controlFlowAnalysis();
  MCBBB.defUseAnalysis(MII);
  MCBBB.livenessAnalysis();
  MCBBB.constructInterferenceGraph(MII);
  MCBBB.colorGraphs(MII);
  MCBBB.rewriteRegs(MII);
  MCBBB.mapAndPatchSpills(MII);
  createMCObjects();
  LLVM_DEBUG(dbgs() << "STATISTICS:\n");
  LLVM_DEBUG(dbgs() << "Instructions rewritten: " << InstructionsRewritten << "\n");
  LLVM_DEBUG(dbgs() << "Loads Inserted: " << LoadsInserted << "\n");
  LLVM_DEBUG(dbgs() << "Stores Inserted: " << StoresInserted << "\n");
  LLVM_DEBUG(dbgs() << "Instructions analyzed: " << InstructionsAnalyzed << "\n");
  LLVM_DEBUG(dbgs() << "Total instructions: " << TotalInstructions << "\n");
}
// Since we emit functions on function "switches", we must emit one
// last time at EOF.
void AArch64LiftingStreamer::emitLastBlock() { }
