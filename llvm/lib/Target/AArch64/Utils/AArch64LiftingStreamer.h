//===-- AArch64LiftingStreamer.h - Assembly to IR lifter --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <map>
#include <memory>
#include <set>

namespace llvm {

namespace {

// Represents all sorts of constructs in an assembly file. e.g.
// functions, directives, labels, etc.
class MCObject {
protected:
  enum KindTy { 
    k_Inst, 
    k_Data, 
    k_Directive, 
    k_Label
    } Kind;
  MCObject(KindTy K, std::pair<MCSection *, uint32_t> S) : Kind(K), IsLeader(false), IsFuncStart(false), SectionPair(S) {}

public:
  bool IsLeader : 1;
  bool IsFuncStart : 1;
  unsigned BlockNum;
  std::pair<MCSection *, uint32_t> SectionPair;
  virtual ~MCObject() = default;
  bool isInst() const { return Kind == k_Inst; }
  bool isData() const { return Kind == k_Data; }
  bool isDirective() const { return Kind == k_Directive; }
  bool isLabel() const { return Kind == k_Label; }
  virtual void dump(raw_ostream &OS) {
    OS << "unimplemented\n";
  };

  static inline bool classof(MCObject const *O) {
    return true;
  }
};

static inline raw_ostream &operator<<(raw_ostream &OS, MCObject &O) {
  O.dump(OS);
  return OS;
}

class MCInstContainer : public MCObject {
public:
  MCInst Inst;
  bool NeedsPatch : 1;
  MCInstContainer(MCInst I, std::pair<MCSection *, uint32_t> S) : MCObject(k_Inst, S), Inst(I), NeedsPatch(false) {}
  MCInstContainer(MCInst I, std::pair<MCSection *, uint32_t> S, bool Patch) : MCObject(k_Inst, S), Inst(I), NeedsPatch(Patch) {}
  bool needsPatch() { return NeedsPatch; }
  void dump(raw_ostream &OS) override {
    Inst.dump_pretty(OS);
  }
  static inline bool classof(MCObject const *O) {
    return O->isInst();
  }
  
  static inline bool classof(MCInstContainer const *) {
    return true;
  }
};

class MCLabelContainer : public MCObject {
public:
  MCSymbol *Symbol;
  SMLoc Loc;
  MCLabelContainer(MCSymbol *Sym, SMLoc L, std::pair<MCSection *, uint32_t> S) : MCObject(k_Label, S), Symbol(Sym), Loc(L) {}
  void dump(raw_ostream &OS) override {
    OS << Symbol->getName() << ":";
  }
  static inline bool classof(MCObject const *O) {
    return O->isLabel();
  }
  
  static inline bool classof(MCLabelContainer const *) {
    return true;
  }
};

class MCDataContainer : public MCObject {
public:
  bool HasMCExpr : 1;
  // TODO this should be something else..
  const MCExpr *Expr = nullptr;
  unsigned Size = 0;
  std::string Data;
  MCDataContainer(StringRef D, std::pair<MCSection *, uint32_t> S) : MCObject(k_Data, S), HasMCExpr(false), Data(D) {}
  MCDataContainer(const MCExpr *Expr, unsigned Size, std::pair<MCSection *, uint32_t> S) : MCObject(k_Data, S), HasMCExpr(true), Expr(Expr), Size(Size) {} 
  void dump(raw_ostream &OS) override {
    OS << "raw data: ";
    if (HasMCExpr) { 
      OS << *Expr;
    } else {
      OS << Data;
    }
  }
  static inline bool classof(MCObject const *O) {
    return O->isData();
  }
  
  static inline bool classof(MCDataContainer const *) {
    return true;
  }
};

class MCCallback : public MCObject {
public:
  std::function<void()> F;
  MCCallback(std::function<void()> Func, std::pair<MCSection *, uint32_t> S) : MCObject(k_Directive, S), F(Func) {}
};

// A basic block. Contains a sequence of instructions, a list of
// ancestors, and a list of successors.
class MCBasicBlock {
public:
  std::set<MCBasicBlock *> Ancestors;
  std::set<MCBasicBlock *> Successors;
  std::set<MCRegUnit> Defs;
  std::set<MCRegUnit> Uses;
  std::set<MCRegUnit> LiveIn;
  std::set<MCRegUnit> LiveOut;
  std::vector<std::unique_ptr<MCObject>> Objects;
  std::vector<std::set<MCRegUnit>> Dead;
  unsigned BlockNum;
  bool IsStartOfFunc = false;

  MCBasicBlock(unsigned BN) : BlockNum(BN) {}
  static std::unique_ptr<MCBasicBlock> createBasicBlock();
  void dump(raw_ostream &OS);
  // void replaceReg(MCRegister Old, MCRegister New, unsigned Pos, const MCInstrInfo &MII, const MCRegisterInfo *MRI, bool Defined = true);
  // void backwardsReplaceReg(MCRegister Old, MCRegister New, const MCInstrInfo &MII, const MCRegisterInfo *MRI);
};

class InterferenceGraph {
public:
  // TODO actual iterator for this
  using Node = std::pair<MCRegUnit, std::set<MCRegUnit>>;
  std::map<MCRegUnit, std::set<MCRegUnit>> Nodes;
  std::map<MCRegUnit, unsigned> Colors;
  void addNode(MCRegUnit MCRU) {
    if (Nodes.count(MCRU)) return;
    Nodes[MCRU] = std::set<MCRegUnit>();
  }
  void addEdge(MCRegUnit From, MCRegUnit To) {
    if (From == To)
      return; // cannot draw circular edge

    Nodes[From].insert(To);
    Nodes[To].insert(From);
  }

  void addEdges(MCRegUnit From, std::set<MCRegUnit> &Targets) {
    for (MCRegUnit To : Targets) {
      addEdge(From, To);
    }
  }

  void addVertex(MCRegUnit V) { Nodes[V] = std::set<MCRegUnit>(); }

  void dump(raw_ostream &OS, const MCRegisterInfo *MRI);

  void assignColor(MCRegUnit R, unsigned C) { Colors[R] = C; }

  Node removeNode(MCRegister R) {
    // Remove this register from the graph.
    for (MCRegUnit Other : Nodes[R]) {
      Nodes[Other].erase(R);
    }
    // Erase the node itself.
    Node This = std::make_pair(R, Nodes[R]);
    Nodes.erase(R);
    return This;
  }
  void precolor(std::set<MCRegUnit> ToPrecolor, const MCRegisterInfo *MRI);
  MCRegUnit color(std::set<unsigned> &Color, std::map<MCRegUnit, uint64_t> &RegFrequencies, const MCRegisterInfo *MRI);
};

// This class builds basic blocks.
class MCBasicBlockBuilder {

public:
  MCContext *Ctx;
  MCBasicBlockBuilder(MCContext *Ctx)
      : Ctx(Ctx) {}
  std::vector<std::unique_ptr<MCObject>> Objects;
  std::map<StringRef, uint64_t> LabelLocs;
  std::map<uint64_t, std::pair<MCSymbol *, SMLoc>> LocLabels;
  std::set<uint64_t> Leaders;
  std::vector<std::unique_ptr<MCBasicBlock>> Blocks;
  std::vector<uint64_t> FuncStarts;
  std::map<uint64_t, unsigned> InstsToBlocks;
  std::map<MCRegUnit, uint64_t> RegFrequencies;
  std::map<MCRegister, int64_t> SpillsToOffsets;
  std::set<MCLabelContainer *> Labels;
  unsigned Spills = 0;
  unsigned NextVirtualGPR = MCRegister::VirtualRegFlag;
  unsigned NextVirtualVectorReg = MCRegister::VirtualRegFlag | (MCRegister::VirtualRegFlag >> 1);
  InterferenceGraph GPRGraph;
  InterferenceGraph VectorGraph;
  bool NextInstrIsFuncStart = true; // first instr is a leader
  void determineLeaders();
  void createBasicBlocks();
  void controlFlowAnalysis();
  void defUseAnalysis(const MCInstrInfo &MII);
  void livenessAnalysis();
  void constructInterferenceGraph(const MCInstrInfo &MII);
  void colorGraphs(const MCInstrInfo &MII);
  void spill(MCRegUnit MCRU, const MCInstrInfo &MII, std::set<MCRegister> RegsToPreserve);
  void rewriteRegs(const MCInstrInfo &MII);
  void mapAndPatchSpills(const MCInstrInfo &MII);
  void dumpCode(const MCInstrInfo &MII, const MCRegisterInfo *MRI);
};

class MCFunction {
  std::vector<std::unique_ptr<MCObject>> Objects;

public:
  MCBasicBlock *CurrentBlock;
  MCFunction();
  void createAndSwitchBasicBlock();
  void dump(raw_ostream &OS);
};

} // end anonymous namespace

// A class to buffer and view assembly instructions for various analysis
// purposes.
class AArch64LiftingStreamer : public MCStreamer {
private:
  std::vector<std::unique_ptr<MCFunction>> Functions;
  const MCInstrInfo *MII;
  MCStreamer *Streamer;
  MCFunction *CurrentFunction;
  std::vector<MCInst> Instructions;
  MCBasicBlockBuilder MCBBB;
  bool FirstInstrNotEmitted = true;
  std::pair<MCSection *, uint32_t> SectionPair = std::make_pair(nullptr, 0);

  void addInstrToBlock(const MCInst &Inst);
  void createAndSwitchMCFunction();
  MCBasicBlock &getCurrentBasicBlock();

public:
  AArch64LiftingStreamer(MCStreamer *S);
  ~AArch64LiftingStreamer() = default;
  void setMII(const MCInstrInfo *M) {MII = M;}
  void emitLabel(MCSymbol *Symbol, SMLoc Loc = SMLoc()) override;
  void emitInstruction(const MCInst &Inst, const MCSubtargetInfo &) override;
  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override;
  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        Align ByteAlignment) override;
  void emitZerofill(MCSection *Section, MCSymbol *Symbol = nullptr,
                    uint64_t Size = 0, Align ByteAlignment = Align(1),
                    SMLoc Loc = SMLoc()) override;
  void emitBytes(StringRef Data) override;
  void emitDataRegion(MCDataRegionType Kind) override;
  void emitValueToAlignment(Align Alignment, int64_t Value, unsigned ValueSize, unsigned MaxBytesToEmit) override;
  void emitCodeAlignment(Align Alignment, const MCSubtargetInfo *STI, unsigned MaxBytesToEmit) override;
  void emitValueImpl(const MCExpr *Value, unsigned Size, SMLoc Loc) override;
  void initSections(bool NoExecStack, const MCSubtargetInfo &STI) override;
  void switchSection(MCSection *Section, uint32_t Subsection) override;
  void finishImpl() override;
  void emitLastBlock();
  bool consumeAsmFunc();
  void finalize(const MCInstrInfo &MII);
  void createMCObjects();
};

} // end namespace llvm