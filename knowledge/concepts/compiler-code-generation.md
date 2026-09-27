---
type: Concept
title: "Compiler Code Generation Pipeline"
description: "Architectural stages of compiler backend lowering: instruction selection via tree tiling, pipeline instruction scheduling, ABI calling conventions, and peephole rewrites."
tags: [compilers, code-generation, instruction-selection, instruction-scheduling, abi, peephole]
generated:
  by: "antigravity/1.0"
  at: "2026-09-14T10:27:55+05:30"
status: stable
sources:
  - id: compiler-guide
    resource: "Technical Guide: Compiler Code Generation Pipeline"
  - id: deep-internals-notes
    resource: "Engineering Notes: Deep Internals in Security and Compilers"
---

# Compiler Code Generation Pipeline

Code generation is the backend phase of a compiler where machine-independent Intermediate Representation (IR) trees or Static Single Assignment (SSA) control-flow graphs are lowered to target-specific machine instructions.[^compiler-guide] Because simultaneous global optimization of instruction choices, physical registers, and execution order is NP-complete, modern production compilers decompose code generation into discrete heuristic passes.[^compiler-guide]

```
Intermediate Representation (IR)
              │
              ▼
    [ Instruction Selection ]      ── Tree Tiling / Maximal Munch / Pattern Matching
              │
              ▼
    [ Pre-RA Scheduling ]         ── Hides Latency (Maximizes ILP)
              │
              ▼
    [ Register Allocation ]       ── Maps Virtual Regs -> Physical Regs / Spills
              │
              ▼
    [ Post-RA Scheduling ]        ── Schedules Spill Code & Resolves Hazards
              │
              ▼
    [ Frame Lowering & Prologue ] ── Stack Frames, ABI Contracts, Callee Saves
              │
              ▼
    [ Peephole Optimization ]     ── Sliding Window Local Rewrites
              │
              ▼
      Target Machine Code
```

---

## 1. Instruction Selection: The Tree Tiling Problem

Instruction selection translates low-level IR expression trees into machine instructions.[^compiler-guide] This is formalized as **tree tiling**: given an IR tree and a grammar of machine instruction patterns (tiles) with associated costs, the compiler finds a covering of the tree with minimal total cost.[^compiler-guide]

```
IR Expression:  a = b + *(c + 4)

             MEM_STORE [a]
                  │
                 ADD
               ┌──┴──┐
             VAR[b]  MEM_LOAD
                        │
                       ADD
                     ┌──┴──┐
                   VAR[c] CONST[4]

Tiling on x86-64 (CISC Complex Tile):
  `movq 4(%rcx), %rax`   --> covers MEM_LOAD(ADD(VAR[c], CONST[4]))
  `addq %rbx, %rax`      --> covers ADD(VAR[b], ...)
  `movq %rax, a`         --> covers MEM_STORE
```

### Tree Covering Algorithms

| Algorithm | Mechanism | Time Complexity | Optimality | Trade-off |
| :--- | :--- | :--- | :--- | :--- |
| **Maximal Munch** | Top-down greedy matching selecting the largest valid tile at each root. | $O(N)$ | Suboptimal (greedy heuristic) | Fast and simple; can miss globally cheaper smaller tile combinations. |
| **Dynamic Programming (e.g., BURS/IBURG)** | Bottom-up dynamic programming computing optimal cost vector per node, followed by top-down emission. | $O(N)$ with precomputed tables | Optimal for expression trees | Requires tree structure; does not inherently produce optimal covers for Directed Acyclic Graphs (DAGs) with shared subexpressions. |
| **Graph / SSA Selection (e.g., LLVM SelectionDAG/GlobalISel)** | Matches patterns on directed graphs with multiple uses. | $O(N \log N)$ average | Near-optimal heuristic | Handles common subexpressions and condition codes at higher compile-time cost. |

---

## 2. Instruction Scheduling and Latency Hiding

In pipelined CPUs, instructions take multiple clock cycles to complete (e.g., memory loads, floating-point arithmetic, variable-latency division). If an instruction depends immediately on the output of an uncompleted preceding instruction (Read-After-Write hazard), the processor pipeline stalls.[^compiler-guide]

Instruction scheduling reorders independent operations to fill latency slots while preserving data dependencies (RAW, WAR, WAW).[^compiler-guide]
* **In-Order Architectures** (Microcontrollers, embedded cores, early ARM/MIPS): Compiler scheduling is critical. The CPU executes instructions strictly as ordered in the binary; stalls directly degrade throughput.[^compiler-guide]
* **Out-of-Order (OoO) Architectures** (Modern x86, Apple M-series, high-performance ARM): Hardware reorder buffers (ROBs) dynamically schedule operations around stalls. However, compiler scheduling still minimizes pressure on the execution engine and aligns instruction decoders.[^compiler-guide]

---

## 3. Stack Frames and Calling Conventions (ABIs)

Function execution requires an activation record (stack frame) to preserve execution context across calls, house spilled variables, and support re-entrancy/recursion.[^compiler-guide]

The Application Binary Interface (ABI), such as **System V AMD64 ABI** (Linux/macOS x86-64), establishes a contract governing register responsibilities:[^compiler-guide]

```
System V AMD64 ABI Register Partition:
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ Role                         │ Registers                                              │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ Integer Arguments (Caller)   │ RDI (1st), RSI (2nd), RDX (3rd), RCX (4th), R8, R9    │
│ Return Value (Caller)        │ RAX (1st), RDX (2nd)                                   │
│ Caller-Saved (Scratch)       │ RAX, RCX, RDX, RSI, RDI, R8, R9, R10, R11             │
│ Callee-Saved (Preserved)     │ RBX, RSP, RBP, R12, R13, R14, R15                      │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
```

* **Caller-Saved**: The caller must save these registers onto its stack frame before calling a subroutine if it needs their values retained across the call.[^compiler-guide]
* **Callee-Saved**: The subroutine must save and restore these registers in its prologue and epilogue if it mutates them, guaranteeing the caller's environment remains intact.[^compiler-guide]

---

## 4. Peephole Optimization

The final code emission stage runs a sliding window (typically 2–4 instructions) over the linear instruction stream to match and replace inefficient sequences produced by naive lowering passes:[^compiler-guide]

1. **Zeroing Idiom**: Replacing `mov $0, %rax` (7–10 bytes) with `xor %eax, %eax` (2 bytes).[^compiler-guide]
2. **Strength Reduction**: Replacing `imul $8, %rax` with `shl $3, %rax` or `lea (, %rax, 8), %rax`.[^compiler-guide]
3. **Redundant Store/Load Elimination**:
   ```assembly
   # Unoptimized Lowering:
   movq %rax, -8(%rbp)
   movq -8(%rbp), %rax    # Redundant reload eliminated by peephole window
   ```
4. **Branch over Unconditional Jump Inversion**: Eliminates jump-to-jump trampolines.

---

## Related Concepts & References

* [Register Allocation via Graph Coloring](register-allocation-graph-coloring.md): Graph coloring, interference graphs, and linear scan heuristics.
* [Compiler Code Generation Reference Guide](../references/compiler-code-generation-guide.md): Ingested video summary and theoretical complexity notes.
* [Engineering Notes: Deep Internals](../references/engineering-deep-internals.md): Microarchitectural and scheduler trade-offs.

---

## Deep Internals

1. **Zero-Latency Register Renaming for Zeroing Idioms**: Emitting `xor %eax, %eax` instead of `mov $0, %rax` is not merely smaller in byte length (2 bytes vs. 7–10 bytes). Modern x86 processors (Intel Core, AMD Zen) recognize `xor reg, reg` directly in the Register Alias Table (RAT) during the decode stage. The hardware sets the physical register pointer to the architectural zero register and eliminates the instruction entirely before dispatch, consuming zero execution cycles and breaking false data dependencies in the Reorder Buffer (ROB).[^deep-internals-notes]
2. **The System V AMD64 128-Byte "Red Zone"**: Under the System V AMD64 ABI, the 128-byte block of memory immediately below the stack pointer (`%rsp`) is formally reserved as the *red zone*. A leaf function (a function that invokes no subroutines) can allocate local variables and perform spills directly in this 128-byte region without adjusting `%rsp` via `subq`/`addq` instructions in its prologue and epilogue, because operating system kernel interrupt and signal handlers are guaranteed not to clobber memory within 128 bytes below `%rsp`.[^deep-internals-notes]
3. **Phase-Ordering Antagonism in Schedulers (Pre-RA vs. Post-RA)**: Instruction scheduling and register allocation work at cross-purposes. Pre-RA scheduling stretches variable lifetimes to maximize instruction-level parallelism (ILP), which inflates register pressure and forces expensive memory spills. Conversely, post-RA scheduling operates under physical register constraints, where reused registers introduce artificial Write-After-Read (WAR) and Write-After-Write (WAW) dependencies that restrict instruction movement. Compilers resolve this by running a conservative pre-RA pass that throttles ILP when approaching the register pressure threshold, followed by a post-RA pass dedicated to hiding spill load/store latencies.[^deep-internals-notes]

[^compiler-guide]: Technical Guide: Compiler Code Generation Pipeline (Chapters: Instruction Selection, Register Allocation, Instruction Ordering, Stack Frames, Peephole Optimization).
[^deep-internals-notes]: Engineering Notes: Deep Internals in Security and Compilers.
