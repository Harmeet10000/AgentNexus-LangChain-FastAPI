---
type: Reference
title: "Technical Guide: Compiler Code Generation Pipeline"
description: "Reference notes covering compiler backend stages: instruction selection, register allocation, instruction ordering, stack frames, and peephole optimization."
tags: [compilers, code-generation, instruction-selection, register-allocation, peephole, reference]
generated:
  by: "antigravity/1.0"
  at: "2026-09-14T10:27:55+05:30"
status: stable
sources:
  - id: compiler-guide
    resource: "Technical Guide: Compiler Code Generation Pipeline"
---

# Technical Guide: Compiler Code Generation Pipeline

This reference captures the technical video guide on compiler backend architecture and code generation from intermediate representation (IR) to target machine code.[^compiler-guide]

## Source Summary

Compilers transform intermediate representation (IR) into target machine code. Because optimal optimization across variables, instruction selection, and execution scheduling is mathematically intractable (NP-complete or undecidable), modern backends rely on heuristics.[^compiler-guide]

### Key Stages of Code Generation

1. **Instruction Selection (5:10 - 7:05)**: Maps IR expression trees to machine-specific instructions via tree tiling, covering IR nodes with the lowest-cost instruction tiles. Complex CISC ISAs (such as x86) present rich tiling options that merge memory loading, computation, and storage into single instructions.[^compiler-guide]
2. **Register Allocation (8:31 - 11:05)**: Maps an unbounded set of virtual IR registers/variables onto a strictly bounded set of physical CPU registers. Modeled as a graph coloring problem over interference graphs constructed from live ranges. When the chromatic number exceeds available physical registers ($k$), variables are spilled to the stack frame. JIT compilers often substitute linear scan allocation for throughput.[^compiler-guide]
3. **Instruction Ordering / Scheduling (11:05 - 13:26)**: Reorders machine instructions to hide memory latency and prevent execution pipeline stalls, particularly critical on in-order processors and microcontrollers.[^compiler-guide]
4. **Stack Frames & Calling Conventions (13:26 - 16:08)**: Synthesizes activation records on the stack to support recursion and local scope. Adheres to target Application Binary Interfaces (ABIs, such as System V AMD64) defining argument registers (e.g., `rdi`, `rsi`), return registers (`rax`), and caller-saved vs. callee-saved register partitions.[^compiler-guide]
5. **Peephole Optimization (16:08 - 17:33)**: Executes a sliding-window scan over generated machine code to apply local algebraic simplifications (e.g., `xor %rax, %rax` for zeroing, strength reduction converting multiplication by powers of two to bitwise arithmetic shifts).[^compiler-guide]

## Theoretical Complexity

Register allocation via graph coloring is proven NP-complete because it directly reduces to the $k$-coloring problem of an arbitrary undirected interference graph (09:21 - 10:14).[^compiler-guide]

## Derived Concepts

* [Compiler Code Generation Pipeline](../concepts/compiler-code-generation.md): Detailed mechanics of tree tiling, instruction scheduling, calling conventions, and peephole rewrites.
* [Register Allocation via Graph Coloring](../concepts/register-allocation-graph-coloring.md): Deep dive into liveness analysis, interference graphs, Chaitin-Briggs heuristics, and linear scan trade-offs.

[^compiler-guide]: Technical Guide: Compiler Code Generation Pipeline (Chapters: Instruction Selection, Register Allocation, Instruction Ordering, Stack Frames, Peephole Optimization).
