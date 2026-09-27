---
type: Concept
title: "Register Allocation via Graph Coloring and Linear Scan"
description: "Theoretical intractability, liveness analysis, Chaitin-Briggs graph coloring heuristics, spilling trade-offs, and JIT linear scan allocation."
tags: [compilers, register-allocation, graph-coloring, linear-scan, np-complete, liveness-analysis]
generated:
  by: "antigravity/1.0"
  at: "2026-09-14T10:27:55+05:30"
status: stable
sources:
  - id: compiler-guide
    resource: "Technical Guide: Compiler Code Generation Pipeline"
---

# Register Allocation via Graph Coloring and Linear Scan

Register allocation is the phase in a compiler backend that maps an unbounded supply of virtual variables (temporaries) created during Intermediate Representation (IR) generation to a strictly finite set of $k$ physical hardware registers.[^compiler-guide]

Because accessing physical registers requires 0 CPU cycles while memory access introduces cache and bus latency, effective allocation is one of the most performance-critical passes in compiler engineering.[^compiler-guide]

---

## 1. Theoretical Complexity: NP-Completeness

Register allocation is formally NP-complete.[^compiler-guide] It reduces directly to the **Graph $k$-Coloring Problem**:

1. **Undirected Interference Graph ($G = (V, E)$)**:
   * **Vertices ($V$)**: Represent virtual variables / live ranges.[^compiler-guide]
   * **Edges ($E$)**: An undirected edge exists between $u$ and $v$ if and only if both variables are live simultaneously at any instruction in the control-flow graph.[^compiler-guide]
2. **Coloring**: Physical registers correspond to $k$ available colors. Two adjacent vertices (conflicting variables) cannot share the same color (physical register).[^compiler-guide]

Determining whether an arbitrary graph is $k$-colorable for $k \ge 3$ is NP-complete. Because interference graphs derived from unstructured control-flow graphs can take any arbitrary graph topology, global optimal register allocation without spilling cannot be solved in polynomial time.[^compiler-guide]

```
Virtual Live Ranges:          Interference Graph:
  v1: |--------|                (v1) ──────── (v2)
  v2:    |--------|              │   ╲        ╱ │
  v3:       |--------|           │    ╲      ╱  │
  v4:           |--------|       │     (v3)     │
                                 │              │
                                (v4) ───────────┘
```

---

## 2. The Chaitin-Briggs Graph Coloring Heuristic

Production Ahead-of-Time (AOT) compilers (e.g., GCC, Clang/LLVM) utilize variants of the **Chaitin-Briggs** formulation of Kempe’s heuristic:[^compiler-guide]

```
               [ Liveness Analysis ]
                         │
                         ▼
             [ Build Interference Graph ]
                         │
                         ▼
        ┌──────▶ [ Simplify (Degree < k) ]
        │                │
(If graph empty)         ▼ (If all degree >= k)
        │        [ Optimistic Spill ] ── Mark candidate node with lowest spill cost
        │                │
        └────────────────┘
                         │
                         ▼
             [ Select & Color (Pop Stack) ]
                         │
                 ┌───────┴───────┐
                 ▼               ▼
          (Color found)   (Actual Spill)
                 │               │
                 │        [ Insert Load/Store ]
                 │               │
                 │               ▼
                 └──────▶ [ Rebuild & Restart ]
```

1. **Liveness Analysis**: Computes the live-in and live-out sets for every basic block using backward iterative dataflow equations:
   $$\text{In}[B] = \text{Use}[B] \cup (\text{Out}[B] - \text{Def}[B])$$
   $$\text{Out}[B] = \bigcup_{S \in \text{Succ}[B]} \text{In}[S]$$
2. **Simplify**: Finds a node $v$ with degree $< k$. Kempe's rule dictates that if $d(v) < k$, removing $v$ and coloring the remaining graph guarantees that at least one color will remain free for $v$. Node $v$ is removed from the graph and pushed onto a coloring stack.[^compiler-guide]
3. **Spill Heuristic**: If all remaining nodes have degree $\ge k$, the compiler chooses a candidate node to spill to memory based on minimal cost:[^compiler-guide]
   $$\text{SpillCost}(v) = \frac{\sum_{\text{uses, defs}} 10^{\text{loop\_depth}}}{\text{degree}(v)}$$
4. **Select (Briggs Optimism)**: Nodes are popped from the stack in reverse order and assigned an available physical register. Even nodes marked for spilling may successfully receive a color if adjacent neighbors happen to share colors. If a node genuinely cannot be colored, it is spilled: load instructions are placed before uses, and store instructions after definitions, and the loop repeats.[^compiler-guide]

---

## 3. Linear Scan Allocation for JIT Compilers

While graph coloring produces highly efficient code, building and maintaining interference graphs requires $O(V^2)$ to $O(V^3)$ compile-time complexity. For Just-In-Time (JIT) compilers (e.g., Java HotSpot C1, JavaScript V8, .NET RyuJIT), compilation speed directly impacts runtime latency.[^compiler-guide]

JIT compilers employ **Linear Scan Register Allocation** (Poletto & Sarkar):[^compiler-guide]
* Converts variables into continuous 1D live intervals `[start, end]`.
* Sorts intervals by start point and performs a single linear sweep through the interval list.
* Keeps an active list of allocated physical registers. When an interval ends, its register is recycled. If an interval starts when all $k$ registers are occupied, the interval with the furthest end point is spilled to memory.[^compiler-guide]

---

## 4. Architectural Comparison

| Dimension | Chaitin-Briggs Graph Coloring | Linear Scan Allocation |
| :--- | :--- | :--- |
| **Primary Use Case** | Ahead-of-Time (AOT) Compilers (LLVM, GCC, Rustc) | Just-in-Time (JIT) Engines (V8, JVM HotSpot, WebKit FTL) |
| **Algorithmic Complexity** | $O(N^2)$ to $O(N^3)$ (Graph construction + iteration) | $O(N \log N)$ (Sorting live intervals + linear pass) |
| **Memory Footprint** | High: Stores dense or adjacency-list interference graph. | Minimal: Stores linear intervals and active register sets. |
| **Spill Code Quality** | **Superior**: Context-aware spill cost modeling and coalescing. | **Suboptimal**: May spill variables with high usage inside inner loops. |
| **Irregular Holes** | Accurately models non-contiguous live ranges. | Requires live range splitting to avoid over-conservative intervals. |

---

## Related Concepts & References

* [Compiler Code Generation Pipeline](compiler-code-generation.md): Instruction selection, instruction scheduling, and ABI contracts.
* [Compiler Code Generation Reference Guide](../references/compiler-code-generation-guide.md): Ingested video breakdown and complexity summaries.

---

## Deep Internals

1. **Register Coalescing and Conservative Heuristics**: Coalescing eliminates redundant `mov %v1, %v2` instructions by merging nodes $v1$ and $v2$ in the interference graph. However, unconstrained coalescing can turn a $k$-colorable graph into a non-colorable one. Compilers use *Briggs Conservative Coalescing* (merging is safe if the resulting node has fewer than $k$ neighbors of significant degree $\ge k$) or *George Coalescing* (merging $u$ into $v$ is safe if every neighbor of $u$ with degree $\ge k$ is already a neighbor of $v$).
2. **Rematerialization vs. Memory Spilling**: Spilling a register requires writing to the stack frame and emitting memory read operations at each use. Chaitin observed that many values are cheaper to recalculate than to reload from memory. If a spilled variable holds a compile-time constant, a frame-pointer offset, or a simple bitwise mask, the register allocator performs *rematerialization*—recomputing the value on demand via `mov` or `lea`—saving cache traffic and stack bandwidth.
3. **Chordal Interference Graphs and SSA Register Allocation**: On pure Static Single Assignment (SSA) form without critical edges, the interference graph of live ranges is a *chordal graph* (a graph where every cycle of four or more vertices has a chord). Chordal graphs can be colored optimally in polynomial time $O(V + E)$ using Maximum Cardinality Search (MCS). While SSA-form register allocation makes coloring polynomial, inserting $\phi$-node resolutions and handling physical register constraints (like fixed ABI registers and aliasing register sub-banks) reintroduces NP-hardness.

[^compiler-guide]: Technical Guide: Compiler Code Generation Pipeline (Chapters: Instruction Selection, Register Allocation, Instruction Ordering, Stack Frames, Peephole Optimization).
