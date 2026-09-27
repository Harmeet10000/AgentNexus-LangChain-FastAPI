---
okf_version: "0.2"
---

# Knowledge Index

## Concepts
* [Compiler Code Generation Pipeline](concepts/compiler-code-generation.md): Architectural stages of compiler backend lowering: instruction selection via tree tiling, pipeline instruction scheduling, ABI calling conventions, and peephole rewrites.
* [Inter-Process Communication Architectures in Rust](concepts/rust-ipc-protocols.md): Taxonomy, kernel mechanisms, and implementation patterns for remote (HTTP, SSE, WebSockets, QUIC, TCP) and local (Unix Domain Sockets, Pipes, STDIO) IPC in asynchronous Rust.
* [Public Key Certificate Pinning in Android](concepts/android-certificate-pinning.md): Architectural mechanics, threat model, and comparison between certificate pinning and public key (SPKI) pinning.
* [Register Allocation via Graph Coloring and Linear Scan](concepts/register-allocation-graph-coloring.md): Theoretical intractability, liveness analysis, Chaitin-Briggs graph coloring heuristics, spilling trade-offs, and JIT linear scan allocation.

## Playbooks
* [Configuring Android Network Security Config for Certificate Pinning](playbooks/android-network-security-config.md): Practical implementation guide for `network_security_config.xml`, pin-sets, and lockout prevention.

## References
* [Engineering Notes: Deep Internals in Security and Compilers](references/engineering-deep-internals.md): Technical edge cases and hardware/OS-level invariants for Android TLS and compiler backends.
* [Technical Guide: Certificate Pinning in Android Applications](references/android-certificate-pinning-guide.md): Ingested summary and notes from the Android certificate pinning video guide.
* [Technical Guide: Compiler Code Generation Pipeline](references/compiler-code-generation-guide.md): Ingested summary and notes from the compiler code generation pipeline video guide.
* [Technical Guide: Inter-Process Communication (IPC) Patterns in Rust](references/rust-ipc-patterns-guide.md): Ingested summary and notes on remote and local Rust IPC patterns.
