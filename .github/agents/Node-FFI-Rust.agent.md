# Agent Skill: Node.js + Rust Native Module

Node exposes native functionality through **N-API**.

Rust libraries like **napi-rs** make this easy.

Rust example:

```rust
use napi_derive::napi;

#[napi]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

Node usage:

```js
const { add } = require('./index.node')

console.log(add(2,3))
```

Under the hood:

```
JS → V8 → N-API → Rust → N-API → V8 → JS
```

Rust runs at native speed.

---

# Where’s the Performance Cost? (Node + Rust)

Three main areas.

### 1 — V8 ↔ Native boundary

This includes:

* type conversion
* stack switching
* safety checks

Overhead typically ~100–300ns.

---

### 2 — Serialization

JS objects → Rust structs.

Example cost:

```
JS object → JSON → Rust parse
```

Very expensive.

Better:

```
TypedArray → &[u8]
```

Zero copy.

---

### 3 — Memory allocations

Frequent allocations inside Rust or JS.

Example:

```
return new JS object every iteration
```

Triggers GC pressure.

---

# Edge Cases / Sharp Corners (Node + Rust)

### 1. Rust panics crossing FFI

Never allow panic to cross the boundary.

Wrap code:

```
std::panic::catch_unwind(...)
```

---

### 2. Node event loop blocking

If Rust performs long CPU tasks synchronously:

```
Node server freezes
```

Use:

```
napi::Task
```

or worker threads.

---

### 3. Buffer ownership

If Rust holds reference to Node Buffer after function returns → unsafe.

Use:

```
Env::create_reference
```

or copy memory.

---

### 4. Async callbacks

Calling JS from Rust threads requires Node runtime context.

Direct calls from arbitrary threads will crash.

---

### 5. Cross-platform builds

Native modules must be compiled for:

```
linux x64
linux arm64
mac
windows
```

Use CI or prebuild tools.

---

### 6. Struct layout

Rust structs are not ABI stable.

Always expose C-compatible layouts if sharing memory.

```
#[repr(C)]
```

---

# Best Practices (Node + Rust)

Use **napi-rs** rather than raw N-API.

Benefits:

* stable ABI
* memory safety
* simpler build system

---

Design the API around **buffers and primitives**.

Bad:

```
Rust accepts JS object
```

Better:

```
Rust accepts buffer or typed array
```

---

Batch work.

Example:

Bad:

```
for each row → call rust
```

Better:

```
send entire dataset
```

---

Avoid crossing the boundary inside loops.

---

Use async tasks for CPU heavy work.

---

Avoid unnecessary allocations.

---

# Practical Optimization Checklist (Node + Rust)

Before deploying:

* minimize JS ↔ Rust calls
* avoid JSON serialization
* prefer TypedArray / Buffer
* batch operations
* avoid blocking event loop
* wrap Rust panics
* benchmark boundary overhead
* use prebuilt binaries
* avoid holding JS references across threads
* reuse buffers where possible

---

# The Key Mental Model

FFI is not magic.

Think of it like **a toll booth between two countries**.

Cars (data) must:

1. stop
2. be inspected
3. be translated

Then they can continue.

If you send **a million tiny cars**, the toll booth dominates.

If you send **one giant truck**, the trip is efficient.

That mental model predicts nearly every performance problem people encounter with FFI.

---