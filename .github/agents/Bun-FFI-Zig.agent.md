You’re essentially asking for **“agent skills” documentation** — something an engineering agent (or a developer) could follow when building native extensions. Think of it like a playbook. Clean, practical, battle-tested patterns.

Two stacks:

1. **Bun + Zig/C (FFI / Bindgen)**
2. **Node.js + Rust (N-API / napi-rs)**

Let’s build those playbooks.

---

# Agent Skill: Bun + Zig/C Native Binding

This skill describes how to build high-performance native modules for JavaScript using **Bun’s Zig runtime** or **C ABI via bun:ffi**.

Bun is written in Zig, so the integration is extremely tight. Zig functions can be exposed directly to JS via **bindgen**, or C libraries can be loaded dynamically using **bun:ffi**.

Example Zig function exposed to JS:

```zig
const std = @import("std");
const jsc = @import("jsc");

pub fn add(global: *jsc.JSGlobalObject, a: i32, b: i32) !i32 {
    return std.math.add(i32, a, b) catch {
        return global.throwPretty("Integer overflow", .{});
    };
}
```

Binding declaration:

```ts
import { t, fn } from "bindgen";

export const add = fn({
  args: { global: t.globalObject, a: t.i32, b: t.i32 },
  ret: t.i32
});
```

Usage in JS:

```js
import { add } from "bun:math"

console.log(add(2,3))
```

Inside Bun, the binding generator produces glue code that converts JS values to native Zig values and back.

The native Zig code runs **at full native speed**. The only overhead is crossing the JS ↔ native boundary.

---

## Where’s the Performance Cost?

There are two main costs when calling native code.

**1 — Bridge cost (JS runtime → native)**
This is the overhead of the runtime switching contexts and invoking native code.

Includes:

* argument marshaling
* runtime stack transition
* callback wrapping

Typical overhead: **tens to hundreds of nanoseconds**.

This dominates when calling tiny functions repeatedly.

Bad example:

```
for (let i=0;i<1e8;i++) add(1,2)
```

Here the bridge overhead dominates.

---

**2 — Serialization / Data conversion**

When complex objects cross the boundary.

Examples:

* JS string → Zig string
* JS object → struct
* JSON parsing
* Array copying

This can involve:

* memory allocation
* copying buffers
* type checking

This dominates when sending large payloads.

---

### Practical rule

If work per call is large → native wins.

If calls are tiny and frequent → boundary dominates.

Batch operations whenever possible.

---

# Edge Cases / Sharp Corners (Bun + Zig)

These are the real traps engineers hit.

### 1. Exception boundaries

Native exceptions must **never cross the JS boundary**.

Correct:

```
catch { return global.throwPretty(...) }
```

Wrong:

```
panic("error")
```

A panic will crash the runtime.

---

### 2. Memory ownership

If native code allocates memory, you must define who frees it.

Bad pattern:

```
return pointer_to_heap
```

JS has no idea how to free it.

Preferred:

* return JS objects
* or provide `free()` API.

---

### 3. Struct alignment

JS engines assume certain memory layouts.

Avoid passing structs directly unless layout is explicitly defined.

Prefer primitive parameters.

---

### 4. GC interaction

JS garbage collector may move or free objects.

Never hold raw JS references in native code without pinning.

---

### 5. Thread safety

Bun has an event loop.

If native code spawns threads that call JS functions directly, the runtime may crash.

Always route JS callbacks through the runtime event loop.

---

### 6. ABI compatibility

Always expose a **C ABI boundary** if your code may be reused.

```
export fn myFunction(...) callconv(.C)
```

---

# Best Practices (Bun + Zig)

Design native APIs carefully.

**Keep the boundary small**

Do more work in native code.

Bad:

```
for each element → call zig
```

Better:

```
send entire array → process in zig
```

---

**Use typed arrays for zero-copy**

JS buffers can map directly to native memory.

Example:

```
Uint8Array → Zig slice
```

This avoids memory copying.

---

**Avoid dynamic allocation per call**

Reuse buffers or allocate once.

---

**Use predictable data formats**

Binary structs or typed arrays beat JSON.

---

**Expose stable APIs**

Changing struct layouts breaks consumers.

Version your API.

---

# Practical Optimization Checklist (Bun)

Before shipping native code, verify:

* minimize number of JS → native calls
* avoid JSON serialization across boundary
* use typed arrays / buffers
* batch processing in native code
* convert errors to JS exceptions
* never allow Zig panics to escape
* avoid global mutable state
* benchmark boundary latency
* avoid allocating memory per call
* ensure thread safety

---



