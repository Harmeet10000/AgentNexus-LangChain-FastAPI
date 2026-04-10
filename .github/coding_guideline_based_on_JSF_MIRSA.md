Here is a comprehensive **Coding Guidelines Compilation Sheet** (50+ rules) synthesized from the **JSF AV C++ Coding Standards** (2005, safety-critical avionics focus), **MISRA C++:2008** (critical systems safety subset), **Google C++ Style Guide** (large-scale consistency and maintainability), **LLVM Coding Standards** (compiler/toolchain pragmatism and readability), and **Linux Kernel Coding Style** (systems-level simplicity and distributed collaboration).

Rules are organized into clear categories for easy reference. Each entry includes:
- **Guideline**: Clear, actionable statement.
- **Source(s)**: Primary origin(s), with specific rule numbers where available (e.g., JSF AV Rule X, MISRA Rule Y-Z, Google section, LLVM section, Linux section).
- **Language Notes**: Applicability or adaptation for **C++** (original context for most), and how to apply in **Go**, **TypeScript**, or **Python** (universal principles marked as **Universal**). Language-specific rules note restrictions.
- **Rationale**: Why it matters, with nuances, trade-offs, and implications.

This sheet emphasizes **timeless principles** (simplicity, predictability, readability, low complexity, explicitness) while noting source-specific differences. Many rules transfer directly to other languages via analogous practices (e.g., function length limits, naming conventions, avoiding deep nesting).

### 1. General Design & Complexity (Focus: Maintainability and Verifiability)
1. **Keep functions/methods short**: Limit to ≤200 logical source lines (preferred <50–100 for readability).  
   **Source**: JSF AV (code size rules), Linux Kernel (functions should be short and do one thing).  
   **Language Notes**: C++ / **Universal** (apply to Go funcs, TS functions/components, Python defs).  
   **Rationale**: Long functions are hard to understand, test, debug, and maintain. Edge case: Complex state machines may need refactoring into helpers. Implication: Improves test coverage and onboarding.

2. **Limit cyclomatic complexity**: Keep ≤20 per function (exceptions for simple switch/case with documentation).  
   **Source**: JSF AV (complexity metrics).  
   **Language Notes**: C++ / **Universal** (use tools like radon for Python, complexity plugins for Go/TS).  
   **Rationale**: High branching explodes test cases and error paths. Trade-off: Forces cleaner design.

3. **Limit indentation/nesting depth**: Maximum 3–4 levels; deeper nesting signals need for refactoring.  
   **Source**: Linux Kernel, LLVM (early returns to reduce nesting), JSF AV/MISRA (flow control).  
   **Language Notes**: C++ / **Universal**.  
   **Rationale**: Deep nesting hides logic and increases cognitive load.

4. **Favor low coupling and high cohesion**: Modules/functions should have minimal dependencies and single responsibility.  
   **Source**: JSF AV (General Design section), Google (composition over inheritance).  
   **Language Notes**: **Universal**.  
   **Rationale**: Reduces ripple effects from changes. Nuance: In Go, use small packages; in TS/Python, small modules/classes.

5. **No dead/unreachable/infeasible code**.  
   **Source**: MISRA C++ 0-1-1, 0-1-2, 0-1-9.  
   **Language Notes**: C++ / **Universal** (static analyzers or linters).  
   **Rationale**: Indicates defects or incomplete logic.

6. **Every defined function should be called at least once** (except main or library exports).  
   **Source**: MISRA C++ 0-1-10.  
   **Language Notes**: C++ / **Universal**.  
   **Rationale**: Prevents unused code bloat.

### 2. Naming Conventions (Focus: Readability and Consistency)
7. **Use consistent, descriptive naming**: Avoid abbreviations except common ones (e.g., i for loop).  
   **Source**: Linux Kernel (Spartan, descriptive), LLVM (avoid extreme abbreviations), Google (snake_case vars, CamelCase types).  
   **Language Notes**: C++ (project-specific) / **Universal** (Go: exported Capitalized; Python: snake_case; TS: camelCase).  
   **Rationale**: Self-documenting code reduces comments needed.

8. **Classes/Types**: CamelCase or PascalCase.  
   **Source**: Google, JSF AV (naming classes).  
   **Language Notes**: C++ / Adapt in Go (exported), Python (CamelCase for classes), TS.  
   **Rationale**: Distinguishes types from variables.

9. **Functions/Methods**: Start with verb, descriptive (e.g., calculateTotal).  
   **Source**: Google, LLVM (lowercase start in some contexts), JSF AV.  
   **Language Notes**: **Universal**.  
   **Rationale**: Intent clear at call site.

10. **Variables/Parameters**: snake_case or camelCase consistently; no Hungarian notation.  
    **Source**: Linux Kernel, Google.  
    **Language Notes**: **Universal**.  
    **Rationale**: Compiler knows types; focus on meaning.

11. **Constants**: kCamelCase or UPPER_SNAKE_CASE.  
    **Source**: Google, JSF AV.  
    **Language Notes**: C++ / **Universal** (const/readonly in TS, UPPER in Python/Go).  
    **Rationale**: Immediate visual distinction.

12. **Files**: Lowercase with underscores or dashes; descriptive.  
    **Source**: JSF AV, Google.  
    **Language Notes**: **Universal** (Go packages, Python modules).  
    **Rationale**: Easy navigation in large projects.

### 3. Formatting & Style (Focus: Uniformity for Large Teams)
13. **Line length limit**: Prefer 80 columns (up to 100–120 in some projects).  
    **Source**: Linux Kernel, LLVM, JSF AV (style section).  
    **Language Notes**: **Universal** (enforce via editorconfig or linters).  
    **Rationale**: Allows side-by-side editing.

14. **Indentation**: Consistent (tabs 8 chars in Linux; 2 spaces in Google/LLVM).  
    **Source**: Linux Kernel (tabs), Google/LLVM (2 spaces).  
    **Language Notes**: Language-specific formatting / **Universal** principle.  
    **Rationale**: Visual structure; use auto-formatters (gofmt, black, prettier).

15. **Braces placement**: K&R style (opening on same line) preferred in many.  
    **Source**: Linux Kernel.  
    **Language Notes**: C/C++ / Adapt in others (Go/TS/Python often same-line).  
    **Rationale**: Consistency.

16. **One statement per line; avoid complex expressions**.  
    **Source**: JSF AV (expressions), Linux Kernel (no tricks).  
    **Language Notes**: **Universal**.  
    **Rationale**: Easier debugging and reviews.

17. **No trailing whitespace; consistent whitespace rules**.  
    **Source**: Linux Kernel, LLVM.  
    **Language Notes**: **Universal**.  
    **Rationale**: Clean diffs in version control.

### 4. Headers, Includes & Modularity (C++-specific with Universal Lessons)
18. **Headers should be self-contained** and include what they use.  
    **Source**: Google (IWYU), LLVM.  
    **Language Notes**: C++ / **Universal** (import organization in Go/Python/TS).  
    **Rationale**: Prevents hidden dependencies.

19. **Strict include ordering** and minimal includes.  
    **Source**: Google, LLVM.  
    **Language Notes**: C++.

20. **Use forward declarations** where possible to reduce dependencies.  
    **Source**: Google.  
    **Language Notes**: C++.

### 5. Functions & Control Flow (Focus: Predictability)
21. **No recursion** (or heavily restricted in critical paths).  
    **Source**: MISRA C++ 7-5-4, JSF AV.  
    **Language Notes**: C++ / Caution in Go (stack limits), Python (recursion depth).  
    **Rationale**: Bounded stack and analyzable call graphs.

22. **Prefer early returns/continues** to reduce nesting.  
    **Source**: LLVM, Linux Kernel.  
    **Language Notes**: **Universal**.  
    **Rationale**: Flatter, readable control flow.

23. **Conditions in if/loop must be explicit** (e.g., bool type where possible).  
    **Source**: MISRA C++ 5-0-13.  
    **Language Notes**: C++ / **Universal** (explicit comparisons).  
    **Rationale**: Avoids accidental assignments or implicit conversions.

24. **No side effects in conditions or expressions**.  
    **Source**: JSF AV (expressions), MISRA.  
    **Language Notes**: **Universal**.  
    **Rationale**: Prevents surprising behavior.

25. **Limit function parameters**; prefer clear structs/objects for many args.  
    **Source**: Linux Kernel, JSF AV.  
    **Language Notes**: **Universal**.

### 6. Error Handling & Reliability (Focus: Explicitness)
26. **No C++ exceptions** (use return codes or status).  
    **Source**: JSF AV (strict ban), Google (performance reasons).  
    **Language Notes**: C++ only.  
    **Rationale**: Predictable control flow and timing (critical in real-time).

27. **Explicit error handling**; no ignored return values.  
    **Source**: MISRA, JSF AV (fault handling).  
    **Language Notes**: **Universal** (Go: check errors; Python: handle exceptions narrowly; TS: proper async/try).  
    **Rationale**: Forces awareness of failure modes.

28. **No abort/exit** in library code; structured fault handling.  
    **Source**: JSF AV.  
    **Language Notes**: C++ / Adapt (no process kill in services).

29. **Assert liberally for invariants** (but not in release for performance-critical).  
    **Source**: LLVM.  
    **Language Notes**: **Universal** (debug-only).

### 7. Memory, Resources & Safety (Focus: No Undefined Behavior)
30. **Always initialize variables**.  
    **Source**: JSF AV (initialization), MISRA.  
    **Language Notes**: C++ / **Universal** (use defaults, type hints).  
    **Rationale**: Prevents uninitialized read UB.

31. **Restrict dynamic memory allocation** in critical paths (prefer static or RAII).  
    **Source**: JSF AV (memory allocation), MISRA.  
    **Language Notes**: C++ / Caution in embedded; in Go/Python/TS use pools or avoid in hot paths.  
    **Rationale**: Avoids fragmentation/leaks in bounded systems.

32. **Use smart pointers for ownership** (unique_ptr preferred); raw pointers only for observation.  
    **Source**: Google, modern extensions of JSF/MISRA thinking.  
    **Language Notes**: C++ / Analog: Go (no raw ptrs), Python (GC), TS (references carefully).  
    **Rationale**: Automatic resource management.

33. **Avoid pointer arithmetic** where possible.  
    **Source**: JSF AV, MISRA.  
    **Language Notes**: C++ / **Universal** (use slices/arrays safely in Go/Python).

34. **No unions/bit-fields** in safety-critical code (or with restrictions).  
    **Source**: JSF AV.  
    **Language Notes**: C++.

### 8. Classes, OO & Advanced Features (C++-specific)
35. **Prefer composition over deep inheritance**.  
    **Source**: Google, JSF AV (inheritance limits).  
    **Language Notes**: C++ / **Universal** (favor interfaces/composition in Go/TS/Python).

36. **Limit multiple/virtual inheritance**; careful with friends.  
    **Source**: JSF AV.  
    **Language Notes**: C++.

37. **Constructors**: Avoid heavy work; ensure full initialization.  
    **Source**: JSF AV, Google.  
    **Language Notes**: C++.

38. **const correctness** everywhere possible.  
    **Source**: JSF AV, MISRA 7-1-1.  
    **Language Notes**: C++ / **Universal** (readonly, const in others).

39. **Controlled template use** (avoid excessive bloat).  
    **Source**: JSF AV.  
    **Language Notes**: C++.

40. **No RTTI** (Run-Time Type Information).  
    **Source**: Google, LLVM.  
    **Language Notes**: C++.

### 9. Preprocessor, Macros & Portability
41. **Minimize macros** (use only for include guards or simple constants).  
    **Source**: JSF AV (preprocessing), MISRA.  
    **Language Notes**: C++ / **Universal** (avoid in Go/Python; use constants).

42. **Portable code**: No assumptions on data sizes, execution order, or endianness.  
    **Source**: JSF AV (portable code section), MISRA.  
    **Language Notes**: **Universal**.

43. **Use fixed-width types** (int32_t etc.) instead of plain int.  
    **Source**: MISRA 3-9-2.  
    **Language Notes**: C++ / **Universal** (explicit sizes where needed).

### 10. Comments, Documentation & Process
44. **Comments explain "why", not "what"**.  
    **Source**: JSF AV, Linux Kernel.  
    **Language Notes**: **Universal**.  
    **Rationale**: Code should be self-explanatory.

45. **Document all deviations** from rules with justification.  
    **Source**: JSF AV (deviation rules), MISRA.  
    **Language Notes**: **Universal** (code comments or tickets).  
    **Rationale**: Enforces thoughtful decisions.

46. **Follow surrounding style** when modifying existing code (Golden Rule).  
    **Source**: LLVM.  
    **Language Notes**: **Universal**.  
    **Rationale**: Maintains uniformity in collaborative projects.

47. **Treat compiler warnings as errors**; high warning levels.  
    **Source**: LLVM, JSF AV.  
    **Language Notes**: **Universal** (strict linters: golangci-lint, mypy, tsc --strict).

### 11. Additional Cross-Cutting Rules
48. **No global/non-const static variables** (or heavily restricted).  
    **Source**: Google, JSF AV.  
    **Language Notes**: **Universal** (avoid mutable globals).

49. **Prefer explicit types** over heavy auto/type inference in unclear cases.  
    **Source**: Google (cautious on auto); LLVM encourages when readable.  
    **Language Notes**: C++ / Balance in TS (type annotations), Python (type hints), Go (inferred ok).

50. **No self-modifying code** or tricky constructs.  
    **Source**: JSF AV.  
    **Language Notes**: C++ / **Universal** (avoid metaprogramming abuse in Python/TS).

51. **Use consistent project-wide style enforced by tools** (clang-format, gofmt, black, prettier).  
    **Source**: All (implicit via enforcement).  
    **Language Notes**: **Universal**.  
    **Rationale**: Reduces review friction.

52. **Assert "good taste"** — prioritize simplicity and readability over cleverness.  
    **Source**: Linux Kernel, LLVM.  
    **Language Notes**: **Universal**.  
    **Rationale**: Human-maintainable code survives longest.

53. **No using namespace** in headers (C++); avoid broad imports.  
    **Source**: Google, LLVM, MISRA 7-3-4.  
    **Language Notes**: C++ / Analog in others (qualified imports).

**How to Use This Sheet in Your Coding Journey**:
- **For C++ projects**: Layer JSF/MISRA for safety-critical + Google/LLVM for scale + Linux for systems simplicity.
- **For Go/TypeScript/Python**: Adopt universal rules (1–6, 7–12, 13–17, 21–25, 27, 30, 42, 44–48, 51–52). Use linters (e.g., golangci-lint with complexity plugins, ESLint/TSLint strict, pylint/mypy with length/complexity checks).
- **Implementation Tips**: Start with high-impact ones (function size, naming, initialization, error handling). Use auto-formatters and static analyzers. Document deviations. Balance strictness with productivity—safety-critical needs more rigidity than web apps.
- **Nuances & Edge Cases**: Over-application slows prototyping (use "strict mode" for core code). Modern languages (Go's error handling, Python's type hints, TS's strict mode) already embody many principles. In performance hotspots, justify relaxations (e.g., controlled recursion).
- **Tools**: clang-tidy (MISRA/JSF checks), Polyspace, go vet, mypy, tsc --noImplicitAny.

This sheet distills the best from all sources into actionable, cross-language wisdom. It will help you write **reliable, readable, and maintainable code** regardless of language or domain. Print it, integrate into your team's style guide, or reference during reviews. For full originals, consult the source documents (JSF PDF available online, MISRA requires purchase, others are public). Applying even 20–30 of these will noticeably elevate your codebase quality.

Yes, there are additional valuable safety rules and complementary guidelines from the **JSF AV C++ Coding Standards**, **MISRA C++:2008**, **Google C++ Style Guide**, **LLVM Coding Standards**, and **Linux Kernel Coding Style** that complement the previous 53-rule sheet. These focus on deeper safety aspects (e.g., eliminating undefined behavior, ensuring initialization and portability, restricting dangerous constructs), verifiability, determinism, and process discipline—areas that elevate code reliability beyond basic design and formatting.

I have expanded the sheet with **20 new rules** (bringing the total well over 70), selected for high impact, uniqueness, or strong safety/maintainability value. Many come directly from JSF AV (which has ~220 rules total) and MISRA C++:2008 (~228 rules), which emphasize a verifiable "safer subset" of C++. Others draw from Google/LLVM/Linux for scalability and pragmatism.

The new rules are integrated into expanded or new categories for clarity. Each includes:
- **Guideline**
- **Source(s)** (with specific references where possible, e.g., JSF AV Rule X, MISRA Rule Y-Z)
- **Language Notes** (C++ context; adaptations for Go, TypeScript, Python as **Universal** where principles transfer)
- **Rationale** (including nuances, trade-offs, edge cases, and implications for your coding journey)

This ensures completeness while maintaining organization. Universal rules are especially powerful for Go (explicitness, simplicity), TypeScript (strict typing, no implicit any), and Python (type hints, modular design).

### Expanded: 1. General Design & Complexity
(Previous rules 1–6 retained; adding deeper safety metrics)

**54. Restrict all code to very simple control flow constructs** — No `goto`, `setjmp`/`longjmp`, or direct/indirect recursion (except in justified, bounded cases with documentation).  
**Source**: JSF AV (flow control), MISRA C++ 7-5-4 (recursion advisory/required in contexts), Power of 10 influence on similar standards, Linux Kernel (short functions, low nesting).  
**Language Notes**: C++ / **Universal** (avoid recursion in Go/Python due to stack limits; no goto in most modern languages).  
**Rationale**: Simplifies static analysis, bounded stack usage, and full path coverage. Nuance: Recursion can be elegant but leads to stack overflows in embedded/real-time systems. Trade-off: Iterative solutions may use more code but are verifiable. Implication: Critical for safety; in web/services, it still prevents subtle stack issues.

**55. Give all loops a fixed upper bound** that is statically provable.  
**Source**: JSF AV (complexity & flow), Power of 10 Rule 2 (influential on safety standards).  
**Language Notes**: C++ / **Universal** (use `for` with known limits in Go/TS/Python; avoid `while(true)` without clear exit).  
**Rationale**: Prevents infinite loops and enables exhaustive testing/analysis. Edge case: Event-driven loops need careful design with timeouts.

### Expanded: 4. Headers, Includes & Modularity
**56. Do not declare a class, struct, or enum inside the definition of its own type**.  
**Source**: JSF AV Rule 141.  
**Language Notes**: C++-specific.  
**Rationale**: Improves readability and avoids self-referential declaration confusion.

### 12. New Category: Initialization & Declarations (Critical for Safety)
**57. All variables shall be initialized before use**.  
**Source**: JSF AV Rule 142 (revised from MISRA Rule 30), MISRA C++ 0-1-x series (unused/uninitialized handling).  
**Language Notes**: C++ / **Universal** (defaults in Go structs, type hints + initialization in Python/TS; strict mode in TS).  
**Rationale**: Prevents undefined behavior from reading garbage values. Nuance: Exceptions allowed only where declaration precedes meaningful init (e.g., input streams). Implication: Huge bug-preventer across languages; combine with const/readonly.

**58. Variables shall not be introduced until they can be initialized with meaningful values** (declare close to use).  
**Source**: JSF AV Rule 143.  
**Language Notes**: C++ / **Universal** (scope minimization in all languages).  
**Rationale**: Avoids partial objects or uninitialized access by clients.

**59. Use braces for non-zero initialization of arrays and structs**.  
**Source**: JSF AV Rule 144 (MISRA Rule 31).  
**Language Notes**: C++ / Adapt in TS/Python (explicit object literals).  
**Rationale**: Clarity and prevention of partial init surprises.

**60. In enumerator lists, do not explicitly initialize members other than the first unless all are initialized**.  
**Source**: JSF AV Rule 145 (MISRA Rule 32).  
**Language Notes**: C++-specific (enums).  
**Rationale**: Avoids maintenance errors when adding values.

### 13. New Category: Types, Constants & Portability (Preventing UB & Implementation Dependence)
**61. Use fixed-width integer types** (e.g., `int32_t`, `uint16_t` from `<cstdint>`) instead of plain `int`/`long`.  
**Source**: MISRA C++ 3-9-2, JSF AV portable code section.  
**Language Notes**: C++ / **Universal** (use explicit sizes where portability matters; Python `int` is arbitrary but use `typing` for clarity).  
**Rationale**: Ensures consistent size/sign across platforms/compilers. Trade-off: Slightly more verbose.

**62. The plain `char` type shall only be used for character values** (not arithmetic).  
**Source**: MISRA C++ 5-0-11 (or similar), JSF AV types section.  
**Language Notes**: C++ / Caution in others (explicit signed/unsigned where needed).  
**Rationale**: `char` signedness is implementation-defined; prevents subtle bugs.

**63. No octal constants** (except zero); prefer decimal or uppercase hex.  
**Source**: JSF AV Rule 149 (MISRA Rule 19).  
**Language Notes**: C++ / **Universal**.  
**Rationale**: Octal (leading zero) causes confusion and errors.

**64. Floating-point shall comply with a defined standard** (e.g., IEEE 754); do not manipulate bit representations directly.  
**Source**: JSF AV Rule 146–147 (MISRA Rule 15–16).  
**Language Notes**: C++ / **Universal** (avoid bit hacks on floats in any language).  
**Rationale**: Portability and prevention of NaN/rounding surprises.

**65. No unions or bit-fields** (or heavy restrictions with justification).  
**Source**: JSF AV Rule on unions/bit-fields.  
**Language Notes**: C++-specific.  
**Rationale**: Type punning and alignment issues lead to UB or portability problems. Edge case: Low-level hardware interaction may require documented exceptions.

### 14. New Category: Operators, Expressions & Conversions
**66. No implicit narrowing conversions** that change signedness or lose precision.  
**Source**: MISRA C++ 5-0-4, JSF AV type conversions.  
**Language Notes**: C++ / **Universal** (explicit casts or checks in Go/TS/Python).  
**Rationale**: Prevents silent data loss or sign errors.

**67. Avoid side effects in expressions** (especially in conditions or function args).  
**Source**: JSF AV expressions section, MISRA flow control.  
**Language Notes**: **Universal**.  
**Rationale**: Makes behavior predictable and easier to analyze.

**68. Use `const` (or equivalent) liberally** for variables/parameters that are not modified.  
**Source**: MISRA C++ 7-1-1, Google (const correctness), JSF AV.  
**Language Notes**: C++ / **Universal** (`readonly` in TS, `final`/`const` in Python via conventions).  
**Rationale**: Enables compiler checks and documents intent.

### 15. New Category: Memory, Allocation & Resources (Deeper Safety)
**69. Restrict dynamic memory allocation** after initialization in critical paths (prefer static/pre-allocated).  
**Source**: JSF AV memory allocation section (strong restrictions), Google/LLVM performance rationale.  
**Language Notes**: C++ / **Universal** (use object pools or avoid heap in hot paths; Go/Python GC helps but doesn't eliminate fragmentation risks in real-time).  
**Rationale**: Prevents leaks, fragmentation, non-determinism, and exhaustion in bounded systems. Nuance: Modern C++ smart pointers (unique_ptr) are safer alternatives where allowed.

**70. No self-modifying code** or runtime code generation.  
**Source**: JSF AV Rule 2 (explicit ban).  
**Language Notes**: C++ / **Universal** (avoid `eval`, dynamic code loading in Python/TS where possible).  
**Rationale**: Impossible to analyze statically; huge security/safety risk.

### 16. New Category: Fault Handling, Libraries & Environment
**71. No use of `abort()`, `exit()`, or `signal.h`/`time.h`** in library or core code (structured recovery instead).  
**Source**: JSF AV fault handling & libraries section.  
**Language Notes**: C++ / Adapt (no process termination in services; use proper error propagation in Go/Python/TS).  
**Rationale**: Ensures controlled, verifiable fault handling with bounded timing.

**72. Restrict standard library usage** to approved/certified subsets (avoid full STL in strict safety contexts if exceptions or non-determinism possible).  
**Source**: JSF AV libraries, Google (disallowed stdlib features in some cases).  
**Language Notes**: C++-specific with universal caution on heavy dependencies.  
**Rationale**: Many std features assume exceptions or have implementation-defined behavior.

### 17. New Category: Process, Deviations & Verification
**73. Every deviation from a "shall" rule must be documented in the file** (with justification); deviations require formal approval.  
**Source**: JSF AV Rules 5–6 (deviation process).  
**Language Notes**: **Universal** (use code comments, tickets, or architecture decision records).  
**Rationale**: Prevents casual rule-breaking; enforces accountability and review. Implication: Builds discipline—document even in non-critical projects.

**74. Treat all compiler warnings as errors**; enable high warning levels.  
**Source**: LLVM, JSF AV (environment), Google/LLVM.  
**Language Notes**: **Universal** (strict linters: `golangci-lint --strict`, `tsc --strict`, `mypy --strict`).  
**Rationale**: Catches subtle issues early.

**75. Beware of non-determinism** (e.g., pointer ordering in sets/maps, unstable sorts, hash order). Sort explicitly when order matters.  
**Source**: LLVM Coding Standards (explicit section on non-determinism).  
**Language Notes**: **Universal** (Go maps are random; Python 3.7+ dicts ordered but don't rely on it).  
**Rationale**: Ensures reproducible builds/tests/behavior—critical for debugging and certification.

**76. Assert liberally for invariants and preconditions** (debug builds); use `llvm_unreachable` or equivalent for impossible cases.  
**Source**: LLVM (assertions), JSF AV fault handling.  
**Language Notes**: **Universal** (debug asserts in all languages; `unreachable` patterns).  
**Rationale**: Documents assumptions and catches violations early without runtime cost in release.

