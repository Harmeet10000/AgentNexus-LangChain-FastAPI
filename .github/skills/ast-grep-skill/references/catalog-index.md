## Catalog Patterns (`references/catalog/`)

Ready-to-use rule templates for common linting tasks:

| File | Purpose | Rule IDs |
|------|---------|----------|
| `async-function.md` | Detect `await` inside loops | `no-await-in-loop` |
| `no-console.md` | Ban `console.*` in production | `no-console` |
| `react-hooks.md` | Enforce Rules of Hooks | `react-hooks-rules-of-hooks` |
| `secure-coding.md` | Detect `eval`, `innerHTML`, shell exec | `no-eval`, `no-inner-html`, `no-shell-exec` |
| `migration-helper.md` | CJS→ESM (`require`→`import`) | `require-to-import`, `no-full-lodash` |
| `type-patterns.md` | Legacy types (`any`, `React.FC`) | `no-react-fc`, `no-explicit-any` |
| `ban-keyword.md` | Ban arbitrary keywords/patterns | `no-debugger`, `no-todo`, `no-var` |

Each file contains full rule YAML — copy to your project's rule directory.

