#!/usr/bin/env bash
# Stop hook: refresh codebase navigation after a turn that changed code.
#
# Cost (measured 2026-08-16):
#   codegraph sync .   0.7s  — incremental; safe to run unconditionally
set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}" || exit 0

CODEGRAPH="${CODEGRAPH_BIN:-/home/harmeet/.local/bin/codegraph}"
[[ -x "$CODEGRAPH" ]] || CODEGRAPH="$(command -v codegraph 2>/dev/null || true)"

# codegraph — cheap, and closes the stale-symbol-selection window described in
#    orient/SKILL.md "Deep Internals".
[[ -n "$CODEGRAPH" && -d .codegraph ]] && timeout 120 "$CODEGRAPH" sync -q . >/dev/null 2>&1
exit 0
