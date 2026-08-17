#!/usr/bin/env bash
# Stop hook: refresh codebase navigation after a turn that changed code.
#
# Split by cost (measured 2026-08-16):
#   codegraph sync .   0.7s  — incremental; safe to run unconditionally
#   graphify update .   25s  — full AST re-extract + Leiden clustering
#
# The 25s job runs ONLY when a source file is newer than graph.json, so
# doc-only and conversation-only turns cost ~0.7s instead of ~26s.
set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}" || exit 0

CODEGRAPH="${CODEGRAPH_BIN:-/home/harmeet/.local/bin/codegraph}"
GRAPHIFY="${GRAPHIFY_BIN:-/home/harmeet/.local/bin/graphify}"
[[ -x "$CODEGRAPH" ]] || CODEGRAPH="$(command -v codegraph 2>/dev/null || true)"
[[ -x "$GRAPHIFY" ]] || GRAPHIFY="$(command -v graphify 2>/dev/null || true)"

# 1. codegraph — cheap, and closes the stale-symbol-selection window described in
#    orient/SKILL.md "Deep Internals" #3.
[[ -n "$CODEGRAPH" && -d .codegraph ]] && timeout 120 "$CODEGRAPH" sync -q . >/dev/null 2>&1

# 2. graphify — only if code actually moved.
GRAPH=graphify-out/graph.json
if [[ -n "$GRAPHIFY" && -f "$GRAPH" ]]; then
  if find src tests -type f \( -name '*.py' -o -name '*.go' \) -newer "$GRAPH" -print -quit \
       2>/dev/null | grep -q .; then
    # No --no-cluster: it writes raw extraction and drops every community.
    timeout 600 "$GRAPHIFY" update . >/dev/null 2>&1
  fi
fi
exit 0
