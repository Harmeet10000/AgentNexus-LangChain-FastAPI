#!/usr/bin/env bash
# PostToolUse(Edit|Write|MultiEdit) hook: keep the codegraph index in step with
# the edit that just landed.
#
# Why per-edit and not only at Stop: codegraph's freshness guarantee is split.
# `codegraph_explore` re-reads source from disk at call time, so the text is
# current — but WHICH symbols it selects comes from the index, which lags by the
# watcher's ~1s. Mid-turn, that yields byte-perfect source for a stale symbol
# set, and nothing in the output marks it. `codegraph sync -q .` costs ~0.7s and
# closes the window. See .opencode/skills/orient/SKILL.md "Deep Internals" #3.
#
# Only source edits pay the 0.7s — a markdown or JSON write exits immediately.
# Contract: hook JSON on stdin; exit 0 always so a failed sync never blocks Edit.
set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}" || exit 0
[[ -d .codegraph ]] || exit 0

CODEGRAPH="${CODEGRAPH_BIN:-/home/harmeet/.local/bin/codegraph}"
[[ -x "$CODEGRAPH" ]] || CODEGRAPH="$(command -v codegraph 2>/dev/null || true)"
[[ -n "$CODEGRAPH" ]] || exit 0

path="$(python3 -c 'import json,sys
d = json.load(sys.stdin)
r = d.get("tool_response") or {}
i = d.get("tool_input") or {}
print((r.get("filePath") if isinstance(r, dict) else None) or i.get("file_path") or "")' \
  2>/dev/null || true)"

case "$path" in
  *.py|*.pyi|*.go|*.ts|*.tsx|*.js|*.jsx|*.rs|*.java) ;;
  *) exit 0 ;;
esac

timeout 120 "$CODEGRAPH" sync -q . >/dev/null 2>&1
exit 0
