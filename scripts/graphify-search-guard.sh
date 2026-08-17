#!/usr/bin/env bash
# PreToolUse(Bash) guard: forward to `graphify hook-guard search` ONLY when the
# command is actually a search.
#
# Why this wrapper exists: Claude Code hook matchers match the TOOL NAME, not the
# command string. A bare `Bash` matcher therefore fired graphify's "MANDATORY:
# run graphify query first" notice on `git status`, `uv run pytest`, `python3`,
# and `ls` — every shell call in the session paid the tax and read advice that
# did not apply to it.
#
# Contract: hook JSON arrives on stdin; .tool_input.command holds the command.
# Exit 0 silently to allow the call through unannotated.
set -uo pipefail

GRAPHIFY="${GRAPHIFY_BIN:-/home/harmeet/.local/bin/graphify}"
[[ -x "$GRAPHIFY" ]] || GRAPHIFY="$(command -v graphify 2>/dev/null || true)"
[[ -n "$GRAPHIFY" ]] || exit 0

payload="$(cat)"
cmd="$(printf '%s' "$payload" | python3 -c \
  'import json,sys; print((json.load(sys.stdin).get("tool_input") or {}).get("command",""))' \
  2>/dev/null || true)"
[[ -n "$cmd" ]] || exit 0

# Search tools only, matched as a bare word so `git grep`/`xargs rg` still count
# but `ripgrep-config`, `findutils`, or a filename containing "grep" do not.
if printf '%s' "$cmd" | grep -qEw 'rg|grep|egrep|fgrep|find|ast-grep|sg'; then
  printf '%s' "$payload" | "$GRAPHIFY" hook-guard search
fi
exit 0
