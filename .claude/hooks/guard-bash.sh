#!/usr/bin/env bash
# PreToolUse hook (Bash): refuses commands this project must never run
# unattended. Exit 2 blocks the command and tells Claude why.
cmd=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input", {}).get("command", ""))' 2>/dev/null) || exit 0

block() { printf 'Blocked by .claude/hooks/guard-bash.sh: %s\n' "$1" >&2; exit 2; }

# The GPU is shared with the user's ollama models; they stop it themselves.
if printf '%s' "$cmd" | grep -qiE '(pkill|killall|kill)[^|;&]*ollama|ollama +stop|systemctl +(stop|restart|kill)[^|;&]*ollama'; then
  block "never stop ollama; ask the user to free the GPU and state the VRAM conditions instead."
fi

# Releases are tagged by semantic-release in CI (see CLAUDE.md).
if printf '%s' "$cmd" | grep -qE 'git +tag +(-[a-z]+ +)*v[0-9]|git +push[^|;&]*( --tags| v[0-9]| refs/tags/v[0-9])'; then
  block "don't create or push v* tags by hand; CI tags releases from Conventional Commits."
fi

# History on main is shared and drives the release notes.
if printf '%s' "$cmd" | grep -qE 'git +push[^|;&]*(--force|-f |--force-with-lease)[^|;&]*main|git +push[^|;&]*\+main'; then
  block "no force-pushes to main."
fi
exit 0
