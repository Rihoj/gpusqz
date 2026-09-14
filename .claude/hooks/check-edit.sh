#!/usr/bin/env bash
# PostToolUse hook (Edit|Write|MultiEdit): quick checks on the file Claude
# just changed. Problems go back to Claude (exit 2 + stderr); a missing
# tool means the check is skipped, never a failure.
#   src/vk/*.comp, *.glsl   compile the affected shaders (both lane builds)
#   tests/**/*.sh           bash -n, and no GNU-only tools (macOS runs them)
#   .github/workflows/*.yml actionlint if installed, else a YAML parse
root="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
file=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))' 2>/dev/null) || exit 0
[ -n "$file" ] && [ -f "$file" ] || exit 0
rel="${file#"$root"/}"

case "$rel" in
  src/vk/*.comp | src/vk/*.glsl)
    exec bash "$root/.claude/hooks/check-shaders.sh" "$root" "$rel"
    ;;

  tests/*.sh | tests/*/*.sh)
    if ! out=$(bash -n "$file" 2>&1); then
      printf '%s: bash syntax error:\n%s\n' "$rel" "$out" >&2
      exit 2
    fi
    # These scripts run on macOS (bash 3.2, BSD tools) too; see CLAUDE.md.
    hits=$(grep -nE 'stat -c|base64 -w|(^|[;&| ])truncate |sed -i|mapfile|readarray|\$\{[A-Za-z_]+,,|declare -A|date \+%s%N' "$file" | grep -v '^[0-9]*: *#')
    if [ -n "$hits" ]; then
      printf '%s uses GNU/bash-4-only constructs; tests must run on macOS bash 3.2 with BSD tools:\n%s\n' "$rel" "$hits" >&2
      exit 2
    fi
    ;;

  .github/workflows/*.yml)
    if command -v actionlint > /dev/null; then
      if ! out=$(cd "$root" && actionlint -shellcheck= "$rel" 2>&1); then
        printf 'actionlint %s:\n%s\n' "$rel" "$out" >&2
        exit 2
      fi
    elif ! out=$(python3 -c 'import sys, yaml; yaml.safe_load(open(sys.argv[1]))' "$file" 2>&1); then
      case "$out" in *"No module named"*) ;; *) printf '%s is not valid YAML:\n%s\n' "$rel" "$out" >&2; exit 2 ;; esac
    fi
    ;;
esac
exit 0
