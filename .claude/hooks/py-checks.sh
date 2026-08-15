#!/usr/bin/env bash
# PostToolUse hook (Write|Edit|MultiEdit): lint-fix, format, and type-check an
# edited Python file — in that order — so the working tree stays CI-green.
#
# Runs the same ruff + ty gates CI enforces. `ruff check --fix` and ruff format
# mutate the file in place; anything unfixable (undefined names, type errors) is
# fed back to Claude via exit code 2 so it gets corrected immediately.
set -uo pipefail

# Team-shared: if a teammate lacks jq, degrade to a silent no-op instead of
# erroring on every edit (see .claude/hooks/README.md for prerequisites).
command -v jq >/dev/null 2>&1 || exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# The edited file path arrives in the tool-call JSON on stdin.
file_path="$(jq -r '.tool_input.file_path // empty')"
[ -z "$file_path" ] && exit 0
case "$file_path" in
  *.py) ;;
  *) exit 0 ;;   # only Python files are formatted/type-checked
esac
[ -f "$file_path" ] || exit 0

# 1. Auto-fix lint. Fixes rewrite code structure (`I` re-sorts imports, `UP`
#    rewrites syntax), so the formatter has to run after them — see step 2.
#    Output is discarded here on purpose; step 3 is what reports.
uv run ruff check --fix "$file_path" >/dev/null 2>&1

# 2. Format the (possibly rewritten) file in place; noise suppressed.
#    Lint-fix-then-format is Astral's recommended order, so the bytes on disk
#    are formatted whatever the fixer did. Note this is insurance, not a repair:
#    no *safe* fix in the current rule set (E/F/I/UP) was found to leave
#    unformatted output — ruff's isort fix self-formats, and the expanding
#    rewrites like UP031 are unsafe fixes a bare `--fix` never applies.
uv run ruff format "$file_path" >/dev/null 2>&1

# 3. Verify the FINAL bytes. Steps 1 and 2 both mutate the file, so lint must be
#    re-checked after them: a verdict captured before formatting describes
#    content that no longer exists on disk, and its line numbers point into the
#    pre-format file, sending Claude to edit the wrong lines. Reporting only
#    this last check keeps the hook's verdict and the file in agreement.
#
#    Newlines are real (via $'...') rather than literal \n escapes, so the
#    report can be printed with %s. Using %b would re-interpret backslashes
#    inside the tool output itself, mangling diagnostics that quote source such
#    as re.compile(r"\d+\t").
problems=""
if ! check_out="$(uv run ruff check "$file_path" 2>&1)"; then
  problems+="ruff (unfixable lint issues):"$'\n'"${check_out}"$'\n\n'
fi

# 4. Type-check. ty's unit of analysis is the whole project, not just this file.
if ! ty_out="$(uv run ty check 2>&1)"; then
  problems+="ty (type errors):"$'\n'"${ty_out}"$'\n'
fi

if [ -n "$problems" ]; then
  printf 'Post-edit checks failed for %s:\n\n%s' "$file_path" "$problems" >&2
  exit 2   # surface the failures to Claude as actionable feedback
fi
exit 0
