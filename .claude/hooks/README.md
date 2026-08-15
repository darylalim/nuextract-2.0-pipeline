# Claude Code hooks (team-shared)

These hooks run automatically inside every teammate's Claude Code session (they
do **not** run in CI). They enforce the same gates CI does, at edit time.

Wired up in [`../settings.json`](../settings.json).

## Prerequisites

| Tool | Why | Install |
| --- | --- | --- |
| [`uv`](https://docs.astral.sh/uv/) | runs `ruff` / `ty` via `uv run` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [`jq`](https://jqlang.github.io/jq/) | parses the tool-call JSON on stdin | `brew install jq` |

If `jq` is missing, `py-checks` **degrades to a silent no-op** (it won't error or
spam) — you lose its benefit until you install it. Secret protection does not
depend on `jq`: it is enforced by `permissions.deny` in `../settings.json`, not
by a hook.

## The hooks

| Script | Event | Behavior |
| --- | --- | --- |
| [`py-checks.sh`](py-checks.sh) | `PostToolUse` (Write/Edit) | On a `.py` edit: `ruff check --fix` → `ruff format` (both silent), then a final `ruff check` + `ty check` whose output *is* reported back to Claude on failure. Lint-fix, format, and type-check are one ordered script to avoid a parallel-hook race on the file. |

**Why lint-fix runs before format:** `ruff check --fix` rewrites code structure —
`I` re-sorts imports, `UP` rewrites syntax — so the formatter runs after it and
the bytes on disk are formatted regardless of what the fixer did. This ordering
is what Astral's own tooling guidance recommends (it is the order the
[ruff-pre-commit](https://github.com/astral-sh/ruff-pre-commit) hooks run in);
note it is *not* stated at the docs root.

**Why lint is checked twice:** steps 1 and 2 both mutate the file, so the
reported verdict comes from a **third** pass over the final bytes. Capturing
`ruff check --fix` output at step 1 instead would report lint computed against
content that no longer exists, with line numbers pointing into the pre-format
file — sending Claude to edit the wrong lines, and letting the hook exit 0 while
a formatter reflow left the file lint-dirty.

Worth knowing what the reorder is *not*: no safe fix in the current rule set
(`E`/`F`/`I`/`UP`) was found that actually leaves unformatted output. Ruff's
isort fix emits formatted code itself, and the expanding rewrites (e.g. `UP031`,
`%` → `.format()`) are classified **unsafe** and so are never applied by a bare
`--fix`. The ordering is insurance against future rule or config changes, not a
repair for an observed CI failure.

## Secrets are handled by permissions, not a hook

`.env*` and `.streamlit/secrets.toml` hold credentials (`.env` holds `HF_TOKEN`)
and are gitignored. They are protected declaratively in
[`../settings.json`](../settings.json):

```json
"permissions": {
  "deny": ["Read(/.env*)",  "Edit(/.env*)",  "Read(**/.env*)",  "Edit(**/.env*)",
           "Read(/.[eE][nN][vV]*)",   "Edit(/.[eE][nN][vV]*)",
           "Read(**/.[eE][nN][vV]*)", "Edit(**/.[eE][nN][vV]*)",
           "Read(/.streamlit/secrets.toml)",   "Edit(/.streamlit/secrets.toml)",
           "Read(**/.streamlit/secrets.toml)", "Edit(**/.streamlit/secrets.toml)"]
}
```

This replaced a `PreToolUse` hook. It is **broader on tool coverage** — permission
rules sit above tool dispatch, so they also deny **`Read`**, which a matcher
scoped to `Write|Edit|MultiEdit` never covered — but it is not broader on every
axis, and the rule list is shaped to close the three places it would otherwise be
*narrower*.

**Why each path is listed twice (anchoring).** Rule anchoring is not intuitive
([docs](https://code.claude.com/docs/en/permissions#read-and-edit)):

| Pattern | Anchors at |
| --- | --- |
| `/.env*` | `<project root>` — the settings source, for `.claude/settings.json` |
| `**/.env*` or bare `.env*` | any depth **at or under the current directory** |
| `./.env*` | `<cwd>` **only** |

`./.env` — the form in Claude Code's own docs example — silently misses the
project's `.env` in a session started from a subdirectory, because it anchors to
the cwd rather than the project. The `/`-anchored rule is the cwd-independent
one; the `**/` rule adds any nested secrets. Both are listed because neither
alone covers both cases. The old hook's `*/.env` suffix match was cwd-independent,
so getting this wrong would have been a real narrowing.

**Why `.env*` and not `.env` + `.env.*` (glob shape).** In gitignore syntax `*`
matches zero or more characters within a segment, so a single `.env*` subsumes
`.env`, `.env.local`, and — unlike `.env.*` — `.envrc`, a very common home for
`HF_TOKEN`. The old hook had the same `.env.*` gap; this is the one place the
replacement is deliberately *wider* than what it replaced.

**Why the character classes (case).** The old hook set `shopt -s nocasematch` to
catch `.ENV` on case-insensitive filesystems (the macOS default). Gitignore
syntax is case-sensitive and the permissions docs never claim otherwise for
Read/Edit — they state case-insensitivity explicitly only for PowerShell aliases
and WebFetch domains. `.[eE][nN][vV]*` restores that coverage. **The plain
`.env*` rules are kept alongside deliberately**: if this matcher turns out not to
support character classes, the plain rules still cover the lowercase name, so the
class rules can only ever add coverage, never silently replace it.
`.streamlit/secrets.toml` has no case-class variant — add one if that ever
matters.

**There is no `Write(...)` rule, deliberately.** Claude Code checks file
permissions against `Edit(path)` and `Read(path)` rules *only*; a `Write(path)`,
`MultiEdit(path)`, or `NotebookEdit(path)` rule is accepted, never consulted, and
warns at startup. `Edit` rules already apply to every built-in file-editing tool,
and a `Read` deny additionally blocks Edit and Write on that path (Claude Code
≥2.1.228). `Edit` is listed alongside `Read` because `NotebookEdit` is not
covered by a `Read` deny.

## Known limits

**Accident guardrail, not a security boundary — and the bypass is live in this
repo.** Deny rules cover the built-in file tools and the file commands Claude
Code recognizes in Bash (`cat`, `head`, `sed`, …), but *not* arbitrary
subprocesses. `.claude/settings.local.json` allowlists `Bash(uv run *)`, so
`uv run python -c "print(open('.env').read())"` prints `HF_TOKEN` straight into
context and no deny rule fires. Read-rule coverage of Grep/Glob is likewise
documented as only a "best-effort attempt". These rules stop the *accident*
(Claude opening or clobbering the file in the course of ordinary work); they do
not stop a determined path. For OS-level enforcement across all processes,
[enable the sandbox](https://code.claude.com/docs/en/sandboxing).

**Template files are caught too.** `.env*` also matches `.env.example` /
`.env.template`. If you add one to document that `HF_TOKEN` is required, Claude
will not be able to read or edit it, and **a deny rule cannot be overridden by an
allow rule** in `.claude/settings.local.json` — deny always wins. Either name it
without the leading dot (`env.example`) or narrow the rules at that point.

## Notes

- **Activation:** Claude Code snapshots hooks at session start. After pulling
  changes to these files, restart the session (or run `/hooks` to confirm they
  loaded) before they take effect.
- **Personal overrides** go in `.claude/settings.local.json`, which stays
  gitignored — use it for machine-specific tweaks without touching the shared
  set. Note this does *not* extend to loosening the deny rules above.
- **Tests are not hooked, and CI does not cover every branch.** A `Stop` hook
  that ran the suite locally was removed: it duplicated the CI gate, cost ~9s per
  turn, and keyed on `git status` (working-tree state) rather than what the turn
  actually changed, so any dirty `.py` file made it fire on every turn —
  including pure-conversation ones. But `.github/workflows/ci.yml` triggers only
  on `push` to `main` and PRs *targeting* `main`, so **a pushed feature branch
  with no PR runs no tests at all**. Run `uv run pytest` before pushing, or drop
  the `branches:` filter from the `push:` trigger to close the window properly.
