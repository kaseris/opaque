# Opaque

Project-agnostic tool for tracking LLM classifier / structured-extraction performance over
time, tied to prompt versions. Point it at a target project's git repo, onboard it, and it
manages evaluation runs across one or more onboarded projects — built on **MLflow** for
tracking + its built-in UI, with a custom Excel report for extraction and a basic relabeling
UI for cleaning gold data.

`CLAUDE.md` holds the full v1 specification; section references (§) below point into it.

## Install

Uses [`uv`](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
```

## Quickstart (against the bundled demo)

`examples/demo_project/` is a self-contained fake target repo (its own git repo) with an
extraction tool (`invoices`) and a classification tool (`doc_type`).

```bash
# Run + score + log + (for extraction) write the Excel report
uv run opaque run examples/demo_project --tool invoices
uv run opaque run examples/demo_project --tool doc_type

# Browse runs in MLflow's UI
uv run opaque ui                 # -> http://127.0.0.1:5000

# Clean / fill gold labels in the browser, one commit per session (§9)
uv run opaque relabel examples/demo_project --tool invoices   # -> http://127.0.0.1:8000
```

## Commands

| Command | Purpose |
|---|---|
| `opaque onboard <repo> [--project NAME]` | Scaffold `.opaque/config.yaml`, or summarize an existing one (§3) |
| `opaque run <repo> --tool NAME [--prompt role=path …] [--allow-dirty] [--tracking-uri …]` | Combined predict + score + report + log (§2) |
| `opaque report <repo> --tool NAME --out report.xlsx` | On-demand Excel report, no MLflow logging (§10) |
| `opaque ui [--tracking-uri ./mlruns] [--port 5000]` | Launch MLflow's built-in run browser (§8.3) |
| `opaque relabel <repo> --tool NAME [--port 8000]` | Launch the relabeling UI (§9) |
| `opaque hook install <repo> [--gate] [--comment] [--remote origin] [--tracking-uri …]` | Install the pre-push hook that triggers local evals |
| `opaque hook status\|uninstall <repo>` | Inspect or remove that hook |
| `opaque check --repo <repo> [--stdin] [--gate] [--comment] [--out FILE]` | Evaluate the tools whose prompts a push touches |

## PR-triggered evaluation (running locally)

The natural moment to evaluate a prompt change is when it is proposed for review — but an
opaque run needs the target project's model credentials, its eval data, and an MLflow store
that persists across runs, none of which survive an ephemeral CI runner. So the **trigger is
PR-shaped while execution stays local**:

```bash
uv run opaque hook install /path/to/your/repo --tracking-uri ~/opaque/mlruns
```

That installs a `pre-push` hook. Every PR creation and every PR update is preceded by exactly
one push from your machine, so the hook fires once per PR-affecting event — unlike
`post-commit`, which would evaluate half-finished prompt states nobody will act on, and bill
you for each.

```
$ git push origin tweak-extraction
opaque: evaluated 1 tool(s) on tweak-extraction
  • invoices: field_accuracy=0.6231 (baseline 0.5926, ▲ +0.0305)  3 labeled
```

- **Only fires when it matters.** A push is evaluated only if it changes a file listed under a
  tool's `prompts:` or its `field_schema_path`. Docs-only pushes cost nothing.
- **Eval data is deliberately not a trigger.** It is git-versioned in the same repo and the
  relabeling UI commits to it (§9.2), so triggering on it would fire runs whose deltas reflect
  moved gold rather than a changed prompt. If a push changes *both*, the run happens and the
  summary says the delta is not prompt-attributable.
- **Non-blocking by default.** A regression is reported, not enforced; the push proceeds. Add
  `--gate` at install time to block instead, and `--tolerance` to allow a small drop. Skip any
  single push with `OPAQUE_SKIP=1 git push`.
- **`--comment`** posts the summary to the branch's PR via `gh`, best-effort (no `gh`, no PR,
  or no auth are all normal and never fail a push).

### What actually fires it

`pre-push` is bound to the *push operation*, not to a branch, and by default git runs it for
every remote. Measured behaviour:

| Action | Evaluated? |
|---|---|
| `git push origin <branch>` | ✅ |
| `git push backup <branch>` (another remote) | ❌ filtered by `--remote`, default `origin` |
| `git push --tags` / any `refs/tags/*` push | ❌ never a PR event |
| `git push --dry-run` | ⚠️ **yes** — git gives the hook no way to tell a dry run apart |
| `git push` with nothing to send | ✅ runs, but git passes no refs, so it exits immediately |
| Pushing a branch you do not have checked out | ❌ declined — the eval scores the working tree |
| Opening/merging a PR in the GitHub UI, "Update branch", web edits | ❌ nothing runs locally |

The remote filter lives in the hook's shell stub, above the `exec`, so an irrelevant push costs
nothing rather than paying interpreter startup to discover it is irrelevant. Note that git only
passes a remote *name* as `$1` when you push by name — `git push git@host:repo.git main` passes
the URL, which does not match and is skipped. Use `--remote ''` to evaluate every remote.

For a dry run that you do not want scored: `OPAQUE_SKIP=1 git push --dry-run`.

### Where runs land

`prompt_impact.html` diffs consecutive runs *within one experiment* in wall-clock order and has
no branch awareness, so concurrent branches writing to the same experiment would produce deltas
between unrelated prompt states. Runs are therefore routed:

| Pushed branch | Experiment |
|---|---|
| integration branch (`main`) | `{project}/{tool}` — the canonical, serialized history |
| any other branch | `{project}/{tool}@{branch}` — isolated |

Baselines always come from the canonical experiment, so a branch's delta answers "versus what
is on `main` today" regardless of where its own run is logged.

**Known gap:** merging through the GitHub UI creates the merge commit server-side, so no hook
fires and the canonical experiment never gets that run — baselines then go stale. Until a
`post-merge` hook exists, write it by hand after pulling `main`:

```bash
opaque check --repo . --base ORIG_HEAD --head HEAD
```

`--base` is required here: on the integration branch the default base is
`merge-base(HEAD, main)`, which *is* `HEAD`, so the diff comes out empty.

### Interpreter caveat

Git runs hooks with a minimal environment that does **not** include an activated virtualenv, so
a `command:` beginning with a bare `python` can work by hand and fail from the hook. Name an
interpreter that resolves outside your shell (`python3`), or an absolute path to the project's
own venv when the eval script has dependencies.

`--prompt` selects which prompt file fills each role at run time (repeatable, e.g.
`--prompt system=prompts/system.txt --prompt extraction=prompts/extraction.txt`); it
defaults to the tool's declared prompts.

## How it fits together

```
.opaque/config.yaml (in target repo)  ──►  runner renders the eval-script command
prompts + eval data (git-versioned)        from the invocation contract, runs it,
        │                                   loads per-sample JSON (§5), and scores it
        ▼                                                    │
  versioning (§4): git commit +                              ▼
  content hash per prompt / eval set     metrics (§6): classification | extraction
        │                                                    │
        └───────────────►  MLflow run (params · metrics · artifacts, §8)  ──►  report.xlsx (§10)
```

- **Config** (`src/opaque/config/`) — onboarding config + the field-matching override schema
  (§3, §7.1), read-only, committed in the target repo.
- **Versioning** (`src/opaque/versioning/`) — per-file `git log -1` provenance + `sha256[:12]`
  identity + dirty flag; aggregate `prompt_bundle_hash` (§4).
- **Matching** (`src/opaque/matching/`) — flatten to `json_path`s, infer types, compare
  (numeric/date/boolean/string), union-of-paths status = correct/incorrect/missing/extra (§7).
- **Metrics** (`src/opaque/metrics/`) — pluggable registry; `classification` + `extraction` (§6).
- **Runner** (`src/opaque/runner/`) — combined predict+score; blocks on dirty prompts unless
  `--allow-dirty` (§4.1).
- **Tracking / Report / Relabel** — MLflow logging (§8), the 3-sheet Excel report (§10), and
  the FastAPI + Vue relabeling UI (§9).
- **Hooks** (`src/opaque/hooks/`) — the pre-push trigger: push-range parsing, change detection
  against tracked prompt paths, branch-scoped experiment routing, and the PR summary.

## Onboarding config

Lives at `.opaque/config.yaml` in the *target* repo (see `examples/demo_project/.opaque/config.yaml`):

```yaml
project: acme_demo
tools:
  - name: invoices
    task_type: extraction            # extraction | classification
    prompts:
      system: prompts/system.txt     # role -> repo-relative path
      extraction: prompts/extraction.txt
    eval_data: eval_data/invoices    # dir of per-sample JSON, or one .json file
    field_schema_path: field_schema.yaml   # extraction override schema (§7.1)
    model_name: demo-extractor-v1
    temperature: 0.0
    invocation:
      # Placeholders: {input} {output_dir} {prompt.<role>} {model} {temperature}
      command: >-
        python3 eval_script.py --task extraction --input {input} --out {output_dir}
        --system {prompt.system} --extraction {prompt.extraction}
```

The eval script must write the per-sample JSON contract of §5 into `{output_dir}`.

## Development

```bash
uv run pytest                        # backend test suite

# Relabeling frontend (Vue 3 + Vite + Tailwind 4 + PrimeVue)
cd src/opaque/relabel/web
npm install
npm run build                        # -> dist/, served by FastAPI
npm run dev                          # hot-reload dev server (proxies /api to :8000)
```

## Notes / decisions

- **Python 3.12** (pinned): MLflow's scientific stack does not yet have reliable wheels for
  3.13/3.14.
- **MLflow file store** (`./mlruns`): MLflow 3 gates the filesystem backend, so Opaque opts in
  via `MLFLOW_ALLOW_FILE_STORE=true` to keep the zero-config, directory-browsable store the
  spec (§8.3) uses. Migrating to a SQLite backend is a future option.
- **Open decisions (§11)** taken as reversible defaults: aggregates-only in MLflow metrics with
  full detail in the Excel artifact; report generated automatically on extraction runs plus an
  on-demand `opaque report`; relabeling commits to the current branch.
- The Python backend uses standard **4-space** indentation (Python convention); the Vue
  frontend uses 2-space (JS convention).
```
