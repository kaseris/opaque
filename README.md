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
        python eval_script.py --task extraction --input {input} --out {output_dir}
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
