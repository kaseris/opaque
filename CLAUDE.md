# LLM Evaluation Tracking Tool — Specification

## 1. Overview

A standalone, project-agnostic prototype tool for tracking the performance of LLM-based
classifiers and structured extraction tools over time, tied to prompt versions. The tool is
not built into any one project — it is pointed at a project's repo and onboarded, and can
manage multiple onboarded projects over time. Built on MLflow for tracking/storage and
MLflow's built-in UI for run browsing, with a custom Excel report generator and a basic
relabeling/data-cleaning UI layered on top.

**In scope for v1:** project onboarding, prompt/eval-set versioning, a pluggable metrics
system for classification and structured extraction, MLflow run logging, a single-run Excel
report, and a basic in-tool relabeling UI. Evaluation runs are combined (prediction +
scoring together) — not decoupled.

**Explicitly deferred (see §11):** cross-run comparison reports, trend reports, automated
regression detection, decoupled prediction/scoring, and a custom web UI beyond MLflow's
built-in one and the relabeling UI.

This is an early exploration of the approach, not an implementation plan — no project
structure, file layout, or code organization is specified yet.

## 2. Architecture pipeline

```
Onboarding (once per project)
  tool reads config committed in the target repo: prompt file roles · eval script
  invocation contract · eval data location — maintained by the repo's maintainer
  like any other project file
                                    |
                                    v
   Prompt file(s)                          Eval data
   (read-only, observed                    (in-repo, git-versioned,
    from the repo's history)                editable via relabeling UI)
        \                                    ^         |
         \                          Relabeling UI       v
          \                        (relabel / clean)   Eval set version
           \                                            (selected for this run)
            \                                          /
             v                                        v
                Evaluation runner (existing eval script, invoked via contract)
                                    |
                                    v
              Per-sample JSON outputs (gold optional per sample)
                                    |
                                    v
        Metrics aggregator (scored only over samples with gold present)
                                    |
                                    v
        MLflow tracking run (params · metrics · artifacts, tagged by project)
                                 /      \
                                v        v
                      MLflow UI      Excel report (.xlsx)
                    (built-in)      (custom, logged as artifact)
```

## 3. Project onboarding & configuration

### 3.1 What onboarding captures, per project
Onboarding config is created and maintained by the repo's maintainer, committed into the project repo (§3.2). The tool reads it when pointed at a repo; if it's absent, the tool guides the maintainer through creating it.

- Repo path on disk
- The tool(s) within that project (a project can contain more than one classifier/extractor)
- Per tool: prompt file(s) and their role names (§4.1), the evaluation script's invocation
  contract (a command template with placeholders for prompt paths / input path / output
  directory — the one fixed requirement is that its output lands in the per-sample JSON
  schema, §5), and the initial eval/gold data

### 3.2 Where configuration and data actually live
Both onboarding config and eval data live **in the target project's repo**, committed and
maintained the same way as any other project file — this is the repo maintainer's
responsibility, not the tool's. This gives onboarding config the same free sharing (clone the
repo, the config comes with it) and history that prompts already get.

- **Prompt files and the field-matching schema** stay **read-only** from the tool's
  perspective, as before — the developer edits and commits them normally; the tool only
  observes whatever commit is currently checked out.
- **Eval/gold data is also in-repo**, but the tool has write access to it via the relabeling
  UI (§9) — every relabeling session becomes a commit in the project's real git history.

**Trade-off worth flagging:** relabeling commits will interleave with normal development
commits in the same repo. One option, if that turns out to be noisy in practice, is to have
the relabeling UI commit to a dedicated branch the maintainer merges on their own schedule —
keeping data in-repo per this decision while keeping mainline history clean. Not adopted by
default; noted as an option in §11.

### 3.3 Multi-project scoping
Since one tool instance can manage several onboarded projects, MLflow experiment naming and
tagging need a project dimension — see §8.1.

## 4. Versioning

Every artifact that affects "what does correct mean" is versioned the same way: git commit
hash (provenance) + content hash (identity, dedup key). Two different commits can produce a
byte-identical prompt or schema — **content hash determines whether two runs used the same
version; git commit is provenance, not identity.**

### 4.1 Prompt versioning

A project may contain many prompt files across many tools (e.g. a system prompt, a
task/extraction prompt, a few-shot examples file). Rather than assuming one fixed prompt file
per tool, **each evaluation run explicitly selects which prompt file(s) it uses**, giving each
one a role name (e.g. `system`, `extraction`, `fewshot`) chosen at run time.

Each selected prompt file is versioned **individually**, so a given file's history can be
traced regardless of what other prompt files it happened to be paired with in any one run:

- `git_commit` — the most recent commit that modified *this specific file*
  (`git log -1 -- <path>`), not repo HEAD. A prompt file's tracked version only changes when
  its own content changes — not on every unrelated commit elsewhere in the repo.
- `content_hash` — sha256 of this file's content (first 12 hex chars)
- `dirty` — true if this specific file has uncommitted changes at eval time

An aggregate `prompt_bundle_hash` (a hash over the sorted `{role: content_hash}` mapping) is
also computed — a single value to check "did the overall prompt configuration change at all,"
while each role's hash stays independently queryable for isolating which specific file
changed between two runs.

**Reproducibility policy:** the runner blocks execution if *any* selected prompt file is
dirty, overridable with `--allow-dirty`. Dirty status is recorded per file, so an override
doesn't obscure which specific file was uncommitted.

### 4.2 Eval set versioning
Same commit + content hash pattern as prompts, applied within the target project's repo
(§3.2), alongside prompts. Both eval inputs and gold labels are versioned as
one unit (typically one file per sample or one combined file) — relabeling touches gold, but
keeping input and gold together avoids splitting a sample's history across two disconnected
locations, and matches how the relabeling UI presents them side by side anyway (§9).

`eval_set_git_commit` / `eval_set_content_hash` are required params on every run — metrics are
only comparable across runs that used the same eval set version, and this makes that explicit
and filterable in MLflow. Because runs are combined (not decoupled, per §1), revisiting an
older eval-set version means re-running the full evaluation against it, not just re-scoring
existing predictions — see §11.

### 4.3 Field-matching schema versioning
The per-field type/tolerance override schema (§7.1) is versioned the same way (git commit +
content hash). Currently read-only from the tool's perspective, same as prompts (§3.2).
Logged as `field_schema_git_commit` / `field_schema_content_hash` params.

## 5. Per-sample JSON schema (eval script output contract)

Task-agnostic fields, produced by the existing eval script (one JSON file per sample):

| Field | Description |
|---|---|
| `sample_id` | unique identifier for the sample |
| `raw_file_name` | original source file name |
| `input` | the input given to the model |
| `gold` | ground truth — **optional/nullable**, since onboarding allows projects with partial or no gold data (see §6) |
| `prediction` | model output (same shape as `gold`) |
| `latency_ms` | request latency |
| `token_counts` | prompt/completion token counts |

Task-specific shape of `gold` / `prediction`:
- **Classification:** a label (or list of labels, for multi-label)
- **Extraction:** a nested JSON object matching the tool's output schema

## 6. Metrics system

Pluggable interface, one implementation per task type, selected via `--task-type`:

```
MetricComputer (abstract)
  .compute(samples) -> dict[str, float]      # aggregate metrics, logged to MLflow
  .artifacts(samples) -> dict[str, Any]       # supporting artifacts (confusion matrix, etc.)

Registry:
  "classification" -> ClassificationMetrics
  "extraction"      -> ExtractionMetrics
```

New task types are added by implementing the interface and registering them — no changes
required elsewhere in the pipeline.

**Partial/absent gold:** since `gold` is optional per sample (§5), every metric computation
runs only over the subset of samples that actually have gold present. The labeled/unlabeled
sample count is always reported alongside the metric itself (see §10 Summary sheet), so an
accuracy figure is never silently computed over a partial sample without saying so. A project
with no gold data at all can still run evaluations to collect predictions, just without
accuracy metrics.

### 6.1 Classification metrics
Accuracy, macro/micro precision-recall-F1 as headline (MLflow) metrics. Per-class F1 is
**open** on whether it belongs as individual MLflow metrics (searchable/comparable across
runs) or as an artifact table (viewed per-run but not directly comparable) — see §11.

### 6.2 Extraction metrics
Overall field accuracy (`correct fields / total fields`, computed from the union of
predicted + gold paths per §7) and per-sample accuracy. Full field-level detail is not
exploded into individual MLflow metrics — it lives in the Excel report artifact (§10), with
only the aggregates logged to MLflow for cross-run comparability.

## 7. Field-level value matching (extraction)

### 7.1 Field type determination — hybrid
- **Default:** inferred from the ground-truth value's shape (numeric pattern → numeric,
  recognized date pattern → date, `true/false/yes/no` → boolean, else → string)
- **Override:** a per-tool schema definition, keyed by `json_path` *pattern* with array
  indices wildcarded (e.g. `extractionData.*.orderNumber`), so one entry applies across every
  array position and sample. Used for fields that would be inferred wrong (e.g. a
  numeric-looking ID like `"00123"` that needs exact string matching, not numeric coercion
  that would drop the leading zero). Also carries numeric tolerance and date-format hints per
  field, since it's already the natural place for per-field config.

### 7.2 Comparators
| Type | Comparison |
|---|---|
| Numeric | parse both values (strip currency symbols/separators), compare within tolerance (relative tolerance + absolute floor to avoid divide-by-zero); tolerance overridable per field in the schema |
| Date | parse both into a canonical date regardless of source format, compare as dates |
| Boolean | normalize truthy/falsy strings before comparing |
| String (fallback) | case-fold + whitespace-trim, then exact match |

The comparator actually used for each field is recorded (`comparator_used`) so the Excel
report makes matches transparent rather than looking like unexplained leniency.

### 7.3 Array alignment — positional
Predicted and gold array items are matched by index (item 0 vs item 0, etc.).

**Known limitation, accepted for v1:** if the model drops or inserts one array item, every
subsequent item shifts by one index and is marked incorrect even if extracted correctly — a
single miss cascades into N-1 false mismatches. Visible in the report as a run of consecutive
`incorrect` rows, and should be read as "check for an array shift," not taken at face value.
Future upgrade path: key-based alignment — deferred, see §11.

## 8. MLflow tracking schema

### 8.1 Organization
One MLflow **experiment** per (project, tool) pair — named `{project}/{tool}` to avoid
collisions between two different onboarded projects that happen to name a tool the same
thing. `project_id` is logged as a tag on every run regardless, so runs remain filterable
across the whole tracking store even if experiment naming changes later. Flat runs (not
nested) for v1 — searchable via MLflow's query syntax (`params.task_type = "extraction" and
params.eval_set_content_hash = "..."`).

### 8.2 Run schema

| Category | Fields |
|---|---|
| Run name | `{prompt_bundle_hash}-{model}-{timestamp}` |
| Params | `prompt_bundle_hash`, `prompt.<role>.git_commit` + `prompt.<role>.content_hash` (one pair per selected prompt file), `field_schema_git_commit`, `field_schema_content_hash`, `eval_set_git_commit`, `eval_set_content_hash`, `model_name`, `temperature`, `task_type` |
| Tags | `project_id`, `prompt_dirty` (aggregate: true if any selected prompt file is dirty), `git_branch`, `tool_name` |
| Metrics | task-specific aggregates from §6, plus `labeled_sample_count` / `unlabeled_sample_count` |
| Artifacts | `raw_outputs/*.json` (per-sample files), `prompts/<role>.txt` + `prompts/manifest.json`, `field_schema.yaml`, confusion matrix or error samples (classification), `report.xlsx` (extraction) |

### 8.3 MLflow UI
Used as-is, no custom wrapper:
```
mlflow ui --backend-store-uri ./mlruns --port 5000
```
Served at `localhost:5000`. Filterable by `project_id`, by a single prompt role's hash in
isolation, and by eval-set version.

## 9. Relabeling / data-cleaning UI

### 9.1 Scope & workflow
A basic in-tool UI over the project's eval/gold data (§3.2), for filling in missing gold
labels and correcting existing ones. Alongside each sample, if a prior evaluation run exists
for the current prompt version, the model's prediction is shown next to the gold field being
edited — turning relabeling into a review pass ("does the model's answer look right, if not
what's correct") rather than blind data entry.

### 9.2 Commit granularity
Edits accumulate in an uncommitted working state as the user reviews samples. Nothing is
committed per keystroke or per field. The user explicitly saves/closes a "relabeling session,"
which creates **one git commit** covering everything changed in that session, with an
auto-generated message (sample count, date) the user can annotate, committed into the
project's repo (§3.2). This keeps history readable (one entry per review session) while still
giving full go-back-and-forth capability via git log, and gets author + timestamp for free
from git. See §3.2 for the option of committing to a dedicated branch instead of mainline.

### 9.3 Concurrency
Single-user, local tool for v1. Concurrent multi-user editing of the same eval set is out of
scope and not handled — see §11.

## 10. Excel report specification (extraction)

Generated per run, logged back to MLflow as the `report.xlsx` artifact. Consumes the same
per-field match results produced by the metrics aggregator (§6.2) — the report is a
presentation layer, not a second implementation of the matching logic.

### Sheet 1 — Summary
Run metadata (project, tool, prompt versions per role, eval set version, model, timestamp)
plus headline numbers: overall accuracy, total samples, labeled vs. unlabeled sample counts,
total fields, and counts by status (`correct` / `incorrect` / `missing` / `extra`).

### Sheet 2 — Field-level Detail
One row per (sample, field) pair, built from the **union** of paths present in prediction and
ground truth (so missed fields and hallucinated fields both appear, not just predicted ones).
Samples with no gold available are still listed (predictions shown), with `status` reflecting
that no ground truth was available for comparison rather than being silently omitted:

| raw_file_name | json_path | predicted_value | ground_truth_value | status | comparator_used |
|---|---|---|---|---|---|

`json_path` uses dot/index notation, e.g. `extractionData.0.fieldName.0.subFieldName`.

### Sheet 3 — Per-sample Rollup
One row per document, for spotting problem documents at a glance without reading every field:

| raw_file_name | fields_correct | fields_total | accuracy_pct |
|---|---|---|---|

## 11. Open decisions / future work

- **Project structure / code organization**: intentionally not specified yet — this document
  covers behavior and data contracts only, not implementation layout.
- **Relabeling commit strategy**: relabeling sessions currently commit straight to whatever
  branch is checked out (§9.2), which will interleave with normal development commits.
  Committing to a dedicated branch instead is a possible mitigation, not yet decided.
- **Prompt role naming**: whether roles must come from a predefined, validated set per tool,
  or are fully free-form per run — undecided.
- **Field-matching schema storage**: currently read-only/observed like prompts; whether it
  should instead be tool-owned and editable (similar to eval data) is undecided.
- **Per-class / per-field metric granularity in MLflow**: currently leaning toward
  aggregates-only in MLflow metrics, full detail only in the Excel artifact — not finalized.
- **Report generation trigger**: on-demand CLI command vs. automatic generation on every eval
  run — not yet confirmed.
- **Decoupled prediction/scoring**: declined for v1 (runs stay combined). Revisiting an older
  eval-set version currently requires re-running the full evaluation. Worth reconsidering if
  frequent relabeling makes this costly.
- **Comparison report**, **trend report**, and **regression report** — all deferred past v1.
  Baseline definition needs to be fixed before building these.
- **Key-based array alignment** as an upgrade path from positional alignment (§7.3).
- Confirm extraction task type is structured-document field extraction (nested JSON schema)
  rather than span-based NER.
