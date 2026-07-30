"""Evaluation runner (spec §2, §3.1).

Runs are *combined* — prediction + scoring together (§1). The runner renders the target
repo's eval-script command from its invocation contract, enforces the reproducibility policy
(§4.1), executes the script, loads the per-sample JSON it produced (§5), and scores it with
the selected metrics (§6). It does not touch MLflow — the tracking layer consumes a
``RunResult``.
"""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.field_schema import FieldSchema, load_field_schema
from ..config.loader import load_config
from ..config.models import ToolConfig
from ..matching.results import FieldResult
from ..metrics import build as build_metrics
from ..schema.io import load_samples
from ..schema.models import Sample
from ..versioning import git
from ..versioning.versions import FileVersion, PromptBundle, path_version, prompt_bundle


class RunnerError(RuntimeError):
    pass


class DirtyPromptError(RunnerError):
    """A selected prompt file has uncommitted changes and --allow-dirty was not set (§4.1)."""


class EvalScriptError(RunnerError):
    """The target repo's eval script exited non-zero."""


@dataclass
class RunResult:
    repo: Path
    project: str
    tool: ToolConfig
    task_type: str
    model: str
    temperature: float
    prompt_paths: dict[str, str]           # role -> repo-relative path
    prompt_bundle: PromptBundle
    eval_set_version: FileVersion
    field_schema_version: FileVersion | None
    samples: list[Sample]
    metrics: dict[str, float]
    artifacts: dict[str, Any]
    output_dir: Path
    git_branch: str
    timestamp: str
    field_schema: FieldSchema | None = None
    field_results: list[FieldResult] = field(default_factory=list)


def run(
    repo: str | Path,
    tool_name: str,
    *,
    prompts: dict[str, str] | None = None,
    task_type: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    eval_data: str | None = None,
    allow_dirty: bool = False,
    output_dir: str | Path | None = None,
) -> RunResult:
    repo = Path(repo).resolve()
    if not git.is_repo(repo):
        raise RunnerError(f'{repo} is not a git repository (eval data + prompts are git-versioned, §3.2).')

    config = load_config(repo)
    tool = config.tool(tool_name)
    task_type = task_type or tool.task_type
    model = model or tool.model_name
    temperature = tool.temperature if temperature is None else temperature
    eval_rel = eval_data or tool.eval_data

    # Prompt selection (§4.1): explicit role->path, else the tool's declared prompts.
    prompt_paths = dict(prompts) if prompts else dict(tool.prompts)
    if not prompt_paths:
        raise RunnerError(f"Tool '{tool_name}' has no prompt files to select.")
    for role, rel in prompt_paths.items():
        if not (repo / rel).exists():
            raise RunnerError(f"Prompt file for role '{role}' not found: {rel}")

    bundle = prompt_bundle(repo, prompt_paths)
    if bundle.any_dirty and not allow_dirty:
        raise DirtyPromptError(
            'Uncommitted changes in selected prompt file(s): '
            + ', '.join(bundle.dirty_roles)
            + '. Commit them or pass --allow-dirty.'
        )

    eval_set_version = path_version(repo, eval_rel)
    field_schema_version = (
        path_version(repo, tool.field_schema_path) if tool.field_schema_path else None
    )
    field_schema = (
        load_field_schema(repo / tool.field_schema_path) if tool.field_schema_path else None
    )

    out = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix='opaque-run-'))
    out.mkdir(parents=True, exist_ok=True)

    command = _render_command(
        tool.invocation.command,
        input_path=repo / eval_rel,
        output_dir=out,
        prompt_paths={r: repo / p for r, p in prompt_paths.items()},
        model=model,
        temperature=temperature,
    )
    _exec(command, cwd=repo)

    samples = load_samples(out)
    computer = build_metrics(task_type, field_schema=field_schema)
    metrics = computer.compute(samples)
    artifacts = computer.artifacts(samples)
    field_results = artifacts.get('field_results', []) if task_type == 'extraction' else []

    return RunResult(
        repo=repo,
        project=config.project,
        tool=tool,
        task_type=task_type,
        model=model,
        temperature=temperature,
        prompt_paths=prompt_paths,
        prompt_bundle=bundle,
        eval_set_version=eval_set_version,
        field_schema_version=field_schema_version,
        samples=samples,
        metrics=metrics,
        artifacts=artifacts,
        output_dir=out,
        git_branch=git.current_branch(repo),
        timestamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'),
        field_schema=field_schema,
        field_results=field_results,
    )


def _render_command(template, input_path, output_dir, prompt_paths, model, temperature) -> str:
    cmd = template
    replacements = {
        '{input}': str(input_path),
        '{output_dir}': str(output_dir),
        '{model}': str(model),
        '{temperature}': str(temperature),
    }
    for role, path in prompt_paths.items():
        replacements['{prompt.%s}' % role] = str(path)
    for token, value in replacements.items():
        cmd = cmd.replace(token, value)
    return cmd


def _exec(command: str, cwd: Path) -> None:
    argv = shlex.split(command)
    try:
        result = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True)
    except FileNotFoundError as exc:
        # Worth naming explicitly: the eval script's interpreter is resolved against whatever
        # PATH the caller had. An interactive shell with a virtualenv active and a git hook's
        # minimal environment resolve it differently, so a command that works by hand can fail
        # from the pre-push hook.
        raise EvalScriptError(
            f"Cannot run '{argv[0] if argv else command}' — not found on PATH.\n"
            f'  command: {command}\n'
            "  Fix the tool's invocation.command in .opaque/config.yaml: name an interpreter "
            'that exists outside your shell (e.g. python3), or an absolute path to the '
            "project's own virtualenv, since git hooks do not inherit an activated venv."
        ) from exc
    if result.returncode != 0:
        raise EvalScriptError(
            f'Eval script failed (exit {result.returncode}).\n'
            f'  command: {command}\n'
            f'  stderr: {result.stderr.strip()}'
        )
