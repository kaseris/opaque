"""Locating, loading, and scaffolding onboarding config (spec §3.2).

Config lives in the *target* repo, committed like any other project file. When absent, the
tool guides the maintainer through creating it (§3.1) — ``scaffold_config`` writes a
commented starter template.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import OpaqueConfig

CONFIG_RELPATH = Path('.opaque') / 'config.yaml'


class ConfigNotFound(FileNotFoundError):
    """No ``.opaque/config.yaml`` in the target repo."""


def config_path(repo: str | Path) -> Path:
    return Path(repo) / CONFIG_RELPATH


def load_config(repo: str | Path) -> OpaqueConfig:
    path = config_path(repo)
    if not path.exists():
        raise ConfigNotFound(
            f'No onboarding config at {path}. Run `opaque onboard {repo}` to create one.'
        )
    data = yaml.safe_load(path.read_text()) or {}
    return OpaqueConfig.model_validate(data)


_TEMPLATE = """\
# Opaque onboarding config (see CLAUDE.md §3). Committed and maintained like any other
# project file. Paths are relative to this repo's root.
project: {project}

tools:
  - name: my_tool
    task_type: extraction        # extraction | classification
    prompts:                     # role -> prompt file path (a run selects which roles it uses)
      system: prompts/system.txt
      extraction: prompts/extraction.txt
    eval_data: eval_data         # directory of per-sample JSON, or a single .json file
    field_schema_path: field_schema.yaml   # extraction only; omit for classification
    model_name: gpt-4o
    temperature: 0.0
    invocation:
      # Placeholders: {{input}} {{output_dir}} {{prompt.<role>}} {{model}} {{temperature}}
      command: >-
        python eval_script.py --input {{input}} --out {{output_dir}}
        --system {{prompt.system}} --extraction {{prompt.extraction}}
        --model {{model}} --temperature {{temperature}}
"""


def scaffold_config(repo: str | Path, project: str) -> Path:
    """Write a starter ``.opaque/config.yaml`` into the repo. Refuses to overwrite an
    existing one."""
    path = config_path(repo)
    if path.exists():
        raise FileExistsError(f'Config already exists at {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE.format(project=project))
    return path
