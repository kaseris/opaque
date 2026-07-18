"""Onboarding configuration models (spec §3).

The config is authored and committed by the target repo's maintainer at
``.opaque/config.yaml`` and read (never written) by the tool when pointed at the repo.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TaskType = Literal['classification', 'extraction']


class InvocationContract(BaseModel):
    """How to invoke the target repo's existing eval script (§3.1).

    ``command`` is a template rendered from the repo root. Supported placeholders:
      - ``{input}``        — path to the eval input/data selected for the run
      - ``{output_dir}``   — directory the script must write per-sample JSON into (§5)
      - ``{prompt.<role>}``— absolute path to the prompt file bound to ``<role>`` this run
      - ``{model}`` / ``{temperature}`` — the model + temperature for the run

    The one fixed requirement is that the script's output lands in the per-sample JSON
    schema of §5.
    """

    command: str


class ToolConfig(BaseModel):
    """A single classifier/extractor within a project (§3.1). A project may contain more
    than one tool."""

    name: str
    task_type: TaskType
    # role -> repo-relative prompt file path. These are the prompt files available to the
    # tool; a run selects which roles it uses (§4.1).
    prompts: dict[str, str] = Field(default_factory=dict)
    # repo-relative path to eval/gold data (a directory of per-sample files or one file).
    eval_data: str
    invocation: InvocationContract
    # repo-relative path to the field-matching override schema (extraction only, §7.1).
    field_schema_path: str | None = None
    model_name: str = 'unknown'
    temperature: float = 0.0


class OpaqueConfig(BaseModel):
    """Top-level onboarding config for one project (§3)."""

    # Project identifier — used for MLflow experiment naming + the ``project_id`` tag (§8.1).
    project: str
    tools: list[ToolConfig]

    def tool(self, name: str) -> ToolConfig:
        for t in self.tools:
            if t.name == name:
                return t
        available = ', '.join(t.name for t in self.tools) or '(none)'
        raise KeyError(f"No tool named '{name}' in project '{self.project}'. Available: {available}")
