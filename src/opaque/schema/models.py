"""Per-sample JSON schema — the eval script's output contract (spec §5).

Task-agnostic fields, one JSON object per sample. `gold` is optional/nullable, since
onboarding allows projects with partial or no gold data (§6): metrics run only over the
subset of samples that actually carry gold.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class Sample(BaseModel):
    # Unknown keys emitted by an eval script are preserved rather than dropped.
    model_config = ConfigDict(extra='allow')

    sample_id: str
    raw_file_name: str | None = None
    input: Any = None
    # Ground truth — nullable. `None` means "no gold for this sample" (§5/§6).
    gold: Any = None
    # Model output, same shape as `gold` (a label / list of labels for classification,
    # a nested object for extraction).
    prediction: Any = None
    latency_ms: float | None = None
    token_counts: dict[str, Any] | None = None

    @property
    def has_gold(self) -> bool:
        """Whether ground truth is present for this sample (§6 partial/absent gold)."""
        return self.gold is not None
