"""Union-of-paths field matching for extraction (spec §7).

Predicted and gold objects are flattened to ``{json_path: value}`` and compared over the
*union* of paths, so missed fields and hallucinated fields both surface (§10). Array items
are aligned positionally (§7.3) — a dropped/inserted item shifts every later index and shows
up as a run of consecutive ``incorrect`` rows (accepted v1 limitation).
"""

from __future__ import annotations

from ..config.field_schema import FieldSchema
from ..schema.models import Sample
from . import comparators, paths
from .inference import infer_type
from .results import FieldResult, Status


def match_sample(sample: Sample, schema: FieldSchema | None = None) -> list[FieldResult]:
    """Match one sample's prediction against its gold, yielding one result per union path.
    A sample with no gold (§5) yields its predicted fields with ``NO_GOLD`` status so it is
    still listed in the report rather than silently omitted."""
    if not sample.has_gold:
        return [
            FieldResult(
                sample_id=sample.sample_id,
                raw_file_name=sample.raw_file_name,
                json_path=path,
                predicted_value=value,
                ground_truth_value=None,
                status=Status.NO_GOLD,
            )
            for path, value in sorted(paths.flatten(sample.prediction).items(), key=_sort_key)
        ]

    gold_flat = paths.flatten(sample.gold)
    pred_flat = paths.flatten(sample.prediction)
    all_paths = sorted(set(gold_flat) | set(pred_flat), key=paths.natural_key)

    results: list[FieldResult] = []
    for path in all_paths:
        in_gold, in_pred = path in gold_flat, path in pred_flat
        gold_value = gold_flat.get(path)
        pred_value = pred_flat.get(path)

        if in_gold and in_pred:
            resolved = _resolve_type(path, gold_value, schema)
            rule = schema.match(path) if schema else None
            matched, comparator_used = comparators.compare(resolved, pred_value, gold_value, rule)
            status = Status.CORRECT if matched else Status.INCORRECT
        elif in_gold:
            comparator_used, status = None, Status.MISSING
        else:
            comparator_used, status = None, Status.EXTRA

        results.append(
            FieldResult(
                sample_id=sample.sample_id,
                raw_file_name=sample.raw_file_name,
                json_path=path,
                predicted_value=pred_value if in_pred else None,
                ground_truth_value=gold_value if in_gold else None,
                status=status,
                comparator_used=comparator_used,
            )
        )
    return results


def match_samples(samples: list[Sample], schema: FieldSchema | None = None) -> list[FieldResult]:
    results: list[FieldResult] = []
    for sample in samples:
        results.extend(match_sample(sample, schema))
    return results


def _resolve_type(json_path, gold_value, schema):
    """Override schema wins over shape inference (§7.1)."""
    if schema is not None:
        rule = schema.match(json_path)
        if rule is not None and rule.type is not None:
            return rule.type
    return infer_type(gold_value)


def _sort_key(item):
    return paths.natural_key(item[0])
