"""Classification metrics (spec §6.1).

Accuracy + macro/micro precision-recall-F1 as headline metrics. Single- and multi-label are
handled uniformly by treating each label value as a set (a single label is a singleton set;
accuracy is exact set match). Per-class detail lives in artifacts, not headline metrics
(the §11 open decision, resolved toward aggregates-only in MLflow).
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from ..schema.models import Sample
from .base import MetricComputer, split_labeled
from .registry import register


@register('classification')
class ClassificationMetrics(MetricComputer):
    def compute(self, samples: list[Sample]) -> dict[str, float]:
        labeled, unlabeled = split_labeled(samples)
        metrics: dict[str, float] = {
            'labeled_sample_count': float(len(labeled)),
            'unlabeled_sample_count': float(unlabeled),
        }
        if not labeled:
            return metrics

        golds = [_to_set(s.gold) for s in labeled]
        preds = [_to_set(s.prediction) for s in labeled]
        n = len(labeled)

        metrics['accuracy'] = sum(g == p for g, p in zip(golds, preds)) / n

        classes = sorted({c for group in (golds, preds) for s in group for c in s}, key=str)
        precisions, recalls, f1s = [], [], []
        tp_tot = fp_tot = fn_tot = 0
        for c in classes:
            tp = sum(c in g and c in p for g, p in zip(golds, preds))
            fp = sum(c in p and c not in g for g, p in zip(golds, preds))
            fn = sum(c in g and c not in p for g, p in zip(golds, preds))
            precisions.append(_safe_div(tp, tp + fp))
            recalls.append(_safe_div(tp, tp + fn))
            f1s.append(_f1(precisions[-1], recalls[-1]))
            tp_tot, fp_tot, fn_tot = tp_tot + tp, fp_tot + fp, fn_tot + fn

        metrics['macro_precision'] = mean(precisions) if precisions else 0.0
        metrics['macro_recall'] = mean(recalls) if recalls else 0.0
        metrics['macro_f1'] = mean(f1s) if f1s else 0.0
        micro_p = _safe_div(tp_tot, tp_tot + fp_tot)
        micro_r = _safe_div(tp_tot, tp_tot + fn_tot)
        metrics['micro_precision'] = micro_p
        metrics['micro_recall'] = micro_r
        metrics['micro_f1'] = _f1(micro_p, micro_r)
        return metrics

    def artifacts(self, samples: list[Sample]) -> dict[str, Any]:
        labeled, _ = split_labeled(samples)
        golds = [_to_set(s.gold) for s in labeled]
        preds = [_to_set(s.prediction) for s in labeled]
        classes = sorted({c for group in (golds, preds) for s in group for c in s}, key=str)

        per_class = []
        for c in classes:
            tp = sum(c in g and c in p for g, p in zip(golds, preds))
            fp = sum(c in p and c not in g for g, p in zip(golds, preds))
            fn = sum(c in g and c not in p for g, p in zip(golds, preds))
            prec, rec = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
            per_class.append({
                'label': c,
                'precision': prec,
                'recall': rec,
                'f1': _f1(prec, rec),
                'support': sum(c in g for g in golds),
            })

        errors = [
            {'sample_id': s.sample_id, 'gold': s.gold, 'prediction': s.prediction}
            for s, g, p in zip(labeled, golds, preds) if g != p
        ]
        return {
            'per_class': per_class,
            'confusion_matrix': _confusion_matrix(golds, preds, classes),
            'errors': errors,
        }


def _to_set(value: Any) -> set:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return set(value)
    return {value}


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def _confusion_matrix(golds, preds, classes) -> dict | None:
    """Single-label confusion matrix (gold rows × predicted columns). ``None`` for
    multi-label, where a flat matrix does not apply."""
    if any(len(s) != 1 for s in golds + preds):
        return None
    index = {c: i for i, c in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in classes]
    for g, p in zip(golds, preds):
        matrix[index[next(iter(g))]][index[next(iter(p))]] += 1
    return {'labels': [str(c) for c in classes], 'matrix': matrix}
