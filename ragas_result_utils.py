"""Utilities for normalizing Ragas evaluation results into scalar metrics."""

import math
from typing import Any, Dict, List, Mapping, Sequence

METRIC_KEYS = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)


def _to_sequence(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(parsed) or math.isinf(parsed):
        return None

    return parsed


def _extract_raw_metric_values(result: Any, metric_key: str) -> List[Any]:
    # dict-like path
    if isinstance(result, Mapping):
        return _to_sequence(result.get(metric_key))

    # Ragas EvaluationResult path (`__getitem__`, no `.get` in v0.4.x)
    try:
        return _to_sequence(result[metric_key])
    except Exception:
        return []


def summarize_ragas_result(
    result: Any,
    metric_keys: Sequence[str] = METRIC_KEYS,
    failed_policy: str = "ignore",
) -> Dict[str, Dict[str, Any]]:
    """
    Normalize a Ragas result to scalar metrics and coverage.

    Args:
        result: Ragas EvaluationResult or dict-like object.
        metric_keys: Metrics to summarize.
        failed_policy: `ignore` (default) or `zero`.

    Returns:
        Dict with keys: `metrics`, `coverage`, `raw_counts`.
    """
    if failed_policy not in {"ignore", "zero"}:
        raise ValueError("failed_policy must be one of: 'ignore', 'zero'")

    metrics: Dict[str, float] = {}
    coverage: Dict[str, Dict[str, float]] = {}
    raw_counts: Dict[str, Dict[str, int]] = {}

    for metric_key in metric_keys:
        raw_values = _extract_raw_metric_values(result, metric_key)
        valid_values: List[float] = []
        valid_count = 0
        invalid_count = 0

        for raw_value in raw_values:
            numeric_value = _to_float(raw_value)
            if numeric_value is None:
                invalid_count += 1
                if failed_policy == "zero":
                    valid_values.append(0.0)
            else:
                valid_count += 1
                valid_values.append(numeric_value)

        mean_value = sum(valid_values) / len(valid_values) if valid_values else 0.0
        total_count = len(raw_values)
        ratio = (valid_count / total_count) if total_count else 0.0

        metrics[metric_key] = float(mean_value)
        coverage[metric_key] = {
            "valid": valid_count,
            "total": total_count,
            "ratio": float(ratio),
        }
        raw_counts[metric_key] = {
            "valid": valid_count,
            "invalid": invalid_count,
            "total": total_count,
        }

    return {
        "metrics": metrics,
        "coverage": coverage,
        "raw_counts": raw_counts,
    }
