import unittest

from ragas_result_utils import METRIC_KEYS, summarize_ragas_result


class FakeEvaluationResult:
    """Mimics Ragas EvaluationResult with __getitem__ only."""

    def __init__(self, payload):
        self.payload = payload

    def __getitem__(self, key):
        return self.payload[key]


class TestRagasResultUtils(unittest.TestCase):
    def test_scalar_dict_input_returns_scalar_metrics(self):
        result = {
            "context_precision": 0.8,
            "context_recall": 0.6,
            "faithfulness": 0.9,
            "answer_relevancy": 0.7,
        }

        summary = summarize_ragas_result(result)

        self.assertEqual(summary["metrics"], result)
        for metric in METRIC_KEYS:
            self.assertEqual(summary["coverage"][metric]["valid"], 1)
            self.assertEqual(summary["coverage"][metric]["total"], 1)
            self.assertEqual(summary["coverage"][metric]["ratio"], 1.0)

    def test_list_input_ignores_none_and_nan(self):
        result = {
            "context_precision": [0.2, None, float("nan"), 0.8],
            "context_recall": [0.5, 0.7],
            "faithfulness": [0.4],
            "answer_relevancy": [None, "bad", 1.0],
        }

        summary = summarize_ragas_result(result)

        self.assertAlmostEqual(summary["metrics"]["context_precision"], 0.5)
        self.assertEqual(summary["coverage"]["context_precision"]["valid"], 2)
        self.assertEqual(summary["coverage"]["context_precision"]["total"], 4)

        self.assertAlmostEqual(summary["metrics"]["answer_relevancy"], 1.0)
        self.assertEqual(summary["coverage"]["answer_relevancy"]["valid"], 1)
        self.assertEqual(summary["coverage"]["answer_relevancy"]["total"], 3)

    def test_all_invalid_values_return_zero_score_and_zero_coverage(self):
        result = {
            "context_precision": [None, float("nan"), "x"],
            "context_recall": [],
            "faithfulness": None,
            # answer_relevancy intentionally missing
        }

        summary = summarize_ragas_result(result)

        for metric in METRIC_KEYS:
            self.assertEqual(summary["metrics"][metric], 0.0)
            self.assertEqual(summary["coverage"][metric]["ratio"], 0.0)

        self.assertEqual(summary["raw_counts"]["context_precision"]["invalid"], 3)

    def test_works_with_evaluation_result_like_object(self):
        fake = FakeEvaluationResult(
            {
                "context_precision": [0.1, 0.2],
                "context_recall": [0.3, 0.4],
                "faithfulness": [0.5, 0.6],
                "answer_relevancy": [0.7, 0.8],
            }
        )

        summary = summarize_ragas_result(fake)

        self.assertAlmostEqual(summary["metrics"]["context_precision"], 0.15)
        self.assertAlmostEqual(summary["metrics"]["answer_relevancy"], 0.75)

    def test_formatted_output_never_crashes_for_metrics(self):
        result = {
            "context_precision": [0.123456, None],
            "context_recall": [0.5],
            "faithfulness": [float("nan")],
            "answer_relevancy": [1.0, 0.0],
        }

        summary = summarize_ragas_result(result)

        for metric in METRIC_KEYS:
            formatted = f"{summary['metrics'][metric]:.4f}"
            self.assertIsInstance(formatted, str)


if __name__ == "__main__":
    unittest.main()
