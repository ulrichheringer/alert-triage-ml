import math
import unittest

from pipeline import evaluate, predict, sample_input, train_model


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    return math.isfinite(float(value))


class PipelineTests(unittest.TestCase):
    def test_evaluation_is_reported(self):
        metrics = evaluate()
        self.assertTrue(metrics)
        self.assertTrue(finite(metrics))

    def test_model_trains_and_predicts(self):
        model = train_model()
        prediction = predict(model, sample_input())
        self.assertIsNotNone(prediction)


if __name__ == "__main__":
    unittest.main()
