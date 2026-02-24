import unittest
import numpy as np
from optim_EMS import (
    generate_synthetic_microgrid,
    TimeSeriesDataset,
    BatteryOptimizer,
    run_pipeline
)


class TestEMS(unittest.TestCase):

    def test_data_generation_length(self):
        load, pv, price = generate_synthetic_microgrid(days=2)
        self.assertEqual(len(load), 48)
        self.assertEqual(len(pv), 48)
        self.assertEqual(len(price), 48)

    def test_dataset_sequences_shape(self):
        load, _, _ = generate_synthetic_microgrid(days=2)
        ds = TimeSeriesDataset(load, window_size=24)
        X, y = ds.create_sequences()

        self.assertEqual(X.shape[1], 24)
        self.assertEqual(X.shape[2], 1)
        self.assertEqual(len(X), len(y))

    def test_battery_optimizer_output(self):
        load, pv, price = generate_synthetic_microgrid(days=1)

        optimizer = BatteryOptimizer(load, pv, price)
        results = optimizer.solve()

        self.assertEqual(len(results["SOC"]), 24)
        self.assertTrue(all(s is not None for s in results["SOC"]))

    def test_pipeline_runs(self):
        load, pv, price = generate_synthetic_microgrid(days=2)
        results = run_pipeline(load, pv, price)

        self.assertIn("SOC", results)
        self.assertEqual(len(results["SOC"]), 24)


if __name__ == "__main__":
    unittest.main()