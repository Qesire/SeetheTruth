"""Known-number tensor sampling oracle; no model training data."""
import math
import unittest

try:
    import torch
    from telemetry_capture import _sample, _tensor_info
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch optional")
class TensorSamplingOracle(unittest.TestCase):
    def test_small_signed_matrix_exact_statistics(self):
        x = torch.tensor([[0., 1.], [-2., 3.]])
        s = _sample(x, cap=32)
        self.assertEqual(s["population"], 4)
        self.assertEqual(s["sampled"], 4)
        self.assertEqual(s["finite_count"], 4)
        self.assertEqual(s["nonfinite_count"], 0)
        self.assertEqual(s["sample_zero_fraction"], .25)
        self.assertEqual(s["matrix_original_shape"], [2, 2])
        self.assertEqual(s["matrix"], [[0., 1.], [-2., 3.]])
        self.assertEqual(s["statistics"]["min"], -2)
        self.assertEqual(s["statistics"]["max"], 3)
        self.assertAlmostEqual(s["statistics"]["mean"], .5)
        self.assertAlmostEqual(s["statistics"]["std"], math.sqrt(3.25))
        self.assertEqual(sum(s["histogram"]["bins"]), 4)

    def test_tensor_metadata_noncontiguous_view(self):
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        t = base.transpose(0, 1)
        m = _tensor_info(t)
        self.assertEqual(m["shape"], [4, 3])
        self.assertEqual(m["stride"], [1, 4])
        self.assertFalse(m["contiguous"])
        self.assertEqual(m["logical_bytes"], 48)
        s = _sample(t, cap=32)
        self.assertEqual(s["matrix_original_shape"], [4, 3])
        self.assertEqual(s["matrix"][0], [0, 4, 8])

    def test_bounded_sampling_does_not_claim_population_statistics(self):
        large = torch.arange(50000, dtype=torch.float32).reshape(100, 500)
        sample = _sample(large, cap=50)
        self.assertEqual(sample["population"], 50000)
        self.assertLessEqual(sample["sampled"], 50)
        self.assertLessEqual(len(sample["matrix"]), 20)
        self.assertLessEqual(len(sample["matrix"][0]), 20)
        self.assertEqual(sample["sampling"], "strided_linear_sample_and_first_leading_indices_matrix")


if __name__ == "__main__":
    unittest.main()
