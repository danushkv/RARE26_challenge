import csv
import hashlib
import unittest
from collections import Counter
from pathlib import Path


EXPECTED_SHA256 = "dc07de48a0cb1c94a65069a75bff7903be262d04619b78f48c1eb03903e4e2c4"


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).parents[1] / "data/splits/5fold_cv.csv"
        with cls.path.open(newline="") as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_checksum(self):
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), EXPECTED_SHA256)

    def test_counts_and_unique_ids(self):
        self.assertEqual(len(self.rows), 3095)
        self.assertEqual(len({row["sample_id"] for row in self.rows}), 3095)
        self.assertEqual(Counter(row["target"] for row in self.rows), {"0": 2937, "1": 158})
        self.assertEqual(Counter(row["split"] for row in self.rows), {
            "fold_0": 619, "fold_1": 619, "fold_2": 619,
            "fold_3": 619, "fold_4": 619,
        })


if __name__ == "__main__":
    unittest.main()

