import unittest
from datasets import Dataset, DatasetDict
from summarization.cleaning import clean_dataset


class TestCleaningPipeline(unittest.TestCase):
    def setUp(self):
        self.config = {
            "data": {
                "source_column": "document",
                "target_column": "summary",
                "id_column": "id",
            }
        }

        # Construct mock dataset dict representing various edge cases
        self.mock_dataset = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "id": ["t1", "t2", "t3", "t4"],
                        "document": [
                            "Doc A",
                            "Doc B",
                            "   ",
                            "Doc A",
                        ],  # t3 invalid, t4 duplicate kept
                        "summary": ["Sum 1", "Sum 2", "Sum 3", "Sum 4"],
                    }
                ),
                "validation": Dataset.from_dict(
                    {
                        "id": ["v1", "v2", "v3"],
                        "document": [
                            "Doc C",
                            "Doc B",
                            "Doc D",
                        ],  # v2 overlaps with test(t2)
                        "summary": ["Sum 5", "Sum 6", None],  # v3 has invalid summary
                    }
                ),
                "train": Dataset.from_dict(
                    {
                        "id": ["tr1", "tr2", "tr3", "tr4", "tr5", "tr6"],
                        "document": [
                            "Doc E",  # Valid
                            "Doc A",  # Overlaps with test
                            "Doc C",  # Overlaps with validation
                            "Doc E",  # Intra-train duplicate
                            "Doc F",  # Valid, different doc but same summary as tr1
                            None,  # Invalid doc
                        ],
                        "summary": [
                            "Sum 7",
                            "Sum 8",
                            "Sum 9",
                            "Sum 10",
                            "Sum 7",
                            "Sum 11",
                        ],
                    }
                ),
            }
        )

    def test_cleaning_logic(self):
        processed, stats, log = clean_dataset(self.mock_dataset, self.config)

        # Assert Processed Counts
        self.assertEqual(len(processed["test"]), 3)  # t1, t2, t4
        self.assertEqual(len(processed["validation"]), 1)  # v1
        self.assertEqual(len(processed["train"]), 2)  # tr1, tr5

        # Test Split specific asserts
        test_ids = processed["test"]["id"]
        self.assertIn("t4", test_ids, "Test split duplicate was incorrectly removed")
        self.assertNotIn("t3", test_ids, "Invalid text was not removed")

        # Validation Split specific asserts
        val_ids = processed["validation"]["id"]
        self.assertIn("v1", val_ids)
        self.assertNotIn(
            "v2", val_ids, "Validation doc overlapping with Test was not removed"
        )
        self.assertNotIn("v3", val_ids, "Doc with None summary was not removed")

        # Train Split specific asserts
        train_ids = processed["train"]["id"]
        self.assertEqual(
            train_ids, ["tr1", "tr5"]
        )  # tr1 kept, tr5 kept (same summary, diff doc)
        self.assertNotIn(
            "tr2", train_ids, "Train doc overlapping with Test was not removed"
        )
        self.assertNotIn(
            "tr3", train_ids, "Train doc overlapping with Validation was not removed"
        )
        self.assertNotIn("tr4", train_ids, "Intra-train duplicate was not removed")
        self.assertNotIn("tr6", train_ids, "None doc was not removed")

        # Check Exclusion Log reasons mapping correctly
        log_reasons = {item["id"]: item["reason"] for item in log}
        self.assertEqual(log_reasons["t3"], "invalid_text")
        self.assertEqual(log_reasons["v2"], "overlap_with_test")
        self.assertEqual(log_reasons["v3"], "invalid_text")
        self.assertEqual(log_reasons["tr2"], "overlap_with_test")
        self.assertEqual(log_reasons["tr3"], "overlap_with_validation")
        self.assertEqual(log_reasons["tr4"], "duplicate_in_train")
        self.assertEqual(log_reasons["tr6"], "invalid_text")

        # Verify Math
        for split in ["test", "validation", "train"]:
            raw_c = stats[split]["raw_count"]
            proc_c = stats[split]["processed_count"]
            excl_c = sum(
                [v for k, v in stats[split].items() if k.startswith("excluded_")]
            )
            self.assertEqual(raw_c, proc_c + excl_c, f"Math mismatch in {split}")


if __name__ == "__main__":
    unittest.main()
