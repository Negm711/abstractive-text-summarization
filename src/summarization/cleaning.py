import json
import csv
import logging
from datetime import datetime
from pathlib import Path

from datasets import DatasetDict
from summarization.config import load_config
from summarization.data import load_raw_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def is_invalid_text(text) -> bool:
    """Check if text is null, non-string, or empty/whitespace."""
    if text is None:
        return True
    if not isinstance(text, str):
        return True
    if not text.strip():
        return True
    return False


def clean_dataset(
    raw_dataset: DatasetDict, config: dict
) -> tuple[DatasetDict, dict, list]:
    """Clean the dataset based on exact priority rules."""
    logger.info("Starting dataset cleaning...")

    source_col = config["data"]["source_column"]
    target_col = config["data"]["target_column"]
    id_col = config["data"]["id_column"]

    exclusion_log = []
    stats = {}

    valid_test_docs = set()
    valid_validation_docs = set()
    seen_train_docs = set()

    processed_dict = {}

    splits_order = ["test", "validation", "train"]
    for split in splits_order:
        logger.info(f"Processing split: {split}")
        split_data = raw_dataset[split]

        docs = split_data[source_col]
        sums = split_data[target_col]
        ids = split_data[id_col]

        indices_to_keep = []

        stats[split] = {
            "raw_count": len(split_data),
            "excluded_invalid": 0,
            "excluded_overlap_test": 0,
            "excluded_overlap_validation": 0,
            "excluded_train_duplicate": 0,
            "processed_count": 0,
        }

        for idx, (doc, summary, doc_id) in enumerate(zip(docs, sums, ids)):
            # Rule 1: Invalid text (Applies to all)
            if is_invalid_text(doc) or is_invalid_text(summary):
                exclusion_log.append(
                    {"split": split, "id": doc_id, "reason": "invalid_text"}
                )
                stats[split]["excluded_invalid"] += 1
                continue

            if split == "test":
                valid_test_docs.add(doc)
                indices_to_keep.append(idx)

            elif split == "validation":
                # Rule 2: Exclude if in valid test
                if doc in valid_test_docs:
                    exclusion_log.append(
                        {"split": split, "id": doc_id, "reason": "overlap_with_test"}
                    )
                    stats[split]["excluded_overlap_test"] += 1
                else:
                    valid_validation_docs.add(doc)
                    indices_to_keep.append(idx)

            elif split == "train":
                # Rule 3 & 4: Exclude if in test, validation, or already seen in train
                if doc in valid_test_docs:
                    exclusion_log.append(
                        {"split": split, "id": doc_id, "reason": "overlap_with_test"}
                    )
                    stats[split]["excluded_overlap_test"] += 1
                elif doc in valid_validation_docs:
                    exclusion_log.append(
                        {
                            "split": split,
                            "id": doc_id,
                            "reason": "overlap_with_validation",
                        }
                    )
                    stats[split]["excluded_overlap_validation"] += 1
                elif doc in seen_train_docs:
                    exclusion_log.append(
                        {"split": split, "id": doc_id, "reason": "duplicate_in_train"}
                    )
                    stats[split]["excluded_train_duplicate"] += 1
                else:
                    seen_train_docs.add(doc)
                    indices_to_keep.append(idx)

        # Select only valid indices to create the processed split
        processed_dict[split] = split_data.select(indices_to_keep)
        stats[split]["processed_count"] = len(processed_dict[split])

    return DatasetDict(processed_dict), stats, exclusion_log


def run_sanity_checks(
    raw_dataset: DatasetDict, processed_dataset: DatasetDict, stats: dict, config: dict
):
    """Verify exact match conditions and data integrity."""
    logger.info("Running post-cleaning sanity checks...")
    source_col = config["data"]["source_column"]

    for split in ["train", "validation", "test"]:
        raw_count = len(raw_dataset[split])
        processed_count = len(processed_dataset[split])
        excluded_count = (
            stats[split]["excluded_invalid"]
            + stats[split]["excluded_overlap_test"]
            + stats[split]["excluded_overlap_validation"]
            + stats[split]["excluded_train_duplicate"]
        )

        assert (
            raw_count == processed_count + excluded_count
        ), f"Count mismatch in {split}"

        # Check for empty docs in processed
        for doc in processed_dataset[split][source_col]:
            assert not is_invalid_text(doc), f"Invalid text found in processed {split}"

    # Cross-split and duplicate checks
    proc_test = set(processed_dataset["test"][source_col])
    proc_val = set(processed_dataset["validation"][source_col])

    assert len(proc_test.intersection(proc_val)) == 0, "Test/Validation overlap exists"

    proc_train = processed_dataset["train"][source_col]
    assert (
        len(proc_test.intersection(set(proc_train))) == 0
    ), "Train/Test overlap exists"
    assert (
        len(proc_val.intersection(set(proc_train))) == 0
    ), "Train/Validation overlap exists"
    assert len(proc_train) == len(set(proc_train)), "Intra-train duplicates exist"

    logger.info("Sanity checks passed successfully.")


def save_reports(stats: dict, exclusion_log: list, config: dict):
    reports_dir = config["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_path = reports_dir / "cleaning_summary.json"
    summary_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_name": config["data"]["dataset_name"],
            "resolved_commit_sha": config.get("resolved_commit_sha", "unknown"),
        },
        "policies": [
            "1. Invalid text (null, empty, whitespace) removed from all splits.",
            "2. Validation docs overlapping with valid Test docs are removed.",
            "3. Train docs overlapping with valid Test or Validation docs are removed.",
            "4. Train duplicates are removed (keeping first occurrence).",
            "5. Test split duplicates are INTENTIONALLY KEPT to preserve original test distribution.",
        ],
        "statistics": stats,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)

    csv_path = reports_dir / "removed_examples.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "id", "reason"])
        writer.writeheader()
        writer.writerows(exclusion_log)

    md_path = reports_dir / "data_quality.md"
    md_content = f"""# Data Quality & Cleaning Report
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset SHA:** {config.get("resolved_commit_sha", "unknown")}

## Cleaning Policies
- Invalid text (empty, null, whitespace) is removed from all datasets.
- **Test Set:** Retains exact duplicates. Excludes invalid text only. Kept to mirror official benchmark.
- **Validation Set:** Excludes docs present in the Test set.
- **Train Set:** Excludes docs present in Test or Validation sets. Intra-train duplicates are removed (first occurrence kept).
- **Target summaries:** Exact duplicate summaries with different source docs are KEPT.

## Results
"""
    for split in ["train", "validation", "test"]:
        s = stats[split]
        md_content += f"\n### {split.capitalize()}\n"
        md_content += f"- **Raw:** {s['raw_count']}\n"
        md_content += f"- **Processed:** {s['processed_count']}\n"
        md_content += f"- **Excluded (Invalid):** {s['excluded_invalid']}\n"
        md_content += f"- **Excluded (Overlap Test):** {s['excluded_overlap_test']}\n"
        if split == "train":
            md_content += (
                f"- **Excluded (Overlap Val):** {s['excluded_overlap_validation']}\n"
            )
            md_content += (
                f"- **Excluded (Train Dup):** {s['excluded_train_duplicate']}\n"
            )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)


def main():
    config = load_config()

    processed_dir = config["paths"]["data_dir"] / "processed" / "xsum"
    if processed_dir.exists():
        logger.error(f"Processed directory already exists at {processed_dir}.")
        logger.error(
            "Aborting to prevent accidental overwriting. Delete it manually if needed."
        )
        return

    raw_dataset = load_raw_dataset(config)

    processed_dataset, stats, exclusion_log = clean_dataset(raw_dataset, config)

    run_sanity_checks(raw_dataset, processed_dataset, stats, config)

    logger.info(f"Saving processed dataset to {processed_dir}")
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_dataset.save_to_disk(str(processed_dir))

    logger.info("Saving cleaning reports...")
    save_reports(stats, exclusion_log, config)

    logger.info("Cleaning pipeline completed successfully.")


if __name__ == "__main__":
    main()
