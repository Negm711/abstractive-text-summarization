import json
import logging
from datetime import datetime, timezone

import datasets
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi

from summarization.config import load_config

logger = logging.getLogger(__name__)


def load_raw_dataset(config: dict) -> DatasetDict:
    """Load the official dataset at a resolved commit without remote code."""
    dataset_name = config["data"]["dataset_name"]
    revision = config["data"]["revision"]
    cache_dir = config["paths"]["data_dir"] / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Resolving revision '%s' for dataset '%s'...",
        revision,
        dataset_name,
    )

    try:
        dataset_info = HfApi().dataset_info(
            dataset_name,
            revision=revision,
        )
        commit_sha = dataset_info.sha

        if not commit_sha:
            raise ValueError("Hugging Face did not return a commit SHA.")

    except Exception as error:
        raise RuntimeError(
            f"Cannot resolve revision '{revision}' " f"for dataset '{dataset_name}'."
        ) from error

    logger.info("Resolved commit SHA: %s", commit_sha)
    logger.info("Loading dataset...")

    try:
        raw_dataset = load_dataset(
            dataset_name,
            revision=commit_sha,
            cache_dir=str(cache_dir),
            trust_remote_code=False,
        )
    except Exception as error:
        raise RuntimeError(
            "Dataset loading failed. No fallback mirror was used."
        ) from error

    if not isinstance(raw_dataset, DatasetDict):
        raise TypeError("Expected a DatasetDict containing the official splits.")

    expected_splits = ("train", "validation", "test")
    missing_splits = set(expected_splits) - set(raw_dataset.keys())

    if missing_splits:
        raise ValueError(f"Missing required splits: {sorted(missing_splits)}")

    required_columns = {
        config["data"]["source_column"],
        config["data"]["target_column"],
        config["data"]["id_column"],
    }

    for split in expected_splits:
        missing_columns = required_columns - set(raw_dataset[split].column_names)

        if missing_columns:
            raise ValueError(
                f"Missing columns in '{split}': " f"{sorted(missing_columns)}"
            )

    config["resolved_commit_sha"] = commit_sha
    return raw_dataset


def audit_dataset(dataset: DatasetDict, config: dict) -> dict:
    """Inspect data quality without modifying any examples."""
    logger.info("Starting dataset audit...")

    source_col = config["data"]["source_column"]
    target_col = config["data"]["target_column"]
    id_col = config["data"]["id_column"]

    audit_report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_name": config["data"]["dataset_name"],
            "requested_revision": config["data"]["revision"],
            "resolved_commit_sha": config.get("resolved_commit_sha", "unknown"),
            "datasets_version": datasets.__version__,
            "duplicate_definition": (
                "Extra occurrences after the first; text duplicates count "
                "only non-empty strings using exact matching without "
                "normalization."
            ),
            "overlap_definition": (
                "Number of distinct valid document texts shared between "
                "splits, not the number of matching rows."
            ),
            "text_validity_definition": (
                "A valid text is a string containing at least one "
                "non-whitespace character. Null and non-string counts "
                "are separate."
            ),
        },
        "splits_metrics": {},
        "cross_split_overlap": {},
    }

    valid_docs_per_split = {}

    for split in ("train", "validation", "test"):
        logger.info("Auditing split: %s...", split)
        split_data = dataset[split]

        metrics = {
            "total_rows": len(split_data),
            "columns": split_data.column_names,
            "column_types": {
                col: str(split_data.features[col]) for col in split_data.column_names
            },
            "source_column": {
                "nulls": 0,
                "non_strings": 0,
                "empty_or_whitespace": 0,
            },
            "target_column": {
                "nulls": 0,
                "non_strings": 0,
                "empty_or_whitespace": 0,
            },
            "ids": {
                "missing_null": 0,
                "duplicates": 0,
            },
            "duplicates": {
                "exact_documents": 0,
                "exact_doc_sum_pairs": 0,
                "exact_summaries_info_only": 0,
            },
        }

        seen_ids = set()
        seen_docs = set()
        seen_pairs = set()
        seen_sums = set()

        # Iterate over rows without creating full Python column lists.
        audit_columns = split_data.select_columns([source_col, target_col, id_col])

        for row in audit_columns:
            document = row[source_col]
            summary = row[target_col]
            example_id = row[id_col]

            if example_id is None or not str(example_id).strip():
                metrics["ids"]["missing_null"] += 1
            else:
                if example_id in seen_ids:
                    metrics["ids"]["duplicates"] += 1
                seen_ids.add(example_id)

            document_valid = True

            if document is None:
                metrics["source_column"]["nulls"] += 1
                document_valid = False
            elif not isinstance(document, str):
                metrics["source_column"]["non_strings"] += 1
                document_valid = False
            elif not document.strip():
                metrics["source_column"]["empty_or_whitespace"] += 1
                document_valid = False

            summary_valid = True

            if summary is None:
                metrics["target_column"]["nulls"] += 1
                summary_valid = False
            elif not isinstance(summary, str):
                metrics["target_column"]["non_strings"] += 1
                summary_valid = False
            elif not summary.strip():
                metrics["target_column"]["empty_or_whitespace"] += 1
                summary_valid = False

            if document_valid:
                if document in seen_docs:
                    metrics["duplicates"]["exact_documents"] += 1
                seen_docs.add(document)

            if summary_valid:
                if summary in seen_sums:
                    metrics["duplicates"]["exact_summaries_info_only"] += 1
                seen_sums.add(summary)

            if document_valid and summary_valid:
                pair = (document, summary)

                if pair in seen_pairs:
                    metrics["duplicates"]["exact_doc_sum_pairs"] += 1

                seen_pairs.add(pair)

        audit_report["splits_metrics"][split] = metrics
        valid_docs_per_split[split] = seen_docs

        # Only document sets are needed for cross-split comparisons.
        del seen_ids, seen_pairs, seen_sums

    logger.info("Calculating exact cross-split document overlap...")

    audit_report["cross_split_overlap"] = {
        "note": (
            "Exact matching of distinct valid document texts. "
            "Near-duplicates are not detected."
        ),
        "train_validation_overlap": len(
            valid_docs_per_split["train"] & valid_docs_per_split["validation"]
        ),
        "train_test_overlap": len(
            valid_docs_per_split["train"] & valid_docs_per_split["test"]
        ),
        "validation_test_overlap": len(
            valid_docs_per_split["validation"] & valid_docs_per_split["test"]
        ),
    }

    return audit_report


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config()
    dataset = load_raw_dataset(config)
    audit_report = audit_dataset(dataset, config)

    reports_dir = config["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "data_audit.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            audit_report,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        file.write("\n")

    logger.info("=== Data Audit Summary ===")
    logger.info(
        "Dataset SHA: %s",
        audit_report["metadata"]["resolved_commit_sha"],
    )

    for split, metrics in audit_report["splits_metrics"].items():
        logger.info(
            "%s | Rows: %s | Duplicate documents: %s | "
            "Duplicate pairs: %s | Duplicate summaries: %s",
            split.upper(),
            metrics["total_rows"],
            metrics["duplicates"]["exact_documents"],
            metrics["duplicates"]["exact_doc_sum_pairs"],
            metrics["duplicates"]["exact_summaries_info_only"],
        )

        for column_key in ("source_column", "target_column"):
            counts = metrics[column_key]

            if any(counts.values()):
                logger.warning(
                    "%s %s | Null: %s | Non-string: %s | Empty: %s",
                    split,
                    column_key,
                    counts["nulls"],
                    counts["non_strings"],
                    counts["empty_or_whitespace"],
                )

        if any(metrics["ids"].values()):
            logger.warning(
                "%s IDs | Missing: %s | Duplicate: %s",
                split,
                metrics["ids"]["missing_null"],
                metrics["ids"]["duplicates"],
            )

    overlaps = audit_report["cross_split_overlap"]

    logger.info("=== Overlap Summary ===")
    logger.info(
        "Train/Validation: %s",
        overlaps["train_validation_overlap"],
    )
    logger.info("Train/Test: %s", overlaps["train_test_overlap"])
    logger.info(
        "Validation/Test: %s",
        overlaps["validation_test_overlap"],
    )
    logger.info("Full report saved to: %s", report_path)


if __name__ == "__main__":
    main()
