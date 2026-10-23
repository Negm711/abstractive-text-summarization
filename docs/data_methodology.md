# Data methodology and AMIT mapping

The historical XSum audit and the current curated training sample are distinct artifacts. The historical dataset audit uses the pinned dataset revision recorded in `configs/local.yaml`. The curated set combines 1,200 reviewed articles: 1,155 eligible, 45 withheld; 924 train and 231 validation, seed 42. Targets were AI-reviewed and source-checked, not independently human certified.

| AMIT step | Application to this generative NLP project |
|---|---|
| Explore data | Schema, example inspection, source/target word and token lengths; distinguish raw XSum from curated targets |
| Duplicates | Exact IDs/articles, normalized article hashes and cross-split overlap; do not remove distinct articles merely because targets match |
| Missing values | Exclude unusable source/target pairs; do not impute missing article prose with means or fabricated text |
| Outliers | Inspect length tails and context fit; a long valid article is not automatically erroneous |
| Visualizations | Data-quality counts, retention, source/summary lengths, compression, latency, length compliance and documented qualitative review |
| Imbalance | Inspect topic/length coverage; class-level SMOTE is not appropriate for article-summary text generation |
| Over/underfitting | Compare recorded train/validation loss and source-based outputs; use actual Qwen history for curves |
| Evaluation | Lexical/semantic comparisons plus factuality, main-event coverage, fluency, compression and runtime |

The existing `prepare_training_data.py` freezes the eligible set and split, rejects duplicate IDs/normalized articles and records source digests. It starts a new adapter from the pinned base, not a pilot adapter that may have seen validation examples.

## Audit caveat requiring reconciliation

The historical raw audit and cleaning report record different overlap/removal counts and do not provide a shared content digest proving that both ran on identical cached bytes. Distinct-document overlaps and removed-row counts are also different measures. Do not merge these numbers into a single leakage claim until the underlying datasets are re-audited together. Existing figures explicitly identify which report they visualize.

XSum reference summaries can contain unsupported details. Source-check targets and withhold records with insufficient source evidence. No fixed universal data-quality percentage or guaranteed example count is assumed.
