# Evaluation and inference-policy selection

## Keep the splits separate

The curated 924/231 split is train/validation, not train/test. The first 20 validation records were inspected during development. Older files named `test_100` were used repeatedly in prior development and must not be relabeled as untouched final-test evidence.

After selecting a policy, freeze a fresh sample with no train/validation ID or normalized-article overlap and no previous development inspection. Report the selection rule, seed, exclusions and dataset/model revisions. Near-duplicate/semantic leakage requires additional review; an exact hash check does not certify its absence.

## Current 20-example decision

Source: `reports/local_quality/direct_vs_review_validation_20.json`.

- All 20 generations completed in both modes.
- Nineteen pairs were identical; one added “the”. No substantive correction was observed.
- Direct generation averaged 20.90 seconds; the review stage averaged 21.46 seconds in addition to the draft.
- The default remains direct baseline inference. Source review is experimental and unchecked.
- Assistant qualitative review found two clear factual-error examples, plus three uncertainty/wording concerns. The assistant review records the evidence and suggested corrections; this is not an independent human labeling exercise.

## Candidate decision

The alternate `grounded-v1` instructions were evaluated on the same 20 validation articles. They corrected some earlier issues but introduced another unsupported claim and did not reduce the clear-error count in the assistant assessment. The candidate was not promoted. Baseline remains the fixed default; no further prompt tuning is planned before final evaluation.

The three selected UI samples are validation demonstrations. Exclude them, all curated records and previously inspected examples when freezing final-test candidates. Their successful summaries do not establish overall accuracy.

## Final reporting

For the frozen final policy, use ROUGE-1/2/L as lexical reference comparisons, semantic similarity only with its scoring-model identity recorded, and a source-based factuality/coverage review with a documented rubric. Include compression, 40-word compliance and latency distribution. ROUGE, NLI and semantic similarity are not percentages of factual accuracy.

Do not publish unsupported 90% accuracy, unmeasured training gains, or a final test claim for development samples. Distinguish independent human annotation, assistant review and automatic checks explicitly.
