# Final evaluation: Qwen3 curated v2

The deployed Q4_K_M base and trained adapter completed all 100 frozen test articles. The inference policy was fixed before generation; no examples were replaced based on their outputs. IDs and normalized article hashes have zero overlap with the 1,515 known curation/development examples retained in the exclusion snapshot. This does not rule out semantic duplicates or pretraining exposure.

## Recorded performance

| Measure | Result |
|---|---:|
| Completed outputs | 100 / 100 |
| Mean article length | 363.83 words |
| Mean summary length | 32.40 words |
| Summaries within 40 words | 96 / 100 |
| Mean word reduction | 84.37% |
| Mean generation time | 18.83 s |
| Median generation time | 18.02 s |
| 95th percentile generation time | 28.29 s |
| ROUGE-1 / ROUGE-2 / ROUGE-L F1 | 0.2655 / 0.0640 / 0.1877 |

ROUGE uses `rouge-score==0.1.2`, stemming and macro mean F1 against original XSum references. Those references are not assumed source-faithful. Latency describes one sequential run on the Quadro RTX 4000 with CPU offloading, 24 GPU layers and eight threads, with no prompt cache and startup excluded. No matched base-model run was made on this final set, so these results do not establish a causal fine-tuning gain.

## Source-based review

An assistant compared the summaries with the supplied articles. The labels are qualitative and are not independent human annotations:

- 60 summaries: no clear issue identified.
- 19: wording, attribution or qualification concerns.
- 4: unsupported secondary details.
- 17: material factual issues involving an event, actor, role, policy, location, quantity or outcome.

The observed material-issue fraction is 17/100 under this review, not a population error guarantee or proof of 83% accuracy. Additional categories were kept separate rather than silently counted as fully correct. Some judgments about ambiguity and missing source support can vary between reviewers.

Examples of material issues include attributing Leicester's rejected bid to Evans instead of West Brom (ID 40954441), crediting Boco with both goals instead of one goal and an assist to Clark (37807179), describing prison terms as suspended (23352688), and including Wales in a current strike that the article says was averted there (29846243).

The app is a concise news-reading demo with a measured factuality limitation. Important names, numbers, legal outcomes and medical details should be checked against the source. The final set will not be reused for further tuning while still being called untouched.

## Files and reproduction

Public aggregate metrics: `reports/metrics/final_test_100.json`. Figures: `reports/figures/10_final_test_runtime.png` and `11_final_test_source_review.png` (SVG copies included).

Full article-bearing results and per-example assistant review stay locally under `reports/final_evaluation/`, excluded from Git. Recompute metrics with `python scripts/score_final_evaluation.py` after installing `requirements/evaluation.txt`. The script performs no model inference or training.
