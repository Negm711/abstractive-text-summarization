# Frozen final evaluation

The inference policy is fixed: curated-v2 adapter, local Q4_K_M base, baseline instruction, greedy generation, no source-review pass and no prompt cache. This evaluation makes no further training changes.

Preparation reads the existing local processed XSum test split. It excludes IDs and normalized article hashes from the recorded training/validation and known development examples. Selection sorts by SHA256(seed:id), takes 100 unique articles, and applies no length or summary-quality filter. The seed is 20261024. Exact overlap checks do not certify semantic uniqueness or absence from base-model pretraining.

The frozen sample records dataset provenance, exclusions, training-split checksum and inference instruction. Never replace test examples after seeing their outputs. Context rejections and failures remain in reporting; do not silently substitute easier articles.

## Commands

```bash
python -m summarization.local.final_evaluation prepare
```

Once prepared, start the existing local app and wait for READY. In a second terminal:

```bash
python -m summarization.local.final_evaluation generate
```

The app must be the baseline policy and verified v2 adapter. Generation records are saved atomically after each article. If interrupted, rerun the same generation command. Changed contracts are rejected rather than merged. Avoid using the UI while generation is running to prevent request interleaving and distorted latency.

## Interpretation

The original XSum reference is retained for ROUGE comparison; it is not assumed source-faithful. Source-based review must check unsupported facts, altered uncertainty, reversed roles and core-event coverage. Record reviewer identity (human or assistant) explicitly. Automatic similarity, compression and word-limit compliance are not factual-accuracy percentages.

Final metrics must include lexical comparison, output completion, word counts, compliance, latency and a documented source-based review. Do not claim a quality improvement over the base without a matched comparison. Generated outputs are private working reports containing upstream articles; publish aggregate results separately.
