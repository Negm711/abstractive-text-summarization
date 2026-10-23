# Data Quality & Cleaning Report
**Date:** 2026-10-12 23:55:09
**Dataset SHA:** 7d4d486c2f8ef850b1a11aead99b894ff3dd7da9

## Cleaning Policies
- Invalid text (empty, null, whitespace) is removed from all datasets.
- **Test Set:** Retains exact duplicates. Excludes invalid text only. Kept to mirror official benchmark.
- **Validation Set:** Excludes docs present in the Test set.
- **Train Set:** Excludes docs present in Test or Validation sets. Intra-train duplicates are removed (first occurrence kept).
- **Target summaries:** Exact duplicate summaries with different source docs are KEPT.

## Results

### Train
- **Raw:** 204045
- **Processed:** 203832
- **Excluded (Invalid):** 28
- **Excluded (Overlap Test):** 43
- **Excluded (Overlap Val):** 55
- **Excluded (Train Dup):** 87

### Validation
- **Raw:** 11332
- **Processed:** 11326
- **Excluded (Invalid):** 5
- **Excluded (Overlap Test):** 1

### Test
- **Raw:** 11334
- **Processed:** 11333
- **Excluded (Invalid):** 1
- **Excluded (Overlap Test):** 0
