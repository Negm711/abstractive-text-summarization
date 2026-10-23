# Model card: Qwen3-14B curated v2

## Intended use

Short English news summaries for portfolio demonstration and general reading support. Not trained or evaluated here for Arabic summaries, arbitrary documents, medical/legal decisions or public multi-user serving.

## Model and training identity

- Base: Qwen/Qwen3-14B, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`.
- Method: QLoRA; NF4 with BF16 compute, LoRA rank 8, alpha 16, dropout 0.05, linear layers.
- Dataset: 924 curated training articles, 231 validation, seed 42.
- Three epochs, 348 optimizer steps; best validation at step 232, epoch 2.
- Best validation loss: 1.136410117149353.
- Best adapter SHA256: `667932dc21e1c6acd66c0ccb008deaabda1d8af375c14a6d5024990476138cb3`.
- Converted adapter SHA256: `9d59dfce93b8c5fefc77369ca060a35eee33dc199ad27a1ab312189b4f9adae6`.
- Converter commit: `620d2225cc436ab0b001e3200e590e3b286ef9ab`.

The complete 74-record run log is retained in `reports/training/qwen3_curated_v2/training_history.json`: 70 training observations, three validation observations and one final summary. Best-checkpoint metadata matches the selected adapter SHA256. The figures use only these recorded observations.

## Local inference

Official Qwen3-14B Q4_K_M GGUF with the trained adapter loaded separately. llama.cpp `b11669`, CUDA 12.8 runtime, 4,096 context, 160 new-token cap, greedy generation, thinking disabled. Recorded local comparison used 24 GPU layers and eight CPU threads on a Quadro RTX 4000 host. Quantization can alter outputs relative to BF16 Colab inference; the local comparison evaluates this deployed path directly.

## Known practical limits

The 40-word instruction is occasionally exceeded. Unsupported event status, negative claims and awkward relational compression appeared in the development sample. Same-model review did not fix these cases. Long inputs are rejected without truncation; model startup is separate from generation latency.

## Redistribution

Base weights, adapter, data and native runtime binaries are excluded from Git. Check each upstream license and dataset redistribution terms before publishing weights or article text; this project does not relicense upstream assets. The repository currently contains article-bearing development reports that require a publication decision before a public GitHub push.
