# Setup and operations

## Existing local workstation

Use WSL and Python 3.10+. The root launcher reuses `.local_tools/qwen_conversion/venv` and does not upgrade the Conda environment. The model and converted adapter should already exist in the documented local_model directory.

```bash
python run_local_summary.py --check
python run_local_summary.py
```

If GPU memory is insufficient during startup, the launcher retries at 20 layers after a higher automatic setting. Explicit controls are available:

```bash
python run_local_summary.py --gpu-layers 20 --threads 8
```

A 127.0.0.1 port collision means an earlier application is still running. Stop that application with Ctrl+C before starting another. Do not delete or redownload the model to resolve a port collision.

## First-time downloads

The official Qwen GGUF base is approximately 9 GB. The official native CUDA 12.8 binaries/libraries total approximately 770 MB. CPU conversion dependencies and Gradio are additional downloads. Native source/model revisions are pinned. A running local summary request uses no remote inference API.

`prepare_local_adapter.py` validates the expected trained-adapter SHA256, creates its isolated conversion environment and converts LoRA without downloading/loading BF16 base weights. Keep the user's verified adapter path or place the v2 adapter under the documented export directory. The conversion requires the base configuration only, not the full original checkpoint.

## Separate dependency scopes

- `requirements/local.txt`: Gradio application.
- `requirements/reports.txt`: offline figures.
- `requirements/colab-training.txt`: recorded Colab fine-tuning packages; install only with a suitable CUDA PyTorch environment.
- `requirements/data.txt`: dataset loading, cleaning and curation.

For Colab training reproduction see `scripts/train_qwen3_a100_curated_v2.py` and its data/Drive requirements. Do not rerun training merely to launch the local app.

## Routine shutdown and resume

A generated comparison stage is saved immediately. Ctrl+C stops generation; rerun the same evaluation command to resume. Ctrl+C in the application terminal closes Gradio and its native child server. Outputs and model files stay on disk. Summary download files are temporary; save the download if it should survive application shutdown.

## VS Code and temporary sharing

Open the folder in a WSL VS Code window. The existing local Run and Debug configurations are intentionally preserved by the cleanup update. Start one instance, wait for READY, then open the local or newly generated public link. Do not run two copies on the same ports.

`python run_local_summary.py --share` prompts for a password only when no matching saved login is available. Saved credentials live outside the repository in `~/.config/negm_summarizer/share_auth.json`; do not upload this file. The share link stops working when the app or host stops.

Setup implementation scripts now live in `scripts/`. Existing root commands delegate to them and retain the same artifact locations.

## Regenerate recorded figures

Install `requirements/reports.txt` in a reporting environment, then run `PYTHONPATH=src python -m summarization.local.reporting`. The recorded Qwen logs must be under `reports/training/qwen3_curated_v2/`. Historical raw-data and local-quality figures also require their existing local source reports, which contain upstream articles and are not distributed as public Git content.
