"""Project-relative paths and pinned identity, with executable local defaults."""
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SETTINGS = json.loads((PROJECT_ROOT / "configs/qwen_local.json").read_text(encoding="utf-8"))
MODEL = SETTINGS["model"]
BASE_REVISION = SETTINGS["base_revision"]
TAG = SETTINGS["runtime_release"]
ADAPTER_SHA = SETTINGS["adapter_sha256"]
if (MODEL != "Qwen/Qwen3-14B" or
    ADAPTER_SHA != "667932dc21e1c6acd66c0ccb008deaabda1d8af375c14a6d5024990476138cb3"):
    raise ValueError("Local configuration does not identify the expected trained Qwen3 v2 model.")
MODEL_DIRECTORY = PROJECT_ROOT / "artifacts/qwen3_14b_curated_v2_1200/local_model"
