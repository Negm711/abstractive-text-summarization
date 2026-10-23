"""Convert the trained Qwen3 adapter without loading base model weights.

Run from the project root with Python 3.10 or newer. Downloads conversion
tools into an isolated virtual environment; does not modify the active env.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import venv
import zipfile

COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"
REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
ADAPTER_SHA = "667932dc21e1c6acd66c0ccb008deaabda1d8af375c14a6d5024990476138cb3"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "QwenLocalSetup/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def run(command):
    subprocess.run([str(x) for x in command], check=True)


def find_trained_adapter(root):
    artifacts = root / "artifacts"
    candidates = sorted(artifacts.rglob("adapter_model.safetensors"))
    for weights in candidates:
        if (weights.parent / "adapter_config.json").is_file() and digest(weights) == ADAPTER_SHA:
            return weights.parent
    raise FileNotFoundError("The expected trained v2 adapter was not found under artifacts. No download was started.")


def main():
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required.")
    root = Path(__file__).resolve().parents[1]
    package = root / "artifacts/qwen3_14b_curated_v2_1200"
    adapter = find_trained_adapter(root)
    weights = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    base = package / "local_model/Qwen3-14B-Q4_K_M.gguf"
    for path in (weights, config, base):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    metadata = json.loads(config.read_text(encoding="utf-8"))
    if metadata.get("base_model_name_or_path") != "Qwen/Qwen3-14B":
        raise RuntimeError("This adapter is not for Qwen/Qwen3-14B.")
    if digest(weights) != ADAPTER_SHA:
        raise RuntimeError("Adapter checksum does not match the best v2 checkpoint.")
    with base.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise RuntimeError("The downloaded base file is not a GGUF model.")
    output = base.parent / "qwen3_curated_v2_adapter_f16.gguf"
    record = output.with_suffix(".json")
    if output.is_file() and record.is_file():
        previous = json.loads(record.read_text())
        if previous.get("adapter_sha256") == ADAPTER_SHA and previous.get("gguf_sha256") == digest(output):
            print("Trained adapter already converted:", output, flush=True)
            return

    tools = root / ".local_tools/qwen_conversion"
    tools.mkdir(parents=True, exist_ok=True)
    environment = tools / "venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        print("Creating isolated conversion environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    marker = tools / "dependencies_ready.json"
    if not marker.is_file():
        print("Installing CPU conversion tools. Base model weights will not be downloaded.", flush=True)
        run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "typing_extensions",
             "--index-url", "https://pypi.org/simple"])
        run([python, "-m", "pip", "install", "--disable-pip-version-check", "torch==2.11.0",
             "--index-url", "https://download.pytorch.org/whl/cpu"])
        run([python, "-m", "pip", "install", "--disable-pip-version-check",
             "numpy==2.2.6", "transformers==4.57.6", "sentencepiece==0.2.1",
             "protobuf>=4.21,<5", "pyyaml>=6,<7", "tqdm>=4.27", "requests>=2.25"])
        marker.write_text(json.dumps({"torch": "2.11.0", "transformers": "4.57.6"}))

    source = tools / ("llama.cpp-" + COMMIT)
    if not (source / "convert_lora_to_gguf.py").is_file():
        print("Downloading pinned official conversion source...", flush=True)
        archive = tools / "llama_source.zip"
        download("https://codeload.github.com/ggml-org/llama.cpp/zip/" + COMMIT, archive)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                destination = (tools / member.filename).resolve()
                if not destination.is_relative_to(tools.resolve()):
                    raise RuntimeError("Unsafe archive entry.")
            bundle.extractall(tools)
        archive.unlink()
    base_config = tools / "base_config"
    base_config.mkdir(exist_ok=True)
    if not (base_config / "config.json").is_file():
        download(f"https://huggingface.co/Qwen/Qwen3-14B/resolve/{REVISION}/config.json",
                 base_config / "config.json")
    temporary_output = output.with_suffix(".gguf.partial")
    print("Converting the trained adapter on CPU...", flush=True)
    run([python, source / "convert_lora_to_gguf.py", "--base", base_config,
         "--outtype", "f16", "--outfile", temporary_output, adapter])
    with temporary_output.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise RuntimeError("Conversion output failed GGUF format validation.")
    temporary_output.replace(output)
    record.write_text(json.dumps({"adapter_sha256": ADAPTER_SHA,
                                 "gguf_sha256": digest(output),
                                 "base_revision": REVISION,
                                 "converter_commit": COMMIT}, indent=2))
    print("SUCCESS: Trained adapter converted.", flush=True)
    print("Adapter:", output, flush=True)
    print("No training or base model download was performed.", flush=True)


if __name__ == "__main__":
    main()
