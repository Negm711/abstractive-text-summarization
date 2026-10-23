"""Prepare a Linux/WSL runtime; large model download requires an explicit flag."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts/qwen3_14b_curated_v2_1200/local_model"
BASE = "Qwen3-14B-Q4_K_M.gguf"
ADAPTER = "qwen3_curated_v2_adapter_f16.gguf"
BASE_SHA = "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"
ADAPTER_SHA = "9d59dfce93b8c5fefc77369ca060a35eee33dc199ad27a1ab312189b4f9adae6"
SOURCE_SHA = "667932dc21e1c6acd66c0ccb008deaabda1d8af375c14a6d5024990476138cb3"
REVISION = "c75e7b2d0234068f674a1bacf548ea32e27ccd29"

def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

def install_bundle(bundle):
    record_name = Path(ADAPTER).with_suffix(".json").name
    with zipfile.ZipFile(bundle) as archive:
        if sorted(archive.namelist()) != sorted([ADAPTER, record_name]):
            raise ValueError("Bundle must contain exactly the adapter GGUF and its JSON record.")
        metadata = json.loads(archive.read(record_name))
        if metadata.get("adapter_sha256") != SOURCE_SHA or metadata.get("gguf_sha256") != ADAPTER_SHA:
            raise ValueError("Adapter record does not match the released v2 identity.")
        destination = MODEL_DIR / ADAPTER
        if destination.exists():
            if digest(destination) != ADAPTER_SHA:
                raise ValueError("Existing adapter differs; refusing to overwrite it.")
        else:
            partial = destination.with_suffix(".partial")
            try:
                with archive.open(ADAPTER) as source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if digest(partial) != ADAPTER_SHA:
                    raise ValueError("Adapter checksum mismatch.")
                partial.replace(destination)
            finally:
                partial.unlink(missing_ok=True)
        record = MODEL_DIR / record_name
        encoded = json.dumps(metadata, indent=2) + "\n"
        if record.exists() and json.loads(record.read_text()) != metadata:
            raise ValueError("Existing adapter record differs; refusing to overwrite it.")
        if not record.exists():
            record.write_text(encoded, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--download-base", action="store_true", help="Download approximately 9 GB from official Qwen hosting")
    parser.add_argument("--check", action="store_true", help="Check local model checksums only; no downloads or environment changes")
    args = parser.parse_args()
    if args.check and (args.bundle or args.download_base):
        parser.error("--check cannot be combined with installation options")
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64") or sys.version_info < (3,10):
        raise RuntimeError("Use Linux/WSL x86_64 and Python 3.10 or newer.")
    if not args.check:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if args.bundle:
            install_bundle(args.bundle)
    base = MODEL_DIR / BASE
    if not base.exists() and args.download_base:
        if shutil.disk_usage(MODEL_DIR).free < 11 * 1024**3:
            raise RuntimeError("At least 11 GiB free disk is required before the base download.")
        url = f"https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/{REVISION}/{BASE}"
        partial = base.with_suffix(".gguf.part")
        print("Downloading pinned Qwen base (approximately 9 GB)...", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent":"NeGmSetup/1.0"})
        with urllib.request.urlopen(request, timeout=180) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output)
        if digest(partial) != BASE_SHA:
            raise ValueError("Base checksum mismatch; completed file was not installed.")
        partial.replace(base)
    for path, expected in ((base, BASE_SHA), (MODEL_DIR / ADAPTER, ADAPTER_SHA)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path.name}. Supply the trained bundle and use --download-base if needed.")
        if digest(path) != expected:
            raise ValueError(f"Checksum mismatch: {path.name}")
    subprocess.run([sys.executable, str(ROOT / "run_local_summary.py"), "--check"], check=True)
    if args.check:
        return
    environment = ROOT / ".local_tools/qwen_conversion/venv"
    python = environment / "bin/python"
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    installed = subprocess.run([str(python), "-c", "import importlib.metadata as m; assert m.version('gradio') == '5.49.1'"], capture_output=True)
    if installed.returncode:
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements/local.txt")], check=True)
    print("Setup complete. Run: python run_local_summary.py")
    print("First launch may download the pinned native CUDA runtime (approximately 770 MB).")

if __name__ == "__main__":
    main()
