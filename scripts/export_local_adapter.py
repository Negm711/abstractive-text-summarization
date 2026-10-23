"""Package only the verified deployed adapter and its identity record, offline."""
from pathlib import Path
import json
import zipfile
from setup_local import MODEL_DIR, ADAPTER, ADAPTER_SHA, SOURCE_SHA, ROOT, digest

def main():
    weights = MODEL_DIR / ADAPTER
    record = weights.with_suffix(".json")
    identity = json.loads(record.read_text())
    if identity.get("adapter_sha256") != SOURCE_SHA or identity.get("gguf_sha256") != ADAPTER_SHA or digest(weights) != ADAPTER_SHA:
        raise ValueError("Trained adapter identity mismatch; no bundle exported.")
    destination = ROOT / "artifacts/release/negm-qwen3-v2-local-adapter.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(weights, weights.name)
        archive.write(record, record.name)
    temporary.replace(destination)
    print("Bundle:", destination)
    print("Size MiB:", round(destination.stat().st_size / 1024**2, 2))
    print("SHA256:", digest(destination))
    print("No base weights, articles, credentials or runtime files were included.")

if __name__ == "__main__":
    main()
