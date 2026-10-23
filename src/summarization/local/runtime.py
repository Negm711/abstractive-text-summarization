"""Pinned native runtime setup and local HTTP transport."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request
from .settings import TAG
def checksum(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "NajmSummarizer/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)

def download(url, path, expected=None):
    partial = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "NajmSummarizer/1.0"})
    print("Downloading:", path.name, flush=True)
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
        total = int(response.headers.get("Content-Length", 0))
        written = 0
        last = 0
        while True:
            block = response.read(4 * 1024 * 1024)
            if not block:
                break
            handle.write(block)
            written += len(block)
            if written - last >= 50 * 1024 * 1024:
                print(f"  {written / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
                last = written
    if expected and checksum(partial) != expected.removeprefix("sha256:"):
        raise RuntimeError("Official runtime checksum verification failed.")
    partial.replace(path)

def install_runtime(root):
    destination = root / ".local_tools" / ("llama-" + TAG)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / "installed.json"
    if not marker.is_file():
        release = get_json("https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/" + TAG)
        assets = {asset["name"]: asset for asset in release["assets"]}
        for prefix in ("llama-", "cudart-llama-"):
            name = prefix + TAG + "-bin-ubuntu-cuda-12.8-x64.tar.gz"
            asset = assets[name]
            archive = destination / name
            if not archive.is_file() or archive.stat().st_size != asset["size"]:
                download(asset["browser_download_url"], archive, asset.get("digest"))
            elif asset.get("digest") and checksum(archive) != asset["digest"].removeprefix("sha256:"):
                download(asset["browser_download_url"], archive, asset["digest"])
            with tarfile.open(archive) as bundle:
                # Python 3.10 also provides the data filter in maintained releases.
                if hasattr(tarfile, "data_filter"):
                    bundle.extractall(destination, filter="data")
                else:
                    for member in bundle.getmembers():
                        target = (destination / member.name).resolve()
                        if not target.is_relative_to(destination.resolve()):
                            raise RuntimeError("Unsafe runtime archive entry.")
                        if member.issym() or member.islnk():
                            link = (target.parent / member.linkname).resolve()
                            if not link.is_relative_to(destination.resolve()):
                                raise RuntimeError("Unsafe runtime archive link.")
                    bundle.extractall(destination)
        marker.write_text(json.dumps({"release": TAG}))
        for archive in destination.glob("*.tar.gz"):
            archive.unlink()
    binaries = list(destination.rglob("llama-server"))
    if len(binaries) != 1:
        raise RuntimeError("Could not locate the installed llama-server executable.")
    binary = binaries[0]
    binary.chmod(binary.stat().st_mode | 0o111)
    directories = {str(binary.parent), "/usr/lib/wsl/lib"}
    directories.update(str(path.parent) for path in destination.rglob("*.so*"))
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(sorted(directories)) + ":" + environment.get("LD_LIBRARY_PATH", "")
    probe = subprocess.run([str(binary), "--version"], env=environment, capture_output=True, text=True)
    if probe.returncode:
        raise RuntimeError("Native runtime failed to start:\n" + probe.stdout + probe.stderr)
    return binary, environment

def post(port, endpoint, data):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}", data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)

def choose_gpu_layers():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True, timeout=10)
        free = int(result.stdout.strip().splitlines()[0])
        return 24 if free >= 6900 else 20
    except (OSError, ValueError, subprocess.SubprocessError):
        return 20
