"""Application lifecycle: native server, summarization service and Gradio UI."""

import argparse
import atexit
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
from .share_auth import get_share_auth
from .settings import PROJECT_ROOT, ADAPTER_SHA, TAG, SETTINGS
from .prompts import POLICIES
from .runtime import checksum, get_json, install_runtime, choose_gpu_layers
from .service import LocalSummarizer, SummaryError
from .ui import build_interface, metric_cards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-layers", type=int, default=None)
    parser.add_argument("--port", type=int, default=SETTINGS["gradio_port"])
    parser.add_argument("--server-port", type=int, default=SETTINGS["server_port"])
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument(
        "--policy", choices=list(POLICIES), default=SETTINGS["default_policy"]
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify local files without loading the model",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a password-protected public Gradio link",
    )
    parser.add_argument(
        "--username", default="negm", help="Username for the shared demo"
    )
    args = parser.parse_args()
    if args.gpu_layers is None:
        args.gpu_layers = choose_gpu_layers()
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("Run this script in the WSL terminal.")
    root = PROJECT_ROOT
    model_dir = root / "artifacts/qwen3_14b_curated_v2_1200/local_model"
    base = model_dir / "Qwen3-14B-Q4_K_M.gguf"
    adapter = model_dir / "qwen3_curated_v2_adapter_f16.gguf"
    record = adapter.with_suffix(".json")
    for path in (base, adapter, record):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")
    identity = json.loads(record.read_text())
    if identity.get("adapter_sha256") != ADAPTER_SHA or checksum(
        adapter
    ) != identity.get("gguf_sha256"):
        raise RuntimeError("Trained adapter identity verification failed.")
    if args.check:
        print("Verified trained adapter:", adapter)
        print("Base GGUF present:", base)
        print("No model was loaded or downloaded.")
        return
    # Reuse the isolated environment created during adapter conversion.
    python = root / ".local_tools/qwen_conversion/venv/bin/python"
    if not python.is_file():
        raise RuntimeError("The isolated conversion environment was not found.")
    if Path(sys.executable).absolute() != python.absolute():
        subprocess.run(
            [str(python), str(PROJECT_ROOT / "run_local_summary.py"), *sys.argv[1:]],
            check=True,
        )
        return
    if importlib.util.find_spec("gradio") is None:
        print("Installing Gradio into the isolated local environment...", flush=True)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "gradio==5.49.1",
                "--index-url",
                "https://pypi.org/simple",
            ],
            check=True,
        )
    import gradio as gr

    auth = None
    if args.share:
        if not args.username.strip():
            raise ValueError("The demo username cannot be empty.")
        auth = get_share_auth(args.username)
        print("Public sharing enabled. Login username:", args.username, flush=True)
    binary, environment = install_runtime(root)
    for port in (args.port, args.server_port):
        with socket.socket() as connection:
            if connection.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(
                    f"Port {port} is already in use. Close the previous app first."
                )
    logs = root / "reports/local_runtime"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "llama_server.log"
    log = log_path.open("w", encoding="utf-8")
    command = [
        str(binary),
        "-m",
        str(base),
        "--lora",
        str(adapter),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.server_port),
        "--ctx-size",
        "4096",
        "--parallel",
        "1",
        "--batch-size",
        "128",
        "--ubatch-size",
        "64",
        "--n-gpu-layers",
        str(args.gpu_layers),
        "--threads",
        str(args.threads),
        "--threads-batch",
        str(args.threads),
    ]
    server = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env=environment
    )

    def cleanup():
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        log.close()

    atexit.register(cleanup)
    print("Loading the base model AND the verified trained adapter...", flush=True)
    print("GPU layers:", args.gpu_layers, "| CPU threads:", args.threads, flush=True)
    print("Runtime log:", log_path, flush=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if server.poll() is not None:
            log.flush()
            failure = log_path.read_text(errors="replace")[-12000:]
            memory_failure = any(
                term in failure.lower()
                for term in ("out of memory", "cuda error 2", "failed to allocate")
            )
            if args.gpu_layers > 20 and memory_failure:
                print(
                    "GPU memory was insufficient; retrying with 20 GPU layers.",
                    flush=True,
                )
                args.gpu_layers = 20
                command[command.index("--n-gpu-layers") + 1] = "20"
                server = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=environment
                )
                continue
            raise RuntimeError("Runtime stopped:\n" + failure[-7000:])
        try:
            if (
                get_json(f"http://127.0.0.1:{args.server_port}/health").get("status")
                == "ok"
            ):
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1)
    else:
        raise RuntimeError(f"Model startup timed out. See {log_path}")
    (logs / "launch.json").write_text(
        json.dumps(
            {
                "release": TAG,
                "command": command,
                "adapter": identity,
                "policy": args.policy,
                "instruction": POLICIES[args.policy],
            },
            indent=2,
        )
    )

    exports = tempfile.TemporaryDirectory(prefix="negm_summaries_")
    atexit.register(exports.cleanup)
    client = LocalSummarizer(port=args.server_port, policy=args.policy)

    def summarize(article, review, progress=gr.Progress()):
        try:
            result = client.summarize(article, review=review, progress=progress)
        except (SummaryError, urllib.error.URLError, TimeoutError) as error:
            raise gr.Error(str(error)) from error
        destination = Path(exports.name) / f"summary_{time.time_ns()}.txt"
        destination.write_text(result["summary"] + "\n", encoding="utf-8")
        result.update(gpu_layers=args.gpu_layers, cpu_threads=args.threads)
        progress(1, desc="Done")
        return (
            result["summary"],
            metric_cards(
                result["article_words"],
                result["summary_words"],
                result["total_seconds"],
            ),
            result["note"],
            result["draft"],
            result,
            gr.update(value=str(destination), interactive=True),
        )

    demo = build_interface(gr, summarize)
    print("READY: http://127.0.0.1:" + str(args.port), flush=True)
    try:
        demo.queue(default_concurrency_limit=1, max_size=8).launch(
            server_name="127.0.0.1",
            server_port=args.port,
            share=args.share,
            auth=auth,
            show_api=False,
            inbrowser=False,
            auth_message="Private demo. Enter the access details shared by NeGm.",
        )
    finally:
        cleanup()


if __name__ == "__main__":
    main()
