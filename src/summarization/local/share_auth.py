"""Local shared-demo credentials; deliberately stored outside the repository."""
import getpass
import json
import os
from pathlib import Path


def get_share_auth(username="negm"):
    folder = Path.home() / ".config" / "negm_summarizer"
    path = folder / "share_auth.json"
    if path.is_symlink():
        raise RuntimeError("The demo credential file must not be a symlink.")
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        password = saved.get("password")
        if saved.get("username") == username and isinstance(password, str) and len(password) >= 8:
            path.chmod(0o600)
            return username, password
    password = getpass.getpass("Choose a demo password (at least 8 characters; input is hidden): ")
    while len(password) < 8:
        password = getpass.getpass("Password is too short. Enter at least 8 characters: ")
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    folder.chmod(0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"username": username, "password": password}, handle)
        handle.write("\n")
    path.chmod(0o600)
    print("Demo login saved locally outside the project. It will be reused next time.", flush=True)
    return username, password
