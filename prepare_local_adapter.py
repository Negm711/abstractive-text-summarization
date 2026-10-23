"""Run the project prepare local adapter script."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts/prepare_local_adapter.py"), run_name="__main__")
