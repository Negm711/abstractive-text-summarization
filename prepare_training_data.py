"""Run the project prepare training data script."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts/prepare_training_data.py"), run_name="__main__")
