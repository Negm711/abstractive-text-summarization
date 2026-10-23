"""Launch the trained local summarizer. Run from WSL: python run_local_summary.py."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
# Compatibility exports for existing evaluation scripts.
from summarization.local.settings import ADAPTER_SHA
from summarization.local.prompts import INSTRUCTION, make_prompt
from summarization.local.runtime import get_json, post
from summarization.local.app import main
if __name__ == "__main__":
    main()
