"""Reproducible scientific claim verification experiments."""
from pathlib import Path
import os

# Must precede imports of huggingface_hub/transformers, which resolve cache paths at import.
os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
