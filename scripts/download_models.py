"""Download only required public weights into the project's own cache."""
from pathlib import Path
import os
import sys
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download
from scifact_project.model_revisions import revision

MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "cross-encoder/ms-marco-MiniLM-L6-v2",
          "allenai/specter2_base", "allenai/specter2", "allenai/specter2_adhoc_query"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-scibert', action='store_true', help='Also cache the full GPU pair encoder')
    args = parser.parse_args()
    if args.include_scibert:
        MODELS.append('allenai/scibert_scivocab_uncased')
    for model in MODELS:
        print("Downloading", model, flush=True)
        path = snapshot_download(model, revision=revision(model), allow_patterns=["*.json", "*.txt", "*.safetensors",
                                                        "pytorch_model.bin", "pytorch_adapter.bin"],
                                 ignore_patterns=["onnx/*", "openvino/*"])
        print("Ready", path, flush=True)
