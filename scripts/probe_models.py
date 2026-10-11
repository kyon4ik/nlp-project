"""Real forward passes, including the pinned SPECTER2 adapter compatibility fix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scifact_project.config import Config
from scifact_project.training import PairClassifier
from scifact_project.retrieval import load_dense
from scifact_project.model_revisions import revision
from adapters import AutoAdapterModel
from transformers import AutoTokenizer
import torch

c = Config.for_mode("smoke").prepare()
model = load_dense(c)
print("MiniLM", model.encode(["Aspirin reduces pain."]).shape, flush=True)
p = PairClassifier(c)
print("Pair classifier", p.probabilities([{"claim": "Aspirin reduces pain.", "text": "Pain fell with aspirin."}]), flush=True)
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base", revision=revision("allenai/specter2_base"))
specter = AutoAdapterModel.from_pretrained("allenai/specter2_base", revision=revision("allenai/specter2_base"))
specter.load_adapter("allenai/specter2", source="hf", load_as="document", set_active=True,
                     version=revision("allenai/specter2"))
specter.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="query",
                     version=revision("allenai/specter2_adhoc_query"))
specter = specter.to(c.device).eval()
for adapter, text in [("document", "Aspirin" + tokenizer.sep_token + "Pain fell with aspirin."),
                      ("query", "Aspirin reduces pain.")]:
    specter.set_active_adapters(adapter)
    inputs = tokenizer([text], return_tensors="pt", return_token_type_ids=False).to(c.device)
    with torch.inference_mode():
        output = specter(**inputs).last_hidden_state[:, 0]
    print("SPECTER2", adapter, output.shape, flush=True)
