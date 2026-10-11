"""Public model commits used for the delivered experiments (no remote custom code)."""
REVISIONS = {
    "allenai/scibert_scivocab_uncased": "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    "allenai/specter2_base": "3447645e1def9117997203454fa4495937bfbd83",
    "allenai/specter2": "2081559630a80fc5851d8f798a05ba81e9468089",
    "allenai/specter2_adhoc_query": "3f4448817028388648a74349ece07af4518ec5bd",
}


def revision(source):
    return REVISIONS.get(str(source))
