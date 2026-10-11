"""Official data loading, provenance, grouped splits and annotation-preserving pairs."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re
import tarfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
LABELS = ["CONTRADICT", "SUPPORT"]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_claim(text):
    return " ".join(text.lower().split())


def tokens(text):
    return re.findall(r"\b\w+\b", text.lower())


def document_text(document, title=True):
    abstract = " ".join(document["abstract"])
    return f"{document['title']} {abstract}" if title else abstract


def gold_docs(claim):
    return {int(doc) for doc, rationales in claim.get("evidence", {}).items() if rationales}


def gold_sentences(claim, doc_id):
    return {int(i) for rationale in claim.get("evidence", {}).get(str(doc_id), [])
            for i in rationale["sentences"]}


def claim_label(claim):
    labels = {r["label"] for rs in claim.get("evidence", {}).values() for r in rs}
    if not labels:
        return "NOT_ENOUGH_INFO"
    # Explicit claim aggregation, not an official SciFact label.
    return "CONTRADICT" if "CONTRADICT" in labels else "SUPPORT"


def ensure_data(data_dir):
    data_dir = Path(data_dir)
    if all((data_dir / name).exists() for name in
           ["corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"]):
        return
    root = data_dir.parent
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "data.tar.gz"
    urllib.request.urlretrieve(DATA_URL, archive)
    with tarfile.open(archive) as handle:
        # Extract only expected regular JSONL files; preserve archive's data directory.
        for member in handle.getmembers():
            if member.name in {f"data/{name}" for name in
                               ["corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"]}:
                if not member.isfile():
                    raise ValueError("Unexpected archive member")
                handle.extract(member, root, filter="data")
    if not (data_dir / "corpus.jsonl").exists():
        raise ValueError("data_dir must be the archive's data directory, e.g. data/raw/data")


class Dataset:
    def __init__(self, config):
        ensure_data(config.data_dir)
        root = Path(config.data_dir)
        self.documents = read_jsonl(root / "corpus.jsonl")
        self.corpus = {int(d["doc_id"]): d for d in self.documents}
        self.original_train = read_jsonl(root / "claims_train.jsonl")
        self.dev = read_jsonl(root / "claims_dev.jsonl")
        self.test = read_jsonl(root / "claims_test.jsonl")
        self.validate()
        # Only dev text is used for exact-duplicate hygiene; no dev annotations here.
        dev_texts = {normalize_claim(c["claim"]) for c in self.dev}
        removed = [c["id"] for c in self.original_train if normalize_claim(c["claim"]) in dev_texts]
        eligible = [c for c in self.original_train if c["id"] not in removed]
        self.groups = related_groups(eligible)
        groups = [self.groups[c["id"]] for c in eligible]
        splitter = GroupShuffleSplit(n_splits=1, test_size=config.validation_fraction,
                                     random_state=config.split_seed)
        tr, va = next(splitter.split(eligible, groups=groups))
        self.train = [eligible[i] for i in tr]
        self.validation = [eligible[i] for i in va]
        self.manifest = {
            "source": DATA_URL,
            "sha256": {p.name: sha256(p) for p in root.glob("*.jsonl")},
            "split_seed": config.split_seed,
            "grouping": "Connected components sharing any cited document or identical normalized text",
            "excluded_train_exact_dev_duplicates": removed,
            "train_ids": [c["id"] for c in self.train],
            "validation_ids": [c["id"] for c in self.validation],
            "dev_ids": [c["id"] for c in self.dev],
            "group_count": len(set(groups)),
            "test_unlabeled": True,
        }
        assert not ({self.groups[c["id"]] for c in self.train} &
                    {self.groups[c["id"]] for c in self.validation})
        if config.mode == "smoke":
            # Corpus always contains positives for each smoke split. Metrics are diagnostics only.
            self.train = self.train[:40]
            self.validation = self.validation[:16]
            self.dev = self.dev[:16]
            needed = set().union(*(gold_docs(c) for c in self.train + self.validation + self.dev))
            chosen = needed | {d["doc_id"] for d in self.documents[:128]}
            self.documents = [d for d in self.documents if d["doc_id"] in chosen]
            self.corpus = {d["doc_id"]: d for d in self.documents}
        self.doc_ids = np.array([d["doc_id"] for d in self.documents], dtype=np.int64)
        self.doc_index = {int(doc): i for i, doc in enumerate(self.doc_ids)}
        manifest_path = config.out / 'split_manifest.json'
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            for key in ['sha256', 'split_seed', 'train_ids', 'validation_ids', 'excluded_train_exact_dev_duplicates']:
                if previous.get(key) != self.manifest[key]:
                    raise ValueError(f'Existing run data/split changed ({key}). Use a new output_dir to avoid stale caches.')
        write_json(manifest_path, self.manifest)

    def validate(self):
        assert len(self.documents) == len(self.corpus), "Duplicate document IDs"
        splits = [self.original_train, self.dev, self.test]
        id_sets = [{c["id"] for c in split} for split in splits]
        for i, split in enumerate(splits):
            assert len(id_sets[i]) == len(split), "Duplicate claim IDs"
            for other in id_sets[i + 1:]:
                assert not id_sets[i] & other
            for claim in split:
                for doc, rs in claim.get("evidence", {}).items():
                    assert int(doc) in self.corpus
                    assert len({r["label"] for r in rs}) <= 1
                    for rationale in rs:
                        assert rationale["label"] in LABELS
                        assert rationale["sentences"], "Empty annotated rationale"
                        assert len(set(rationale["sentences"])) == len(rationale["sentences"])
                        assert all(0 <= i < len(self.corpus[int(doc)]["abstract"])
                                   for i in rationale["sentences"])
        assert all("evidence" not in c for c in self.test), "Test unexpectedly labeled"


def related_groups(claims):
    """Union-find by citation metadata; labels are not used for splitting."""
    parent = {c["id"]: c["id"] for c in claims}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for c in claims:
        keys = [("doc", d) for d in c.get("cited_doc_ids", [])]
        keys.append(("text", normalize_claim(c["claim"])))
        for key in keys:
            if key in owner:
                parent[find(c["id"])] = find(owner[key])
            else:
                owner[key] = c["id"]
    return {c["id"]: find(c["id"]) for c in claims}


def eda_tables(dataset):
    rows = []
    for name, claims in [("original_train", dataset.original_train), ("internal_train", dataset.train),
                         ("internal_validation", dataset.validation), ("official_dev", dataset.dev),
                         ("unlabeled_test", dataset.test)]:
        count = Counter(claim_label(c) for c in claims) if name != "unlabeled_test" else {}
        rationale_count = Counter(r["label"] for c in claims for rs in c.get("evidence", {}).values() for r in rs)
        rows.append({"split": name, "claims": len(claims), "evaluable_retrieval_claims":
                     sum(bool(gold_docs(c)) for c in claims) if name != "unlabeled_test" else None,
                     **{label: count.get(label) if name != "unlabeled_test" else None
                        for label in ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]},
                     "support_rationales": rationale_count.get("SUPPORT", 0) if name != "unlabeled_test" else None,
                     "contradict_rationales": rationale_count.get("CONTRADICT", 0) if name != "unlabeled_test" else None})
    docs = pd.DataFrame([{"doc_id": d["doc_id"], "words": len(tokens(document_text(d))),
                          "sentences": len(d["abstract"])} for d in dataset.documents])
    claims = pd.DataFrame([{"split": name, "claim_id": c["id"], "words": len(tokens(c["claim"])),
                            "evidence_documents": len(gold_docs(c))} for name, cs in
                           [("train", dataset.original_train), ("dev", dataset.dev)] for c in cs])
    rationales = pd.DataFrame([{"split": name, "claim_id": c["id"], "doc_id": int(doc),
                                "label": r["label"], "sentences": len(r["sentences"])}
                               for name, cs in [("train", dataset.original_train), ("dev", dataset.dev)]
                               for c in cs for doc, rs in c.get("evidence", {}).items() for r in rs])
    return pd.DataFrame(rows), docs, claims, rationales


def evidence_examples(claims, corpus, include_background=True):
    rows = []
    for c in claims:
        annotated = gold_docs(c)
        # Cited-but-unannotated is only a weak negative; not a stance/NEI ground truth.
        docs = annotated | (set(c.get("cited_doc_ids", [])) if include_background else set())
        for doc in sorted(docs):
            if doc not in corpus:
                continue
            positive = gold_sentences(c, doc)
            for i, text in enumerate(corpus[doc]["abstract"]):
                rows.append({"claim_id": c["id"], "doc_id": doc, "sentence_index": i,
                             "claim": c["claim"], "text": text, "label": int(i in positive),
                             "negative_source": None if i in positive else
                             "annotated_document" if doc in annotated else "unannotated_cited_document"})
    return rows


def stance_examples(claims, corpus):
    rows = []
    for c in claims:
        for doc, rationales in c.get("evidence", {}).items():
            for number, rationale in enumerate(rationales):
                indices = sorted(rationale["sentences"])
                rows.append({"claim_id": c["id"], "doc_id": int(doc), "rationale_id": number,
                             "sentence_indices": indices, "claim": c["claim"],
                             "text": " ".join(corpus[int(doc)]["abstract"][i] for i in indices),
                             "label": LABELS.index(rationale["label"])})
    return rows
