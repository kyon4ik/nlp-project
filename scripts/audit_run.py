"""Check saved annotations, model negative manifests and exported predictions."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scifact_project.data import read_jsonl, Dataset, gold_docs
from scifact_project.config import Config
from scifact_project.experiment import Study
from scifact_project.metrics import official_metrics


def audit(config):
    dataset = Dataset(config)
    train_ids = {c['id'] for c in dataset.train}
    validation_ids = {c['id'] for c in dataset.validation}
    dev_ids = {c['id'] for c in dataset.dev}
    assert not train_ids & validation_ids
    assert not train_ids & dev_ids
    assert not validation_ids & dev_ids
    assert not {dataset.groups[i] for i in train_ids} & {dataset.groups[i] for i in validation_ids}
    training_claims = {c['id']: c for c in dataset.train}
    for name in ['random', 'hard']:
        for r in read_jsonl(config.out / 'tables' / f'retriever_{name}_pairs.jsonl'):
            assert r['claim_id'] in train_ids
            assert set(r['positives']) == gold_docs(training_claims[r['claim_id']])
            assert r['positive'] in r['positives']
            assert set(r['positives']) <= set(r['protected'])
            assert r['negative'] not in r['protected']
    for path in (config.out / 'predictions').glob('official_*.jsonl'):
        rows = read_jsonl(path)
        assert {r['id'] for r in rows} == dev_ids
        assert len(rows) == len(dev_ids)
        for r in rows:
            for doc, prediction in r['evidence'].items():
                assert int(doc) in dataset.corpus
                assert prediction['label'] in {'SUPPORT', 'CONTRADICT'}
                indices = prediction['sentences']
                assert len(set(indices)) == len(indices)
                assert all(0 <= i < len(dataset.corpus[int(doc)]['abstract']) for i in indices)
        computed = official_metrics(dataset.dev, {str(r['id']): r for r in rows})
        saved = config.out / 'metrics/end_to_end.json'
        if saved.exists():
            setting = path.stem.removeprefix('official_')
            assert computed == json.loads(saved.read_text())[setting]['official'], 'Stored official scores disagree with exported predictions'
    print('Audit passed: split isolation, mined negatives, IDs, indices and recomputed official scores.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='outputs/full_gpu_seed42')
    args = p.parse_args()
    config = Config(**json.loads((Path(args.run) / 'config.json').read_text())).prepare()
    audit(config)
