"""Check saved annotations, model negative manifests and exported predictions."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scifact_project.data import read_jsonl, Dataset, gold_docs, normalize_claim, write_json
from scifact_project.config import Config
from scifact_project.metrics import official_metrics, pipeline_metrics, retrieval_metrics


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
    exports = sorted((config.out / 'predictions').glob('official_*.jsonl'))
    assert exports, 'No exported official predictions found'
    for path in exports:
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
    return dataset


def audit_text_overlap(dataset, run):
    """Screen by text alone, then rescore frozen predictions on the unflagged dev claims."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    splits = {'train': dataset.train, 'validation': dataset.validation, 'dev': dataset.dev}
    texts = {name: [normalize_claim(c['claim']) for c in claims] for name, claims in splits.items()}
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
    vectorizer.fit([text for claims in texts.values() for text in claims])
    vectors = {name: vectorizer.transform(claims) for name, claims in texts.items()}
    threshold = 0.90
    pairs = {}
    flagged_dev_ids = set()
    for left, right in [('train', 'validation'), ('train', 'dev'), ('validation', 'dev')]:
        similarity = (vectors[left] @ vectors[right].T).toarray()
        matches = [{'left_id': splits[left][i]['id'], 'right_id': splits[right][j]['id'],
                    'cosine': float(similarity[i, j])} for i, j in zip(*(similarity >= threshold).nonzero())]
        pairs[f'{left}_vs_{right}'] = matches
        if right == 'dev':
            flagged_dev_ids.update(pair['right_id'] for pair in matches)

    retained = [claim for claim in dataset.dev if claim['id'] not in flagged_dev_ids]
    result = {
        'scope': 'Post hoc sensitivity check; original training, model selection and full dev scores retained',
        'method': 'Lowercase and collapse spaces; cosine on L2-normalized character 3–5-gram TF-IDF',
        'threshold': threshold,
        'corpus_quality': {
            'blank_claims': sum(not c['claim'].strip() for claims in (dataset.original_train, dataset.dev, dataset.test)
                                for c in claims),
            'blank_titles': sum(not d['title'].strip() for d in dataset.documents),
            'abstracts_without_sentences': sum(not d['abstract'] for d in dataset.documents),
            'blank_sentences': sum(not s.strip() for d in dataset.documents for s in d['abstract']),
        },
        'pair_counts': {name: len(matches) for name, matches in pairs.items()},
        'pairs': pairs,
        'excluded_dev_ids': sorted(flagged_dev_ids),
        'retained_dev_claims': len(retained),
        'verification': {},
        'retrieval': {},
    }
    for name in ['bm25_predicted', 'hybrid_without_reranking', 'predicted_all']:
        predictions = json.loads((run / 'predictions' / f'pipeline_{name}.json').read_text())
        scores = pipeline_metrics(retained, predictions)
        result['verification'][name] = {'accuracy': scores['accuracy'], 'macro_f1': scores['macro_f1'],
                                         'abstract_rationalized': scores['official']['abstract_rationalized']}
    for name in ['bm25_title_abstract', 'hybrid_reranked']:
        rankings = json.loads((run / 'predictions' / f'retrieval_{name}.json').read_text())['rankings']['dev']
        result['retrieval'][name] = retrieval_metrics(retained, rankings)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='outputs/full_gpu_seed42')
    p.add_argument('--overlap-output', type=Path, help='Write a text overlap and frozen-prediction sensitivity audit.')
    args = p.parse_args()
    run = Path(args.run)
    config = Config(**json.loads((run / 'config.json').read_text()))
    config.output_dir = str(run)
    dataset = audit(config.prepare())
    if args.overlap_output:
        write_json(args.overlap_output, audit_text_overlap(dataset, run))
