"""Repeated main retrieval training, with unchanged split and full-profile hyperparameters."""
import argparse
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from scifact_project.config import Config
from scifact_project.experiment import Study


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', default='outputs/full_gpu_seed42')
    parser.add_argument('--seeds', nargs='+', type=int, default=[43, 44])
    args = parser.parse_args()
    reference_dir = Path(args.reference)
    source = Config(**json.loads((reference_dir / 'config.json').read_text())).prepare()
    reference = Study(source)
    tables = [reference.save_retrieval_tables()]
    for seed in sorted(set(args.seeds) - {source.seed}):
        values = source.dictionary() | {'seed': seed, 'output_dir': f'outputs/retriever_repeats_{source.profile}_seed{seed}'}
        study = Study(Config(**values))
        assert study.dataset.manifest['train_ids'] == reference.dataset.manifest['train_ids']
        assert study.dataset.manifest['validation_ids'] == reference.dataset.manifest['validation_ids']
        study.eda()
        for name in ['bm25_title_abstract', 'dense_pretrained']:
            if name in study.rankings:
                continue
            metadata = {k: v for k, v in reference.records[name].items()
                        if k not in {'experiment', 'seed', 'mode', 'device', 'config', 'query_seconds'}}
            metadata.update({'reused_baseline_from': str(reference_dir), 'baseline_timing_not_remeasured': True})
            results = {split: (reference.rankings[name][split], reference.scores[name][split],
                               reference.records[name]['query_seconds'][split]) for split in reference.splits}
            study._store_retrieval(name, results, metadata)
        study.contrastive('random')
        study.contrastive('hard')
        tables.append(study.save_retrieval_tables())
        study._done('retriever_repetition', {'reference': str(reference_dir), 'same_split': True})
    raw = pd.concat(tables)
    raw = raw.loc[(raw['split'] == 'dev') & raw['experiment'].isin(['dense_random', 'dense_hard'])]
    raw.to_csv(reference_dir / 'metrics/retriever_seed_runs.csv', index=False)
    metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'MRR@10']
    rows = []
    for name, group in raw.groupby('experiment'):
        rows.append({'experiment': name, 'seeds': ','.join(str(s) for s in sorted(group['seed'])),
                     'runs': len(group), **{f'{m}_{stat}': group[m].agg(stat)
                                            for m in metrics for stat in ['mean', 'std']}})
    pd.DataFrame(rows).to_csv(reference_dir / 'tables/retriever_seed_summary.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
