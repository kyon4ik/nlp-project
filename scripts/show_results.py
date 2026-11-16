"""Display saved development-set results without ML dependencies or training."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs/full_gpu_seed42')
    args = parser.parse_args()
    tables = args.run / 'tables'
    try:
        with (tables / 'retrieval_final.csv').open(newline='') as handle:
            retrieval = list(csv.DictReader(handle))
        with (tables / 'end_to_end_final.csv').open(newline='') as handle:
            pipeline = list(csv.DictReader(handle))
    except OSError as error:
        parser.error(str(error))
    print(f'Saved results: {args.run}\nRetrieval (dev only):')
    for row in retrieval:
        if row['split'] == 'dev':
            print(f"  {row['experiment']:28} Recall@10={float(row['Recall@10']):.4f}  MRR@10={float(row['MRR@10']):.4f}")
    print('\nVerification (gold settings are diagnostic upper-information comparisons):')
    for row in pipeline:
        print(f"  {row['setting']:38} claim macro F1={float(row['claim_macro_f1']):.4f}  rationale F1={float(row['official_abstract_rationalized_f1']):.4f}")


if __name__ == '__main__':
    main()
