"""Record direct/transitive package versions and code/checkpoint fingerprints for the final run."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scifact_project.data import sha256, write_json

PACKAGES = ['torch', 'transformers', 'sentence-transformers', 'adapters', 'numpy', 'pandas', 'scipy',
            'scikit-learn', 'matplotlib', 'rank-bm25', 'nbformat', 'nbclient', 'ipykernel', 'pytest',
            'huggingface-hub', 'tokenizers', 'safetensors', 'tqdm', 'joblib', 'threadpoolctl', 'Pillow',
            'requests', 'PyYAML', 'regex', 'jupyter-client', 'jupyter-core', 'traitlets', 'pyzmq']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='outputs/full_gpu_seed42')
    args = parser.parse_args()
    run = Path(args.run)
    # Explicit class names are metadata only; output head order and weights remain unchanged.
    for task, labels in [('selector', ['NOT_EVIDENCE', 'EVIDENCE']), ('stance', ['CONTRADICT', 'SUPPORT'])]:
        path = run / 'checkpoints' / task / 'config.json'
        if path.exists():
            value = json.loads(path.read_text())
            value['id2label'] = {str(i): label for i, label in enumerate(labels)}
            value['label2id'] = {label: i for i, label in enumerate(labels)}
            write_json(path, value)
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    (run / 'requirements-executed.txt').write_text('\n'.join(f'{name}=={v}' for name, v in versions.items()) + '\n')
    sources = ([Path('requirements.txt'), Path('README.md'), Path('scientific_claim_verification.ipynb')] +
               list(Path('scifact_project').glob('*.py')) + list(Path('scripts').glob('*.py')) +
               list(Path('tests').glob('*.py')))
    weights = list((run / 'checkpoints').rglob('*.safetensors')) + list((run / 'checkpoints').rglob('*.joblib'))
    write_json(run / 'artifact_fingerprints.json', {
        'code_sha256': {str(p): sha256(p) for p in sources},
        'checkpoint_sha256': {str(p): sha256(p) for p in weights},
        'note': 'Executed package versions are an environment record; choose appropriate CUDA/CPU PyTorch wheels on another machine.'})
    print(f'Recorded {len(sources)} source hashes and {len(weights)} checkpoint hashes.', flush=True)
