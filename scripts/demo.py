"""Run a trained SciFact verifier on a new claim, or measure warm inference."""
import argparse
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')


def load_verifier(run, device):
    import torch
    from scifact_project.config import Config
    from scifact_project.data import Corpus, sha256
    from scifact_project.pipeline import ClaimVerifier

    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))

    config = Config(**read(run / 'config.json'))
    config.output_dir = str(run)
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable to this process. Check GPU access or use --device cpu.')
    if device != 'auto':
        config.profile = 'gpu' if device == 'cuda' else 'cpu'
    selection = read(run / 'selection.json')
    embeddings = run / 'cache' / Path(selection['dense_embeddings_path']).name
    for directory in [f"dense_{selection['dense_mode']}", 'selector', 'stance']:
        if not (run / 'checkpoints' / directory / 'model.safetensors').is_file():
            raise FileNotFoundError(f'Missing trained weights for {directory} in {run / "checkpoints"}')
    if not embeddings.is_file():
        raise FileNotFoundError(f'Missing corpus embeddings at {embeddings}')
    selection['dense_embeddings_path'] = str(embeddings)
    corpus_path = ROOT / config.data_dir / 'corpus.jsonl'
    expected_hash = read(run / 'split_manifest.json')['sha256']['corpus.jsonl']
    if sha256(corpus_path) != expected_hash:
        raise ValueError('The corpus differs from the one used to build the saved embeddings.')
    corpus = Corpus(corpus_path, read(embeddings.with_suffix('.json'))['doc_ids'])
    verifier = ClaimVerifier(config.prepare(), corpus, selection)
    if len(corpus.doc_ids) != len(verifier.embeddings):
        raise ValueError('The corpus and saved embeddings have different row counts.')
    return verifier


def validate_claim(claim, verifier):
    tokenizer = verifier.selector.tokenizer
    max_claim_tokens = verifier.config.pair_length - tokenizer.num_special_tokens_to_add(pair=True) - 1
    if not claim.strip():
        raise ValueError('Enter a nonempty scientific claim.')
    if len(tokenizer.tokenize(claim)) > max_claim_tokens:
        raise ValueError(f'Use at most {max_claim_tokens} claim WordPieces so the pair can include evidence.')


def show_prediction(claim, verifier):
    from scifact_project.pipeline import winning_document

    result = verifier.predict(claim, trace=True)
    trace = result['trace']
    print(f'\nInput: {claim}')
    print('BM25 tokens:', ' '.join(trace['bm25_tokens']))
    if trace['encoding']:
        print(f"{trace['pair_head'].capitalize()} context: {trace['pair_text']}")
        preview_length = 24
        wordpieces = trace['wordpieces']
        print('WordPieces:', *wordpieces[:preview_length], '…' if len(wordpieces) > preview_length else '')
        for key, values in trace['encoding'].items():
            print(f"{key} ({len(values)} total, single pair):", *values[:preview_length],
                  '…' if len(values) > preview_length else '')
    print(f"\nVerdict: {result['verdict']}")
    winner = winning_document(result['documents'])
    if winner:
        print(f"Abstract {winner['doc_id']}: {winner['title']}")
        for evidence in winner['evidence']:
            print(f"  [{evidence['index']}] {evidence['text']}")
        for label, score in winner['stance_probabilities'].items():
            print(f'  {label} softmax: {score:.3f}')
        print(f"  Best sentence score: {winner['selection_score']:.3f} · Reranker logit: {winner['retrieval_score']:.3f}")
    else:
        print('No retrieved abstract passed all evidence, stance and relevance gates.')
        for candidate in trace['candidates']:
            evidence = candidate['best_evidence_score']
            stance = candidate['stance_probabilities']
            print(f"  Abstract {candidate['doc_id']} · reranker logit {candidate['document_logit']:.3f}"
                  f" · best sentence {evidence if evidence is None else round(evidence, 3)}"
                  f" · max stance {max(stance) if stance is not None else 'skipped'}")
    print('Model scores are uncalibrated. Evidence is scoped to the SciFact corpus.')


def benchmark(verifier, claims, repeats, load_seconds):
    import torch

    device = verifier.config.device
    use_cuda = device == 'cuda'
    for claim in claims:
        verifier.predict(claim)
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(repeats):
        for claim in claims:
            start = perf_counter()
            verifier.predict(claim)
            if use_cuda:
                torch.cuda.synchronize()
            durations.append(perf_counter() - start)
    loaded = {'dense': verifier.dense, 'reranker': verifier.reranker.model,
              'selector': verifier.selector.model, 'stance': verifier.stance.model}
    return {
        'scope': 'Sequential warm claim-to-verdict and evidence, one claim at a time, without display or token tracing',
        'device': torch.cuda.get_device_name() if use_cuda else 'CPU',
        'torch': torch.__version__,
        'load_seconds': load_seconds,
        'claims': claims,
        'warmup_predictions': len(claims),
        'timed_predictions': len(durations),
        'inference_batch_limit': verifier.config.inference_batch,
        'median_latency_ms': median(durations) * 1000,
        'throughput_claims_per_second': len(durations) / sum(durations),
        'peak_torch_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else None,
        'parameters': {name: sum(p.numel() for p in model.parameters()) for name, model in loaded.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('claims', nargs='*', help='Quote each claim. Omit to enter claims interactively.')
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/full_gpu_seed42')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--benchmark-repeats', type=int, help='Report timings for the provided claims as JSON.')
    args = parser.parse_args()
    if args.benchmark_repeats is not None and (args.benchmark_repeats < 1 or not args.claims):
        parser.error('--benchmark-repeats needs a positive count and at least one quoted claim')
    started = perf_counter()
    try:
        verifier = load_verifier(args.run, args.device)
        for claim in args.claims:
            validate_claim(claim, verifier)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    load_seconds = perf_counter() - started
    if args.benchmark_repeats is not None:
        print(json.dumps(benchmark(verifier, args.claims, args.benchmark_repeats, load_seconds), indent=2))
    elif args.claims:
        for claim in args.claims:
            show_prediction(claim, verifier)
    else:
        print(f'Verifier ready on {verifier.config.device.upper()}. Submit an English claim or a blank line to exit.')
        while True:
            try:
                claim = input('Claim> ')
            except (EOFError, KeyboardInterrupt):
                break
            if not claim.strip():
                break
            try:
                validate_claim(claim, verifier)
            except ValueError as error:
                print(error, file=sys.stderr)
                continue
            show_prediction(claim, verifier)


if __name__ == '__main__':
    main()
