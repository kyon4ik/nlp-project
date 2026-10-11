"""Extended repeated training; split_seed stays 42 while model seeds vary."""
import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=["cpu", "gpu"], default="gpu")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = p.parse_args()
    tables = []
    for seed in args.seeds:
        subprocess.run([sys.executable, "scripts/run_experiments.py", "--mode", "extended", "--profile", args.profile,
                        "--seed", str(seed)], check=True)
        tables.append(pd.read_csv(f"outputs/extended_{args.profile}_seed{seed}/metrics/retrieval.csv"))
    result = pd.concat(tables)
    metrics = [f"Recall@{k}" for k in [1, 3, 5, 10, 20]] + ["MRR@10", "Hit@10"]
    result.loc[result.split == "dev"].groupby("experiment")[metrics].agg(["mean", "std"]).to_csv(
        "outputs/extended_retrieval_mean_std.csv")
