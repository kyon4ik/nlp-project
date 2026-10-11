import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scifact_project.config import Config
from scifact_project.experiment import Study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "extended"], default="full")
    parser.add_argument("--profile", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir")
    parser.add_argument("--skip-specter2", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true", help="Recompute final metrics with existing trained models")
    parser.add_argument("--retry-specter2", action="store_true", help="Retry a recorded adapter failure with --stage specter2")
    parser.add_argument("--stage", choices=["all", "retrieval", "supervised", "evaluate", "specter2", "report"], default="all")
    args = parser.parse_args()
    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.skip_specter2:
        overrides["specter2"] = False
    config = Config.for_mode(args.mode, args.profile, args.seed, **overrides)
    study = Study(config)
    if args.stage == "all":
        study.run()
    elif args.stage == "retrieval":
        study.eda(); study.bm25(); study.pretrained()
        study.contrastive("random"); study.contrastive("hard"); study.hybrid(); study.reranking()
    elif args.stage == "supervised":
        study.supervised()
    elif args.stage == "evaluate":
        study.final_evaluation(force=args.force_evaluation)
        study.save_retrieval_tables()
    elif args.stage == "specter2":
        study.specter2(retry=args.retry_specter2)
        study.save_retrieval_tables()
    elif args.stage == "report":
        from scifact_project.report import generate_report
        study.refresh_fusion_costs()
        study.save_retrieval_tables()
        generate_report(study)
