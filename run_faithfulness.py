"""
Standalone faithfulness evaluation script.
Does NOT import llm_runner or audio_encoder to avoid hang on model loading.

Usage:
    python run_faithfulness.py --dataset meld --split test
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import CFG
from data_loader import load_dialogues
from preprocessing import build_instances, filter_valid
from grounding_evaluator import evaluate_faithfulness, summarize_faithfulness

METHODS = [
    "llm_only",
    "full_context_llm",
    "bm25_rag",
    "dense_rag",
    "text_only_rag",
    "graph_rag",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default="meld")
    p.add_argument("--split",       default="test")
    p.add_argument("--method",      default="all")
    p.add_argument("--max-samples", default=500, type=int,
                   help="Max predictions per method to evaluate (default: 500)")
    args = p.parse_args()

    # Load instances
    dlg_path = Path(CFG.data_dir) / "processed" / args.dataset / f"{args.split}_dialogues.json"
    dialogues = load_dialogues(dlg_path)
    instances = build_instances(dialogues, args.dataset)
    instances = filter_valid(instances, args.dataset)
    print(f"Instances loaded: {len(instances)}")
    print(f"Evaluating up to {args.max_samples} samples per method")

    pred_dir     = Path(CFG.output_dir) / "predictions" / args.dataset / args.split
    grounding_dir = Path(CFG.output_dir) / "grounding"  / args.dataset / args.split
    grounding_dir.mkdir(parents=True, exist_ok=True)

    methods = METHODS if args.method == "all" else [args.method]

    for method in methods:
        pred_path   = pred_dir     / f"{method}.json"
        judged_path = grounding_dir / f"{method}_judged.json"

        if judged_path.exists():
            print(f"[SKIP] {method}: already evaluated")
            continue
        if not pred_path.exists():
            print(f"[MISSING] {method}: no prediction file")
            continue

        with open(pred_path) as f:
            preds = json.load(f)

        preds = preds[:args.max_samples]
        print(f"\nEvaluating faithfulness for {method} ({len(preds)} predictions)...")
        judged  = evaluate_faithfulness(preds, instances, out_path=judged_path)
        summary = summarize_faithfulness(judged)

        summary_path = grounding_dir / f"{method}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"{method}: {summary}")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
