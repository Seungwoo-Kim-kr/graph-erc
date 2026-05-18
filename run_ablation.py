"""
Ablation study runner.

Variants:
  1. graph_rag (full)          — baseline (already done)
  2. graph_rag --no-same-speaker
  3. graph_rag --no-emotion-shift
  4. graph_rag --no-audio
  5. text_only_rag (no-graph)  — already done

Usage:
    python run_ablation.py --dataset meld --split test --max-instances 500
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
from graph_builder import load_graphs
from llm_runner import run_experiment, load_predictions
from evaluator import compute_metrics, save_metrics
from grounding_evaluator import evaluate_faithfulness, summarize_faithfulness
from utils import save_json

ABLATION_VARIANTS = [
    {
        "tag":              "no_same_speaker",
        "method":           "graph_rag",
        "no_same_speaker":  True,
        "no_emotion_shift": False,
        "no_audio":         False,
    },
    {
        "tag":              "no_emotion_shift",
        "method":           "graph_rag",
        "no_same_speaker":  False,
        "no_emotion_shift": True,
        "no_audio":         False,
    },
    {
        "tag":              "no_audio",
        "method":           "graph_rag",
        "no_same_speaker":  False,
        "no_emotion_shift": False,
        "no_audio":         True,
    },
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",           default="meld")
    p.add_argument("--split",             default="test")
    p.add_argument("--max-instances",     default=500, type=int)
    p.add_argument("--max-samples",       default=500, type=int)
    p.add_argument("--skip-inference",    action="store_true")
    p.add_argument("--skip-faithfulness", action="store_true")
    args = p.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[1] Loading {args.dataset} / {args.split}")
    dlg_path = Path(CFG.data_dir) / "processed" / args.dataset / f"{args.split}_dialogues.json"
    dialogues = load_dialogues(dlg_path)
    instances = build_instances(dialogues, args.dataset)
    instances = filter_valid(instances, args.dataset)
    print(f"    Instances: {len(instances)}")

    graph_path = Path(CFG.data_dir) / "graphs" / args.dataset / f"{args.split}_graphs.pkl"
    graphs = load_graphs(graph_path)
    print(f"    Graphs: {len(graphs)}")

    outdir        = Path(CFG.output_dir) / "predictions"  / args.dataset / args.split
    metrics_dir   = Path(CFG.output_dir) / "tables"       / args.dataset / args.split
    grounding_dir = Path(CFG.output_dir) / "grounding"    / args.dataset / args.split
    for d in [outdir, metrics_dir, grounding_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Run inference ─────────────────────────────────────────────────────────
    if not args.skip_inference:
        print("\n[2] Running ablation inference...")
        for v in ABLATION_VARIANTS:
            tag    = v["tag"]
            method = v["method"]
            fname  = f"{method}_{tag}.json"
            out_path = outdir / fname

            if out_path.exists():
                print(f"  [SKIP] {fname} already exists")
                continue

            print(f"\n  → {fname} ({args.max_instances} instances)")
            CFG.use_same_speaker_history = not v["no_same_speaker"]
            CFG.use_emotion_shift        = not v["no_emotion_shift"]
            CFG.use_audio_feature        = not v["no_audio"]

            run_experiment(
                instances, method, graphs,
                out_path=out_path,
                max_instances=args.max_instances,
            )

        # Reset flags
        CFG.use_same_speaker_history = True
        CFG.use_emotion_shift        = True
        CFG.use_audio_feature        = True

    # ── Prediction metrics ────────────────────────────────────────────────────
    print("\n[3] Prediction metrics...")
    all_variants = [
        ("graph_rag",                  "graph_rag"),
        ("text_only_rag",              "text_only_rag"),
        ("graph_rag_no_same_speaker",  "graph_rag_no_same_speaker"),
        ("graph_rag_no_emotion_shift", "graph_rag_no_emotion_shift"),
        ("graph_rag_no_audio",         "graph_rag_no_audio"),
    ]

    for label, fname in all_variants:
        path = outdir / f"{fname}.json"
        if not path.exists():
            print(f"  [MISSING] {fname}.json")
            continue
        preds = load_predictions(path)[:args.max_instances]
        m = compute_metrics(preds, args.dataset)
        save_metrics(m, metrics_dir / f"{fname}_metrics.json")
        print(f"  {label:<35} acc={m.get('accuracy',0):.4f}  macro_f1={m.get('macro_f1',0):.4f}")

    # ── Faithfulness evaluation ───────────────────────────────────────────────
    if not args.skip_faithfulness:
        print("\n[4] Faithfulness evaluation (claim-level, parallel)...")
        for label, fname in all_variants:
            pred_path   = outdir       / f"{fname}.json"
            judged_path = grounding_dir / f"{fname}_judged.json"

            if judged_path.exists():
                print(f"  [SKIP] {fname} already judged")
                continue
            if not pred_path.exists():
                continue

            preds = load_predictions(pred_path)[:args.max_samples]
            print(f"  → {fname} ({len(preds)} predictions)...")
            judged  = evaluate_faithfulness(preds, instances, out_path=judged_path)
            summary = summarize_faithfulness(judged)
            save_json(summary, grounding_dir / f"{fname}_summary.json")
            faith = (summary.get("faithfulness_score") or {}).get("mean", 0)
            print(f"    faithfulness_score: {faith:.4f}")

    # ── Final summary table ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  ABLATION RESULTS — " + args.dataset.upper())
    print("="*70)
    print("{:<35} {:>7} {:>7} {:>7}".format("Variant", "Faith", "MacF1", "Acc"))
    print("-"*70)

    for label, fname in all_variants:
        mp = metrics_dir   / f"{fname}_metrics.json"
        sp = grounding_dir / f"{fname}_summary.json"
        m  = json.load(open(mp)) if mp.exists() else {}
        s  = json.load(open(sp)) if sp.exists() else {}
        faith = (s.get("faithfulness_score") or {}).get("mean", 0) or 0
        mf1   = m.get("macro_f1", 0)
        acc   = m.get("accuracy", 0)
        print("{:<35} {:>7.3f} {:>7.4f} {:>7.4f}".format(label, faith, mf1, acc))

    print("="*70)
    print("\nDone.")


if __name__ == "__main__":
    main()
