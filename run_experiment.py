"""
Main experiment runner.

Usage examples
──────────────
# Run all methods on MELD test split (constrained prompt, gpt-4o-mini)
python run_experiment.py --dataset meld --split test

# Run with baseline prompt (no grounding constraints)
python run_experiment.py --dataset meld --split test --prompt-mode baseline

# Run only graph_rag with structured serialization
python run_experiment.py --dataset meld --split test --method graph_rag --serial triple

# Ablation: remove same-speaker history
python run_experiment.py --dataset meld --split test --method graph_rag \
    --no-same-speaker --tag ablation_no_same_speaker

# Evaluate faithfulness on saved predictions
python run_experiment.py --dataset meld --split test --eval-only

# Run on EmoryNLP (secondary dataset)
python run_experiment.py --dataset emorynlp --split test
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import CFG
from data_loader import load_dataset, save_dialogues, load_dialogues
from preprocessing import build_instances, filter_valid
from audio_encoder import build_audio_cache
from graph_builder import build_all_graphs, save_graphs, load_graphs
from llm_runner import run_experiment, load_predictions, METHODS
from evaluator import compute_metrics, compare_methods, extract_case_studies, save_metrics, save_case_studies
from grounding_evaluator import evaluate_faithfulness, summarize_faithfulness
from utils import set_seed, ensure_dirs, print_summary, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       default="meld",        choices=["meld", "emorynlp"])
    p.add_argument("--split",         default="test",        choices=["train", "dev", "test"])
    p.add_argument("--method",        default="all",         help="Method name or 'all'")
    p.add_argument("--prompt-mode",   default="constrained", choices=["baseline", "constrained"],
                   help="Prompt mode: 'baseline' (no grounding rules) | 'constrained' (must cite evidence)")
    p.add_argument("--serial",        default="nl",          choices=["nl", "triple"])
    p.add_argument("--tag",           default="",            help="Tag appended to output filenames")
    p.add_argument("--max-instances", default=None,          type=int)
    p.add_argument("--max-samples",   default=500,           type=int,
                   help="Max samples for faithfulness evaluation")
    p.add_argument("--eval-only",     action="store_true")
    p.add_argument("--skip-faithfulness", action="store_true")
    p.add_argument("--no-same-speaker",  action="store_true")
    p.add_argument("--no-emotion-shift", action="store_true")
    p.add_argument("--no-audio",         action="store_true")
    p.add_argument("--no-graph",         action="store_true")
    p.add_argument("--seed",          default=42, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Apply ablation flags
    CFG.serialization_format     = args.serial
    CFG.use_same_speaker_history = not args.no_same_speaker
    CFG.use_emotion_shift        = not args.no_emotion_shift
    CFG.use_audio_feature        = not args.no_audio
    CFG.use_graph                = not args.no_graph

    # Output directory is separated by prompt mode
    mode_suffix = f"_{args.prompt_mode}" if args.prompt_mode == "baseline" else ""
    tag    = f"_{args.tag}" if args.tag else ""
    outdir = Path(CFG.output_dir) / "predictions" / args.dataset / f"{args.split}{mode_suffix}"

    print(f"\nPrompt mode : {args.prompt_mode}")
    print(f"LLM model   : {CFG.llm_model}")
    print(f"Judge model : {CFG.judge_model}")
    print(f"Output dir  : {outdir}")

    # ── 1. Load & process data ────────────────────────────────────────────
    print(f"\n[1] Loading {args.dataset} / {args.split}")
    processed_path = Path(CFG.data_dir) / "processed" / args.dataset / f"{args.split}_dialogues.json"

    if processed_path.exists():
        dialogues = load_dialogues(processed_path)
        print(f"    Loaded {len(dialogues)} dialogues from cache")
    else:
        dialogues = load_dataset(args.dataset, args.split)
        save_dialogues(dialogues, processed_path)

    instances = build_instances(dialogues, args.dataset)
    instances = filter_valid(instances, args.dataset)
    if args.max_instances:
        instances = instances[:args.max_instances]
    print(f"    Instances: {len(instances)}")

    # ── 2. Audio cache ────────────────────────────────────────────────────
    audio_cache = None
    if CFG.use_audio and args.dataset == "meld":
        print("\n[2] Building audio cache")
        cache_path = Path(CFG.audio_cache_dir) / f"{args.dataset}_{args.split}_audio.pkl"
        audio_cache = build_audio_cache(dialogues, cache_path)
    else:
        print("\n[2] Skipping audio (not available or disabled)")

    # ── 3. Build graphs ───────────────────────────────────────────────────
    print("\n[3] Building dialogue graphs")
    graph_path = Path(CFG.data_dir) / "graphs" / args.dataset / f"{args.split}_graphs.pkl"

    if graph_path.exists():
        graphs = load_graphs(graph_path)
        print(f"    Loaded {len(graphs)} graphs from cache")
    else:
        graphs = build_all_graphs(dialogues, instances, audio_cache)
        save_graphs(graphs, graph_path)

    if args.eval_only:
        _run_evaluation(args, instances, outdir, tag)
        return

    # ── 4. Run inference ──────────────────────────────────────────────────
    methods_to_run = METHODS if args.method == "all" else [args.method]
    print(f"\n[4] Running inference: {methods_to_run}")

    for method in methods_to_run:
        print(f"\n  → {method} [{args.prompt_mode}]")
        out_path = outdir / f"{method}{tag}.json"
        if out_path.exists():
            print(f"    Already exists, skipping. Delete to re-run.")
            continue
        run_experiment(
            instances, method, graphs, out_path,
            max_instances=args.max_instances,
            prompt_mode=args.prompt_mode,
        )

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    _run_evaluation(args, instances, outdir, tag)


def _run_evaluation(args, instances, outdir, tag):
    print("\n[5] Evaluating predictions")
    results_by_method = {}
    for method in METHODS:
        path = outdir / f"{method}{tag}.json"
        if path.exists():
            results_by_method[method] = load_predictions(path)

    if not results_by_method:
        print("    No prediction files found.")
        return

    # Prediction metrics
    metrics_dir = Path(CFG.output_dir) / "tables" / args.dataset / outdir.name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for method, preds in results_by_method.items():
        m = compute_metrics(preds, args.dataset)
        print_summary(m, method)
        save_metrics(m, metrics_dir / f"{method}{tag}_metrics.json")

    # Comparison table
    comparison = compare_methods(results_by_method, args.dataset)
    save_json(comparison, metrics_dir / f"comparison{tag}.json")

    if args.skip_faithfulness:
        print("\nDone (faithfulness skipped).")
        return

    # Faithfulness evaluation for all RAG methods
    rag_methods = ["bm25_rag", "dense_rag", "text_only_rag", "graph_rag"]
    grounding_dir = Path(CFG.output_dir) / "grounding" / args.dataset / outdir.name
    grounding_dir.mkdir(parents=True, exist_ok=True)

    for method in rag_methods:
        if method not in results_by_method:
            continue
        judged_path = grounding_dir / f"{method}{tag}_judged.json"
        if judged_path.exists():
            print(f"\n[6] Faithfulness [{method}]: already judged, skipping.")
            continue
        print(f"\n[6] Faithfulness evaluation: {method}")
        preds  = results_by_method[method][:args.max_samples]
        judged = evaluate_faithfulness(preds, instances, out_path=judged_path)
        summary = summarize_faithfulness(judged)
        save_json(summary, grounding_dir / f"{method}{tag}_summary.json")
        faith = (summary.get("faithfulness_score") or {}).get("mean", 0)
        print(f"    faithfulness_score: {faith:.4f}")

    # Case studies
    if "text_only_rag" in results_by_method and "graph_rag" in results_by_method:
        print("\n[7] Extracting case studies")
        cases = extract_case_studies(
            results_by_method["text_only_rag"],
            results_by_method["graph_rag"],
        )
        cs_dir = Path(CFG.output_dir) / "case_studies"
        save_case_studies(cases, cs_dir / f"{args.dataset}{tag}_cases.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
