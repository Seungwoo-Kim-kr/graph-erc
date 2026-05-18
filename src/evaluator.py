"""
Prediction evaluation: Accuracy, Macro F1, Weighted F1, Per-class F1,
Worst-class F1, Confusion Matrix.
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from config import CFG


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(predictions: List[dict], dataset: str) -> dict:
    labels = CFG.emotion_list(dataset)

    gold  = [p["gold_emotion"] for p in predictions]
    preds = [p["predicted_emotion"] for p in predictions]

    # Filter to valid labels only
    valid = [(g, p) for g, p in zip(gold, preds) if g in labels]
    if not valid:
        return {"error": "No valid predictions"}

    gold_v, pred_v = zip(*valid)

    acc      = accuracy_score(gold_v, pred_v)
    macro_f1 = f1_score(gold_v, pred_v, average="macro",    labels=labels, zero_division=0)
    wtd_f1   = f1_score(gold_v, pred_v, average="weighted", labels=labels, zero_division=0)
    per_cls  = f1_score(gold_v, pred_v, average=None,        labels=labels, zero_division=0)
    worst_f1 = float(np.min(per_cls))

    report = classification_report(
        gold_v, pred_v, labels=labels, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(gold_v, pred_v, labels=labels).tolist()

    return {
        "accuracy":        round(acc, 4),
        "macro_f1":        round(macro_f1, 4),
        "weighted_f1":     round(wtd_f1, 4),
        "worst_class_f1":  round(worst_f1, 4),
        "per_class_f1":    {
            label: round(float(f1), 4)
            for label, f1 in zip(labels, per_cls)
        },
        "classification_report": report,
        "confusion_matrix": cm,
        "n_valid": len(valid),
        "n_total": len(predictions),
    }


# ── Multi-method comparison ────────────────────────────────────────────────────

def compare_methods(
    results_by_method: dict,
    dataset: str,
) -> dict:
    """
    results_by_method : { method_name: [prediction dicts] }
    Returns a summary table as dict.
    """
    summary = {}
    for method, preds in results_by_method.items():
        m = compute_metrics(preds, dataset)
        summary[method] = {
            "accuracy":       m.get("accuracy"),
            "macro_f1":       m.get("macro_f1"),
            "weighted_f1":    m.get("weighted_f1"),
            "worst_class_f1": m.get("worst_class_f1"),
        }
    return summary


# ── Case study extractor ──────────────────────────────────────────────────────

def extract_case_studies(
    baseline_preds: List[dict],
    proposed_preds: List[dict],
    n: int = 10,
) -> dict:
    """
    Find cases where baseline fails but proposed succeeds, and vice versa.
    Keyed by utterance_id for alignment.
    """
    base_map     = {p["utterance_id"]: p for p in baseline_preds}
    proposed_map = {p["utterance_id"]: p for p in proposed_preds}

    success_cases  = []  # baseline wrong, proposed right
    failure_cases  = []  # both wrong (proposed fails)
    improve_cases  = []  # both right but proposed has better explanation

    for uid, prop in proposed_map.items():
        base = base_map.get(uid)
        if base is None:
            continue
        gold = prop["gold_emotion"]
        base_correct = base["predicted_emotion"] == gold
        prop_correct = prop["predicted_emotion"] == gold

        entry = {
            "utterance_id":     uid,
            "gold_emotion":     gold,
            "baseline_pred":    base["predicted_emotion"],
            "baseline_expl":    base.get("explanation", ""),
            "proposed_pred":    prop["predicted_emotion"],
            "proposed_expl":    prop.get("explanation", ""),
            "evidence_used":    prop.get("evidence_used", []),
        }

        if not base_correct and prop_correct:
            success_cases.append(entry)
        elif not prop_correct:
            failure_cases.append(entry)
        elif base_correct and prop_correct:
            improve_cases.append(entry)

    return {
        "success_cases":     success_cases[:n],
        "failure_cases":     failure_cases[:n],
        "improvement_cases": improve_cases[:n],
    }


# ── Save / load ───────────────────────────────────────────────────────────────

def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved → {path}")


def save_case_studies(cases: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"Case studies saved → {path}")
