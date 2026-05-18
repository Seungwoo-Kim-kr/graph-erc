"""
Faithfulness Evaluation Framework (Main Contribution).

Claim-level faithfulness evaluation:
  Step 1 - Decompose: extract atomic claims from the explanation
  Step 2 - Verify: judge each claim against retrieved evidence
  Score per claim: supported=1.0, partially=0.5, unsupported=0.0

Aggregate metrics:
  faithfulness_score  : mean claim support score (0.0 ~ 1.0)
  supported_ratio     : fraction of fully supported claims
  unsupported_ratio   : fraction of unsupported claims
  n_claims            : total number of claims extracted

Judge reliability is validated by comparing two LLM judges (GPT-4o vs GPT-4o-mini)
and reporting Spearman correlation in the paper.
"""

import json
import re
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from config import CFG

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ── Step 1: Claim decomposition ───────────────────────────────────────────────

_DECOMPOSE_SYSTEM = (
    "You are an expert at extracting atomic factual claims from text. "
    "Each claim must be a single, self-contained statement that can be independently verified."
)

_DECOMPOSE_USER = """\
Extract all atomic claims from the following emotion explanation for a dialogue utterance.
Each claim should be a single verifiable statement about the speaker's emotional state or the evidence for it.

[Explanation]
{explanation}

Return ONLY a JSON object:
{{
  "claims": ["<claim 1>", "<claim 2>", ...]
}}

Guidelines:
- Break compound sentences into individual claims
- Each claim must be independently verifiable
- Include 2-6 claims typically
- Keep claims concise (one fact per claim)"""


def decompose_claims(explanation: str, model: str = None, retries: int = 3) -> List[str]:
    model = model or CFG.judge_model
    if not explanation or explanation.strip() == "":
        return []

    for attempt in range(retries):
        try:
            resp = _get_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _DECOMPOSE_SYSTEM},
                    {"role": "user",   "content": _DECOMPOSE_USER.format(explanation=explanation)},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
            obj = json.loads(cleaned)
            claims = obj.get("claims", [])
            return [c for c in claims if isinstance(c, str) and c.strip()]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return []


# ── Step 2: Claim verification ────────────────────────────────────────────────

_VERIFY_SYSTEM = (
    "You are an expert at verifying whether a claim is supported by provided evidence. "
    "Be strict: only mark as supported if the evidence clearly backs the claim."
)

_VERIFY_USER = """\
Determine whether the following claim is supported by the retrieved evidence.

[Dialogue Context]
{context}

[Retrieved Evidence]
{evidence}

[Claim to Verify]
{claim}

Rate the support level:
- "supported"   : the evidence clearly and directly supports this claim
- "partially"   : the evidence partially supports or implies the claim
- "unsupported" : the evidence does not support or contradicts this claim

Return ONLY a JSON object:
{{
  "verdict": "supported" | "partially" | "unsupported",
  "reason": "<one sentence>"
}}"""


_VERDICT_SCORE = {"supported": 1.0, "partially": 0.5, "unsupported": 0.0}


def verify_claim(
    claim: str,
    context_str: str,
    evidence_str: str,
    model: str = None,
    retries: int = 3,
) -> dict:
    model = model or CFG.judge_model

    for attempt in range(retries):
        try:
            resp = _get_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _VERIFY_SYSTEM},
                    {"role": "user",   "content": _VERIFY_USER.format(
                        context=context_str,
                        evidence=evidence_str,
                        claim=claim,
                    )},
                ],
                max_tokens=128,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
            obj = json.loads(cleaned)
            verdict = obj.get("verdict", "unsupported").lower()
            if verdict not in _VERDICT_SCORE:
                verdict = "unsupported"
            return {
                "claim":   claim,
                "verdict": verdict,
                "score":   _VERDICT_SCORE[verdict],
                "reason":  obj.get("reason", ""),
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return {"claim": claim, "verdict": "unsupported", "score": 0.0, "reason": str(e)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(utts: List[dict]) -> str:
    if not utts:
        return "(none)"
    return "\n".join(f"{u['speaker']}: {u['text']}" for u in utts)


def _aggregate(claim_results: List[dict]) -> dict:
    if not claim_results:
        return {
            "faithfulness_score": None,
            "supported_ratio":    None,
            "unsupported_ratio":  None,
            "n_claims":           0,
        }
    scores   = [c["score"] for c in claim_results]
    verdicts = [c["verdict"] for c in claim_results]
    n = len(scores)
    return {
        "faithfulness_score": round(sum(scores) / n, 4),
        "supported_ratio":    round(verdicts.count("supported") / n, 4),
        "unsupported_ratio":  round(verdicts.count("unsupported") / n, 4),
        "n_claims":           n,
    }


# ── Per-prediction evaluation ─────────────────────────────────────────────────

def judge_explanation(
    instance: dict,
    prediction: dict,
    judge_model: str = None,
) -> dict:
    judge_model = judge_model or CFG.judge_model
    context_str  = _format_context(instance.get("context_utterances", []))
    evidence_str = (
        prediction.get("graph_evidence_text")
        or prediction.get("retrieved_text")
        or "None"
    )
    explanation = prediction.get("explanation", "")

    # Step 1: decompose
    claims = decompose_claims(explanation, model=judge_model)

    if not claims:
        return {
            "claim_results":      [],
            "faithfulness_score": None,
            "supported_ratio":    None,
            "unsupported_ratio":  None,
            "n_claims":           0,
            "error":              "no claims extracted",
        }

    # Step 2: verify each claim
    claim_results = [
        verify_claim(claim, context_str, evidence_str, model=judge_model)
        for claim in claims
    ]

    agg = _aggregate(claim_results)
    return {
        "claim_results": claim_results,
        **agg,
        "error": None,
    }


# ── Batch evaluation (parallel) ───────────────────────────────────────────────

def evaluate_faithfulness(
    predictions: List[dict],
    instances: List[dict],
    judge_model: str = None,
    out_path: Optional[Path] = None,
    max_workers: int = 8,
) -> List[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    inst_map = {
        inst["target_utterance"]["utterance_id"]: inst
        for inst in instances
    }

    results = [None] * len(predictions)
    completed = 0

    def _process(idx_pred):
        idx, pred = idx_pred
        uid  = pred["utterance_id"]
        inst = inst_map.get(uid, {})
        scores = judge_explanation(inst, pred, judge_model)
        return idx, {**pred, **{f"judge_{k}": v for k, v in scores.items()}}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process, (i, pred)): i
            for i, pred in enumerate(predictions)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed += 1
            if completed % 20 == 0:
                print(f"  Judged {completed}/{len(predictions)}", flush=True)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Faithfulness results → {out_path}")

    return results


# ── Judge reliability (inter-model agreement) ─────────────────────────────────

def validate_judge_reliability(
    sample_predictions: List[dict],
    sample_instances: List[dict],
    judge_a: str = "gpt-4o",
    judge_b: str = "gpt-4o-mini",
) -> dict:
    """Evaluate with two judges and report Spearman correlation."""
    from scipy.stats import spearmanr

    results_a = evaluate_faithfulness(sample_predictions, sample_instances, judge_a)
    results_b = evaluate_faithfulness(sample_predictions, sample_instances, judge_b)

    correlations = {}
    scores_a = [r.get("judge_faithfulness_score") for r in results_a]
    scores_b = [r.get("judge_faithfulness_score") for r in results_b]
    valid = [(a, b) for a, b in zip(scores_a, scores_b) if a is not None and b is not None]
    if valid:
        rho, p = spearmanr([v[0] for v in valid], [v[1] for v in valid])
        correlations["faithfulness_score"] = {
            "spearman_rho": round(rho, 3),
            "p_value":      round(p, 4),
            "n":            len(valid),
        }
    return correlations


# ── Aggregate summary ─────────────────────────────────────────────────────────

def summarize_faithfulness(judged_results: List[dict]) -> dict:
    import numpy as np

    scores = [
        r["judge_faithfulness_score"]
        for r in judged_results
        if r.get("judge_faithfulness_score") is not None
    ]
    supported = [
        r["judge_supported_ratio"]
        for r in judged_results
        if r.get("judge_supported_ratio") is not None
    ]
    unsupported = [
        r["judge_unsupported_ratio"]
        for r in judged_results
        if r.get("judge_unsupported_ratio") is not None
    ]
    n_claims = [
        r["judge_n_claims"]
        for r in judged_results
        if r.get("judge_n_claims", 0) > 0
    ]

    def _s(arr):
        return {
            "mean": round(float(np.mean(arr)), 4) if arr else None,
            "std":  round(float(np.std(arr)), 4)  if arr else None,
            "n":    len(arr),
        }

    return {
        "faithfulness_score": _s(scores),
        "supported_ratio":    _s(supported),
        "unsupported_ratio":  _s(unsupported),
        "avg_n_claims":       round(float(np.mean(n_claims)), 2) if n_claims else None,
    }
