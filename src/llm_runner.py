"""
Run LLM inference for all six methods and persist predictions.

Output format per instance:
{
  "dataset":           str,
  "dialogue_id":       str,
  "utterance_id":      str,
  "method":            str,
  "prompt_mode":       str,   # "baseline" | "constrained"
  "gold_emotion":      str,
  "predicted_emotion": str,
  "explanation":       str,
  "evidence_used":     list[str],
  "raw_response":      str,
  "error":             str | None
}
"""

import json
import re
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from config import CFG
from prompt_builder import build_prompt
from retriever import retrieve_bm25, retrieve_dense
from graph_retriever import retrieve_graph_evidence, flatten_evidence
from serializer import serialize_evidence


client = OpenAI()  # reads OPENAI_API_KEY from env


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(messages: list, retries: int = 3, wait: float = 5.0) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=CFG.llm_model,
                messages=messages,
                max_tokens=CFG.max_tokens,
                temperature=CFG.temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(wait * (attempt + 1))
            else:
                raise e


def parse_response(raw: str) -> dict:
    """Extract JSON from the model output (handles markdown code fences)."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        obj = json.loads(cleaned)
        return {
            "predicted_emotion": str(obj.get("predicted_emotion", "")).strip().lower(),
            "explanation":       str(obj.get("explanation", "")).strip(),
            "evidence_used":     obj.get("evidence_used", []),
        }
    except json.JSONDecodeError:
        match = re.search(
            r'"predicted_emotion"\s*:\s*"([^"]+)"', raw, re.IGNORECASE
        )
        emotion = match.group(1).strip().lower() if match else "unknown"
        return {
            "predicted_emotion": emotion,
            "explanation": raw,
            "evidence_used": [],
        }


# ── Per-instance runner ───────────────────────────────────────────────────────

def run_instance(
    instance: dict,
    method: str,
    graphs: Optional[dict] = None,
    prompt_mode: str = "constrained",
) -> dict:
    result = {
        "dataset":             instance["dataset"],
        "dialogue_id":         instance["dialogue_id"],
        "utterance_id":        instance["target_utterance"]["utterance_id"],
        "method":              method,
        "prompt_mode":         prompt_mode,
        "gold_emotion":        instance["gold_emotion"],
        "predicted_emotion":   "unknown",
        "explanation":         "",
        "evidence_used":       [],
        "graph_evidence_text": None,
        "retrieved_text":      None,
        "raw_response":        "",
        "error":               None,
    }

    try:
        retrieved_utts = None
        graph_evidence_text = None

        corpus = instance["context_utterances"]

        if method == "bm25_rag":
            retrieved_utts = retrieve_bm25(
                instance["target_utterance"]["text"], corpus
            )

        elif method in ("dense_rag", "text_only_rag"):
            retrieved_utts = retrieve_dense(
                instance["target_utterance"]["text"], corpus
            )

        elif method == "graph_rag":
            dlg_id = instance["dialogue_id"]
            tgt_id = instance["target_utterance"]["utterance_id"]
            graph_key = f"{dlg_id}|{tgt_id}"
            graph = graphs.get(graph_key) if graphs else None

            if graph is not None:
                evidence = retrieve_graph_evidence(instance, graph)
                graph_evidence_text = serialize_evidence(
                    evidence,
                    target_speaker=instance["target_utterance"]["speaker"],
                    graph=graph,
                )
            else:
                graph_evidence_text = "Graph not available."

        if graph_evidence_text:
            result["graph_evidence_text"] = graph_evidence_text
        if retrieved_utts:
            result["retrieved_text"] = "\n".join(
                f"{u['speaker']}: {u['text']}" for u in retrieved_utts
            )

        messages = build_prompt(
            method=method,
            instance=instance,
            retrieved_utts=retrieved_utts,
            graph_evidence_text=graph_evidence_text,
            mode=prompt_mode,
        )

        raw = call_llm(messages)
        result["raw_response"] = raw
        parsed = parse_response(raw)
        result.update(parsed)

    except Exception as e:
        result["error"] = str(e)

    return result


# ── Batch runner ──────────────────────────────────────────────────────────────

METHODS = [
    "llm_only",
    "full_context_llm",
    "bm25_rag",
    "dense_rag",
    "text_only_rag",
    "graph_rag",
]


def run_experiment(
    instances: List[dict],
    method: str,
    graphs: Optional[dict] = None,
    out_path: Optional[Path] = None,
    max_instances: Optional[int] = None,
    max_workers: int = 8,
    prompt_mode: str = "constrained",
) -> List[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    subset = instances[:max_instances] if max_instances else instances
    results = [None] * len(subset)
    completed = 0

    def _process(idx_inst):
        idx, inst = idx_inst
        return idx, run_instance(inst, method, graphs, prompt_mode)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process, (i, inst)): i
            for i, inst in enumerate(subset)
        }
        for future in as_completed(futures):
            idx, res = future.result()
            results[idx] = res
            completed += 1
            if completed % 50 == 0:
                print(f"  [{method}|{prompt_mode}] {completed}/{len(subset)} done", flush=True)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(results)} predictions → {out_path}")

    return results


def load_predictions(path: Path) -> List[dict]:
    with open(path) as f:
        return json.load(f)
