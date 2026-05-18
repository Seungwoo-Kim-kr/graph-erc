"""
Enrich existing prediction files with evidence text (no LLM re-call needed).

For graph_rag: re-builds graph_evidence_text from saved graphs.
For bm25_rag, dense_rag, text_only_rag: re-builds retrieved_text via retrieval.
"""

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
from retriever import retrieve_bm25, retrieve_dense
from graph_retriever import retrieve_graph_evidence
from serializer import serialize_evidence


def enrich(preds, instances, graphs, method):
    inst_map = {
        inst["target_utterance"]["utterance_id"]: inst
        for inst in instances
    }

    for pred in preds:
        uid  = pred["utterance_id"]
        inst = inst_map.get(uid)
        if inst is None:
            continue

        corpus = inst["context_utterances"]
        target_text = inst["target_utterance"]["text"]

        if method == "full_context_llm":
            # Evidence is the full preceding dialogue
            full_utts = inst.get("full_context_utterances", corpus)
            pred["retrieved_text"] = "\n".join(
                f"{u['speaker']}: {u['text']}" for u in full_utts
            )

        elif method == "text_only_rag":
            # Evidence is the context window utterances (same as evidence_block in prompt)
            pred["retrieved_text"] = "\n".join(
                f"{u['speaker']}: {u['text']}" for u in corpus
            )

        elif method == "bm25_rag":
            utts = retrieve_bm25(target_text, corpus)
            pred["retrieved_text"] = "\n".join(
                f"{u['speaker']}: {u['text']}" for u in utts
            )

        elif method == "dense_rag":
            utts = retrieve_dense(target_text, corpus)
            pred["retrieved_text"] = "\n".join(
                f"{u['speaker']}: {u['text']}" for u in utts
            )

        elif method == "graph_rag":
            dlg_id = inst["dialogue_id"]
            tgt_id = uid
            graph_key = f"{dlg_id}|{tgt_id}"
            graph = graphs.get(graph_key) if graphs else None
            if graph is not None:
                evidence = retrieve_graph_evidence(inst, graph)
                pred["graph_evidence_text"] = serialize_evidence(
                    evidence,
                    target_speaker=inst["target_utterance"]["speaker"],
                    graph=graph,
                )
            else:
                pred["graph_evidence_text"] = "Graph not available."

    return preds


def main():
    dataset = "meld"
    split   = "test"

    print("Loading instances and graphs...")
    dlg_path = Path(CFG.data_dir) / "processed" / dataset / f"{split}_dialogues.json"
    dialogues = load_dialogues(dlg_path)
    instances = build_instances(dialogues, dataset)
    instances = filter_valid(instances, dataset)[:100]  # match existing 100-instance run

    graph_path = Path(CFG.data_dir) / "graphs" / dataset / f"{split}_graphs.pkl"
    graphs = load_graphs(graph_path)

    pred_dir = Path(CFG.output_dir) / "predictions" / dataset / split

    for method in ["llm_only", "full_context_llm", "bm25_rag", "dense_rag", "text_only_rag", "graph_rag"]:
        path = pred_dir / f"{method}.json"
        if not path.exists():
            print(f"  Skipping {method}: file not found")
            continue

        with open(path) as f:
            preds = json.load(f)

        preds = enrich(preds, instances, graphs, method)

        with open(path, "w") as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)
        print(f"  Enriched {method}: {len(preds)} predictions saved")


if __name__ == "__main__":
    main()
