"""
Deterministic relation-path retrieval from the dialogue graph.

For a target utterance u_t spoken by speaker s_t, we retrieve:
  1. Temporal context path   : u_{t-k} → ... → u_{t-1} → u_t
  2. Same-speaker history    : previous m utterances by s_t
  3. Previous emotion path   : emotion labels of those utterances
  4. Emotion-shift path      : emotion_shift edges among PREVIOUS utterances
  5. Cross-speaker reply path: immediately preceding utterance by a different speaker

Target utterance's gold emotion label is NEVER included (leakage prevention).
"""

from typing import Dict, List, Tuple

import networkx as nx

from config import CFG
from graph_builder import utt_node, speaker_node, emotion_node


# ── Path structs ──────────────────────────────────────────────────────────────

class RelationPath:
    def __init__(self, src: str, rel: str, dst: str):
        self.src = src
        self.rel = rel
        self.dst = dst

    def __repr__(self):
        return f"{self.src} --[{self.rel}]--> {self.dst}"


# ── Core retriever ────────────────────────────────────────────────────────────

def retrieve_graph_evidence(
    instance: dict,
    graph: nx.DiGraph,
) -> Dict[str, List[RelationPath]]:
    """
    Returns a dict of evidence buckets:
      temporal_context, same_speaker_history, emotion_history,
      emotion_shift, reply_context
    """
    target_utt   = instance["target_utterance"]
    tgt_id       = target_utt["utterance_id"]
    tgt_speaker  = target_utt["speaker"]
    tgt_node     = utt_node(tgt_id)
    context_utts = instance["context_utterances"]

    evidence: Dict[str, List[RelationPath]] = {
        "temporal_context":    [],
        "same_speaker_history": [],
        "emotion_history":     [],
        "emotion_shift":       [],
        "reply_context":       [],
    }

    if not CFG.use_graph:
        return evidence

    # ── 1. Temporal context paths ─────────────────────────────────────────
    for utt in context_utts:
        uid = utt_node(utt["utterance_id"])
        if graph.has_edge(uid, tgt_node):
            evidence["temporal_context"].append(
                RelationPath(uid, "temporal_previous", tgt_node)
            )
        else:
            # include as participant even if not direct predecessor
            evidence["temporal_context"].append(
                RelationPath(uid, "precedes", tgt_node)
            )

    # ── 2. Same-speaker history ───────────────────────────────────────────
    if CFG.use_same_speaker_history:
        same_sp_utts = [
            u for u in context_utts if u["speaker"] == tgt_speaker
        ][-CFG.same_speaker_history_m:]

        for utt in same_sp_utts:
            uid = utt_node(utt["utterance_id"])
            evidence["same_speaker_history"].append(
                RelationPath(uid, "same_speaker_prev", tgt_node)
            )

            # ── 3. Emotion history (labels of same-speaker prev utts) ─────
            for _, em_node, data in graph.out_edges(uid, data=True):
                if data.get("rel") == "utterance_emotion":
                    evidence["emotion_history"].append(
                        RelationPath(uid, "has_emotion", em_node)
                    )

    # ── 4. Emotion-shift paths (only among previous utterances) ──────────
    if CFG.use_emotion_shift:
        prev_utt_nodes = {utt_node(u["utterance_id"]) for u in context_utts}
        for src, dst, data in graph.edges(data=True):
            if data.get("rel") == "emotion_shift":
                # Both endpoints must be previous utterances (never target)
                if src in prev_utt_nodes and dst in prev_utt_nodes:
                    evidence["emotion_shift"].append(
                        RelationPath(src, "emotion_shift", dst)
                    )

    # ── 5. Cross-speaker reply context ────────────────────────────────────
    for src, dst, data in graph.in_edges(tgt_node, data=True):
        if data.get("rel") == "reply_context":
            evidence["reply_context"].append(
                RelationPath(src, "reply_context", tgt_node)
            )

    return evidence


def flatten_evidence(evidence: Dict[str, List[RelationPath]]) -> List[RelationPath]:
    paths = []
    for bucket in evidence.values():
        paths.extend(bucket)
    return paths
