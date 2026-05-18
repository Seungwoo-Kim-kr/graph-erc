"""
Construct a directed dialogue graph for each dialogue.

Node types
──────────
  speaker:<name>
  utt:<utterance_id>
  emotion:<label>

Edge types
──────────
  speaker_utterance   speaker → utt
  utterance_emotion   utt → emotion   [MASKED for target in prediction mode]
  temporal_previous   utt_{t-1} → utt_t
  same_speaker_prev   utt_s_prev → utt_s_cur  (same speaker)
  reply_context       utt_other → utt_target   (cross-speaker preceding)
  emotion_shift       utt_prev → utt_cur       (same speaker, label changed)
                      [NEVER constructed using the target utterance's label]

Leakage prevention rules (enforced in build_graph):
  1. The utterance_emotion edge of the target utterance is never added.
  2. Any emotion_shift edge whose SOURCE or DESTINATION is the target
     utterance is never added (constructing it requires the target's label).
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

from config import CFG


# ── Node / edge helpers ───────────────────────────────────────────────────────

def speaker_node(name: str) -> str:
    return f"speaker:{name}"


def utt_node(uid: str) -> str:
    return f"utt:{uid}"


def emotion_node(label: str) -> str:
    return f"emotion:{label}"


def _add_edge(G: nx.DiGraph, src: str, dst: str, rel: str) -> None:
    G.add_edge(src, dst, rel=rel)


# ── Core graph builder ────────────────────────────────────────────────────────

def build_graph(
    dialogue: dict,
    target_utterance_id: str,
    audio_cache: Optional[Dict] = None,
) -> nx.DiGraph:
    """
    Build a dialogue graph for one dialogue.

    Parameters
    ----------
    dialogue             : unified dialogue dict (from data_loader)
    target_utterance_id  : the utterance being predicted (used for masking)
    audio_cache          : utterance_id → np.ndarray (may be None)

    Returns
    -------
    nx.DiGraph with node attribute 'feat' containing text/audio embeddings
    (populated lazily; downstream code may add embeddings separately).
    """
    G = nx.DiGraph()
    utts = dialogue["utterances"]
    n = len(utts)

    # Index utts by id for quick lookup
    utt_map = {u["utterance_id"]: u for u in utts}
    utt_positions = {u["utterance_id"]: i for i, u in enumerate(utts)}

    # ── 1. Speaker nodes ──────────────────────────────────────────────────
    for utt in utts:
        sp = speaker_node(utt["speaker"])
        if not G.has_node(sp):
            G.add_node(sp, node_type="speaker", name=utt["speaker"])

    # ── 2. Utterance nodes ────────────────────────────────────────────────
    for utt in utts:
        uid = utt_node(utt["utterance_id"])
        G.add_node(
            uid,
            node_type="utterance",
            utterance_id=utt["utterance_id"],
            speaker=utt["speaker"],
            text=utt["text"],
            is_target=(utt["utterance_id"] == target_utterance_id),
        )

    # ── 3. Emotion nodes ──────────────────────────────────────────────────
    unique_emotions = {utt["emotion"] for utt in utts}
    for em in unique_emotions:
        en = emotion_node(em)
        if not G.has_node(en):
            G.add_node(en, node_type="emotion", label=em)

    # ── 4. speaker_utterance edges ────────────────────────────────────────
    for utt in utts:
        _add_edge(G, speaker_node(utt["speaker"]), utt_node(utt["utterance_id"]),
                  "speaker_utterance")

    # ── 5. utterance_emotion edges (masking applied) ──────────────────────
    for utt in utts:
        if utt["utterance_id"] == target_utterance_id:
            continue  # RULE 1: never expose target's gold emotion
        _add_edge(G, utt_node(utt["utterance_id"]), emotion_node(utt["emotion"]),
                  "utterance_emotion")

    # ── 6. temporal_previous edges ────────────────────────────────────────
    for i in range(1, n):
        _add_edge(G,
                  utt_node(utts[i - 1]["utterance_id"]),
                  utt_node(utts[i]["utterance_id"]),
                  "temporal_previous")

    # ── 7. same_speaker_prev edges ────────────────────────────────────────
    if CFG.use_same_speaker_history:
        last_by_speaker: Dict[str, str] = {}
        for utt in utts:
            sp = utt["speaker"]
            uid = utt["utterance_id"]
            if sp in last_by_speaker:
                _add_edge(G, utt_node(last_by_speaker[sp]), utt_node(uid),
                          "same_speaker_prev")
            last_by_speaker[sp] = uid

    # ── 8. reply_context edges ────────────────────────────────────────────
    for i in range(1, n):
        cur = utts[i]
        prev = utts[i - 1]
        if cur["speaker"] != prev["speaker"]:
            _add_edge(G,
                      utt_node(prev["utterance_id"]),
                      utt_node(cur["utterance_id"]),
                      "reply_context")

    # ── 9. emotion_shift edges (LEAKAGE-SAFE) ────────────────────────────
    if CFG.use_emotion_shift:
        prev_emotion_by_speaker: Dict[str, tuple] = {}
        for utt in utts:
            sp = utt["speaker"]
            uid = utt["utterance_id"]
            if sp in prev_emotion_by_speaker:
                prev_uid, prev_em = prev_emotion_by_speaker[sp]
                # RULE 2: skip if either endpoint is the target utterance
                if uid == target_utterance_id or prev_uid == target_utterance_id:
                    pass  # do NOT add this edge
                elif utt["emotion"] != prev_em:
                    _add_edge(G, utt_node(prev_uid), utt_node(uid), "emotion_shift")
            # Only update tracker if this utterance is NOT the target
            # (we must not record the target's emotion label)
            if uid != target_utterance_id:
                prev_emotion_by_speaker[sp] = (uid, utt["emotion"])

    return G


# ── Batch build & persist ─────────────────────────────────────────────────────

def build_all_graphs(
    dialogues: List[dict],
    instances: List[dict],
    audio_cache: Optional[Dict] = None,
) -> Dict[str, nx.DiGraph]:
    """
    Build one graph per (dialogue_id, target_utterance_id) pair.
    Key: f"{dialogue_id}|{target_utterance_id}"
    """
    dlg_map = {d["dialogue_id"]: d for d in dialogues}
    graphs: Dict[str, nx.DiGraph] = {}
    for inst in instances:
        dlg_id = inst["dialogue_id"]
        tgt_id = inst["target_utterance"]["utterance_id"]
        key = f"{dlg_id}|{tgt_id}"
        if key not in graphs:
            graphs[key] = build_graph(dlg_map[dlg_id], tgt_id, audio_cache)
    return graphs


def save_graphs(graphs: Dict[str, nx.DiGraph], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graphs, f)
    print(f"Saved {len(graphs)} graphs → {path}")


def load_graphs(path: Path) -> Dict[str, nx.DiGraph]:
    with open(path, "rb") as f:
        return pickle.load(f)
