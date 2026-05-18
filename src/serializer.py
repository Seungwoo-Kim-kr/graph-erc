"""
Convert retrieved relation-path evidence into text for LLM prompts.

Two modes (selected via CFG.serialization_format):
  "nl"     – natural-language sentences
  "triple" – structured subject --[relation]--> object triples
"""

from typing import Dict, List

from graph_retriever import RelationPath
from graph_builder import utt_node, speaker_node, emotion_node


def _shorten(node: str) -> str:
    """Remove node-type prefix for readable output: 'utt:meld_d1_u3' → 'utt_3'."""
    if node.startswith("utt:"):
        raw = node[4:]
        parts = raw.rsplit("_u", 1)
        return f"utt_{parts[-1]}" if len(parts) == 2 else raw
    if node.startswith("speaker:"):
        return node[8:]
    if node.startswith("emotion:"):
        return node[8:]
    return node


# ── Natural-language serializer ───────────────────────────────────────────────

_NL_TEMPLATES = {
    "temporal_previous": "{src} immediately precedes the target utterance.",
    "precedes":          "{src} occurs before the target utterance.",
    "same_speaker_prev": "{src_sp} also spoke at {src}.",
    "has_emotion":       "{src} expressed {dst} emotion.",
    "emotion_shift":     "The speaker's emotion shifted between {src} and {dst}.",
    "reply_context":     "The target utterance responds to {src} (spoken by {src_sp}).",
    "speaker_utterance": "{src} spoke the target utterance.",
}


def _nl_path(path: RelationPath, graph=None) -> str:
    src = _shorten(path.src)
    dst = _shorten(path.dst)
    rel = path.rel

    # Attempt to look up speaker name from graph
    src_sp = src
    if graph is not None and graph.has_node(path.src):
        src_sp = graph.nodes[path.src].get("speaker", src)

    template = _NL_TEMPLATES.get(rel, f"{{src}} --[{rel}]--> {{dst}}.")
    return template.format(src=src, dst=dst, src_sp=src_sp)


def serialize_nl(
    evidence: Dict[str, List[RelationPath]],
    target_speaker: str,
    graph=None,
) -> str:
    lines = [f"- The target utterance was spoken by {target_speaker}."]
    for bucket_name, paths in evidence.items():
        for path in paths:
            line = _nl_path(path, graph)
            if line:
                lines.append(f"- {line}")
    return "\n".join(lines) if lines else "None"


# ── Structured-triple serializer ──────────────────────────────────────────────

def serialize_triple(
    evidence: Dict[str, List[RelationPath]],
    target_speaker: str,
    graph=None,
) -> str:
    lines = [f"speaker:{target_speaker} --[spoke]--> TARGET"]
    for paths in evidence.values():
        for path in paths:
            src = _shorten(path.src)
            dst = _shorten(path.dst)
            lines.append(f"{src} --[{path.rel}]--> {dst}")
    return "\n".join(lines) if lines else "None"


# ── Unified interface ─────────────────────────────────────────────────────────

def serialize_evidence(
    evidence: Dict[str, List[RelationPath]],
    target_speaker: str,
    fmt: str = None,
    graph=None,
) -> str:
    """
    Parameters
    ----------
    fmt : "nl" or "triple". Defaults to CFG.serialization_format.
    """
    from config import CFG
    fmt = fmt or CFG.serialization_format

    if fmt == "triple":
        return serialize_triple(evidence, target_speaker, graph)
    return serialize_nl(evidence, target_speaker, graph)
