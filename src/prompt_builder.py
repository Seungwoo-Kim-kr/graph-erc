"""
Build LLM prompts for all six methods.

Methods
───────
  llm_only         : target utterance only, no context, no evidence
  full_context_llm : entire preceding dialogue as context, no retrieved evidence
  bm25_rag         : BM25-retrieved utterances as evidence
  dense_rag        : Dense-retrieved utterances as evidence
  text_only_rag    : context window utterances as plain text (no graph)
  graph_rag        : serialized relation-path graph evidence
"""

from typing import List, Optional

from config import CFG


# ── Shared components ─────────────────────────────────────────────────────────

_EMOTION_LIST = {
    "meld":     "anger, disgust, fear, joy, neutral, sadness, surprise",
    "emorynlp": "mad, joyful, peaceful, neutral, sad, scared, powerful, surprised",
}

_SYSTEM_PROMPT = (
    "You are an expert at social emotion reasoning in conversations. "
    "Your task is to predict the emotion of a target utterance and explain "
    "your reasoning using ONLY the provided evidence. "
    "Do NOT use information outside the retrieved evidence."
)

_INSTRUCTION = """\
Given the dialogue context and target utterance, predict the emotion \
of the target speaker.

Emotion labels: {emotion_labels}

IMPORTANT RULES:
1. Your explanation MUST directly quote or cite specific utterances from [Retrieved Evidence].
2. Do NOT introduce information, background knowledge, or assumptions not present in the evidence.
3. If the evidence is insufficient, say so explicitly in your explanation.
4. The "evidence_used" field must list the exact phrases or utterances you relied on.

Respond ONLY in the following JSON format:
{{
  "predicted_emotion": "<one label from the list above>",
  "explanation": "<1-2 sentences that DIRECTLY cite specific content from the retrieved evidence>",
  "evidence_used": ["<exact quote or paraphrase from evidence 1>", "<exact quote or paraphrase from evidence 2>", ...]
}}"""


def _format_context(utts: List[dict]) -> str:
    if not utts:
        return "(no prior context)"
    return "\n".join(f"{u['speaker']}: {u['text']}" for u in utts)


def _format_target(utt: dict) -> str:
    return f"{utt['speaker']}: {utt['text']}"


def _build_prompt(
    dataset: str,
    context_block: str,
    target_block: str,
    evidence_block: str,
) -> dict:
    instruction = _INSTRUCTION.format(emotion_labels=_EMOTION_LIST.get(dataset, ""))
    user_content = (
        f"{instruction}\n\n"
        f"[Dialogue Context]\n{context_block}\n\n"
        f"[Target Utterance]\n{target_block}\n\n"
        f"[Retrieved Evidence]\n{evidence_block}"
    )
    return [
        {"role": "system",  "content": _SYSTEM_PROMPT},
        {"role": "user",    "content": user_content},
    ]


# ── Per-method prompt builders ────────────────────────────────────────────────

def build_llm_only_prompt(instance: dict) -> list:
    return _build_prompt(
        dataset=instance["dataset"],
        context_block="(not provided)",
        target_block=_format_target(instance["target_utterance"]),
        evidence_block="None",
    )


def build_full_context_prompt(instance: dict) -> list:
    return _build_prompt(
        dataset=instance["dataset"],
        context_block=_format_context(instance["full_context_utterances"]),
        target_block=_format_target(instance["target_utterance"]),
        evidence_block="None",
    )


def build_bm25_rag_prompt(instance: dict, retrieved_utts: List[dict]) -> list:
    from retriever import format_retrieved_utts
    return _build_prompt(
        dataset=instance["dataset"],
        context_block=_format_context(instance["context_utterances"]),
        target_block=_format_target(instance["target_utterance"]),
        evidence_block=format_retrieved_utts(retrieved_utts),
    )


def build_dense_rag_prompt(instance: dict, retrieved_utts: List[dict]) -> list:
    from retriever import format_retrieved_utts
    return _build_prompt(
        dataset=instance["dataset"],
        context_block=_format_context(instance["context_utterances"]),
        target_block=_format_target(instance["target_utterance"]),
        evidence_block=format_retrieved_utts(retrieved_utts),
    )


def build_text_only_rag_prompt(instance: dict) -> list:
    return _build_prompt(
        dataset=instance["dataset"],
        context_block=_format_context(instance["context_utterances"]),
        target_block=_format_target(instance["target_utterance"]),
        evidence_block=_format_context(instance["context_utterances"]),
    )


def build_graph_rag_prompt(instance: dict, graph_evidence_text: str) -> list:
    return _build_prompt(
        dataset=instance["dataset"],
        context_block=_format_context(instance["context_utterances"]),
        target_block=_format_target(instance["target_utterance"]),
        evidence_block=graph_evidence_text,
    )


# ── Router ────────────────────────────────────────────────────────────────────

def build_prompt(
    method: str,
    instance: dict,
    retrieved_utts: Optional[List[dict]] = None,
    graph_evidence_text: Optional[str] = None,
) -> list:
    if method == "llm_only":
        return build_llm_only_prompt(instance)
    if method == "full_context_llm":
        return build_full_context_prompt(instance)
    if method == "bm25_rag":
        return build_bm25_rag_prompt(instance, retrieved_utts or [])
    if method == "dense_rag":
        return build_dense_rag_prompt(instance, retrieved_utts or [])
    if method == "text_only_rag":
        return build_text_only_rag_prompt(instance)
    if method == "graph_rag":
        return build_graph_rag_prompt(instance, graph_evidence_text or "None")
    raise ValueError(f"Unknown method: {method}")
