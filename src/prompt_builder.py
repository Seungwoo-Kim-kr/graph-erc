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

Prompt modes
────────────
  baseline    : standard prompt (no grounding constraints)
  constrained : evidence-constrained prompt (LLM must cite retrieved evidence)
"""

from typing import List, Optional

from config import CFG


# ── Shared components ─────────────────────────────────────────────────────────

_EMOTION_LIST = {
    "meld":     "anger, disgust, fear, joy, neutral, sadness, surprise",
    "emorynlp": "mad, joyful, peaceful, neutral, sad, scared, powerful, surprised",
}

# ── Baseline prompt (standard, no grounding constraint) ───────────────────────

_SYSTEM_PROMPT_BASELINE = (
    "You are an expert at social emotion reasoning in conversations. "
    "Your task is to predict the emotion of a target utterance and explain "
    "your reasoning based on the provided dialogue context and evidence."
)

_INSTRUCTION_BASELINE = """\
Given the dialogue context and target utterance, predict the emotion \
of the target speaker.

Emotion labels: {emotion_labels}

Respond ONLY in the following JSON format:
{{
  "predicted_emotion": "<one label from the list above>",
  "explanation": "<1-2 sentences explaining your reasoning>",
  "evidence_used": ["<utterance or context you relied on>", ...]
}}"""

# ── Constrained prompt (must cite retrieved evidence) ─────────────────────────

_SYSTEM_PROMPT_CONSTRAINED = (
    "You are an expert at social emotion reasoning in conversations. "
    "Your task is to predict the emotion of a target utterance and explain "
    "your reasoning using ONLY the provided evidence. "
    "Do NOT use information outside the retrieved evidence."
)

_INSTRUCTION_CONSTRAINED = """\
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


def _get_prompts(mode: str):
    """Return (system_prompt, instruction_template) for the given mode."""
    if mode == "constrained":
        return _SYSTEM_PROMPT_CONSTRAINED, _INSTRUCTION_CONSTRAINED
    return _SYSTEM_PROMPT_BASELINE, _INSTRUCTION_BASELINE


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
    mode: str = "constrained",
) -> list:
    system_prompt, instruction_tmpl = _get_prompts(mode)
    instruction = instruction_tmpl.format(emotion_labels=_EMOTION_LIST.get(dataset, ""))
    user_content = (
        "{instruction}\n\n"
        "[Dialogue Context]\n{ctx}\n\n"
        "[Target Utterance]\n{tgt}\n\n"
        "[Retrieved Evidence]\n{evd}"
    ).format(instruction=instruction, ctx=context_block, tgt=target_block, evd=evidence_block)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]


# ── Per-method prompt builders ────────────────────────────────────────────────

def build_llm_only_prompt(instance: dict, mode: str = "constrained") -> list:
    return _build_prompt(
        dataset=instance["dataset"],
        context_block="(not provided)",
        target_block=_format_target(instance["target_utterance"]),
        evidence_block="None",
        mode=mode,
    )


def build_full_context_prompt(instance: dict, mode: str = "constrained") -> list:
    return _build_prompt(
        dataset=instance["dataset"],
        context_block=_format_context(instance["full_context_utterances"]),
        target_block=_format_target(instance["target_utterance"]),
        evidence_block="None",
        mode=mode,
    )


def build_bm25_rag_prompt(instance: dict, retrieved_utts: List[dict], mode: str = "constrained") -> list:
    from retriever import format_retrieved_utts
    return _build_prompt(
        dataset=instance["dataset"],
        context_block=_format_context(instance["context_utterances"]),
        target_block=_format_target(instance["target_utterance"]),
        evidence_block=format_retrieved_utts(retrieved_utts),
        mode=mode,
    )


def build_dense_rag_prompt(instance: dict, retrieved_utts: List[dict], mode: str = "constrained") -> list:
    from retriever import format_retrieved_utts
    return _build_prompt(
        dataset=instance["dataset"],
        context_block=_format_context(instance["context_utterances"]),
        target_block=_format_target(instance["target_utterance"]),
        evidence_block=format_retrieved_utts(retrieved_utts),
        mode=mode,
    )


def build_text_only_rag_prompt(instance: dict, mode: str = "constrained") -> list:
    return _build_prompt(
        dataset=instance["dataset"],
        context_block=_format_context(instance["context_utterances"]),
        target_block=_format_target(instance["target_utterance"]),
        evidence_block=_format_context(instance["context_utterances"]),
        mode=mode,
    )


def build_graph_rag_prompt(instance: dict, graph_evidence_text: str, mode: str = "constrained") -> list:
    return _build_prompt(
        dataset=instance["dataset"],
        context_block=_format_context(instance["context_utterances"]),
        target_block=_format_target(instance["target_utterance"]),
        evidence_block=graph_evidence_text,
        mode=mode,
    )


# ── Router ────────────────────────────────────────────────────────────────────

def build_prompt(
    method: str,
    instance: dict,
    retrieved_utts: Optional[List[dict]] = None,
    graph_evidence_text: Optional[str] = None,
    mode: str = "constrained",
) -> list:
    """
    Build messages list for OpenAI chat API.

    Parameters
    ----------
    method : one of the six retrieval/context methods
    mode   : "baseline" (no grounding rules) | "constrained" (must cite evidence)
    """
    if method == "llm_only":
        return build_llm_only_prompt(instance, mode)
    if method == "full_context_llm":
        return build_full_context_prompt(instance, mode)
    if method == "bm25_rag":
        return build_bm25_rag_prompt(instance, retrieved_utts or [], mode)
    if method == "dense_rag":
        return build_dense_rag_prompt(instance, retrieved_utts or [], mode)
    if method == "text_only_rag":
        return build_text_only_rag_prompt(instance, mode)
    if method == "graph_rag":
        return build_graph_rag_prompt(instance, graph_evidence_text or "None", mode)
    raise ValueError(f"Unknown method: {method}")
