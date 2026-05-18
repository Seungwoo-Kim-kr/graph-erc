"""
Build target instances from dialogue-level data.

Each instance represents one prediction task:
  - context_utterances : previous k utterances (text + speaker)
  - target_utterance   : the utterance whose emotion is predicted
  - gold_emotion       : ground-truth label (masked from graph evidence)
"""

import re
from typing import List

from config import CFG


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# ── Label normalization ───────────────────────────────────────────────────────

def normalize_emotion(emotion: str, dataset: str) -> str:
    emotion = emotion.strip().lower()
    if dataset == "emorynlp":
        return CFG.EMORYNLP_TO_UNIFIED.get(emotion, emotion)
    return emotion


# ── Instance builder ──────────────────────────────────────────────────────────

def build_instances(dialogues: List[dict], dataset: str) -> List[dict]:
    """
    For every utterance in every dialogue, create one prediction instance.
    Context window = previous k utterances (index 0 .. t-1).
    """
    instances = []
    for dlg in dialogues:
        utts = dlg["utterances"]
        for t, target_utt in enumerate(utts):
            context_start = max(0, t - CFG.context_window_k)
            context_utts = utts[context_start:t]

            full_context = utts[:t]  # for Full-context LLM baseline

            instances.append({
                "dataset":          dataset,
                "dialogue_id":      dlg["dialogue_id"],
                "target_utterance": {
                    **target_utt,
                    "text": clean_text(target_utt["text"]),
                    "emotion": normalize_emotion(target_utt["emotion"], dataset),
                },
                "context_utterances": [
                    {**u, "text": clean_text(u["text"]),
                     "emotion": normalize_emotion(u["emotion"], dataset)}
                    for u in context_utts
                ],
                "full_context_utterances": [
                    {**u, "text": clean_text(u["text"]),
                     "emotion": normalize_emotion(u["emotion"], dataset)}
                    for u in full_context
                ],
                "gold_emotion": normalize_emotion(target_utt["emotion"], dataset),
                # position of target inside the full dialogue (used by graph builder)
                "target_index": t,
            })
    return instances


def filter_valid(instances: List[dict], dataset: str) -> List[dict]:
    valid_labels = set(CFG.emotion_list(dataset))
    # For cross-dataset unified evaluation, also accept mapped labels
    if dataset == "emorynlp":
        valid_labels |= set(CFG.EMORYNLP_TO_UNIFIED.values())
    return [inst for inst in instances if inst["gold_emotion"] in valid_labels]
