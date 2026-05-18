"""
Load MELD and EmoryNLP into a unified dialogue-level JSON structure.

Unified utterance format:
{
  "utterance_id": str,
  "speaker": str,
  "text": str,
  "emotion": str,          # original label
  "audio_path": str|None
}

Unified dialogue format:
{
  "dataset": str,
  "dialogue_id": str,
  "utterances": [utterance, ...]
}
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import CFG


# ── MELD ──────────────────────────────────────────────────────────────────────

def load_meld_split(csv_path: Path, audio_dir: Optional[Path] = None) -> List[dict]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Sort to guarantee ordering
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"]).reset_index(drop=True)

    dialogues: Dict[int, dict] = {}
    for _, row in df.iterrows():
        d_id = int(row["Dialogue_ID"])
        u_id = int(row["Utterance_ID"])

        if d_id not in dialogues:
            dialogues[d_id] = {
                "dataset": "meld",
                "dialogue_id": f"meld_d{d_id}",
                "utterances": [],
            }

        audio_path = None
        if audio_dir is not None:
            candidate = audio_dir / f"dia{d_id}_utt{u_id}.wav"
            if candidate.exists():
                audio_path = str(candidate)

        dialogues[d_id]["utterances"].append({
            "utterance_id": f"meld_d{d_id}_u{u_id}",
            "speaker": str(row["Speaker"]).strip(),
            "text": str(row["Utterance"]).strip(),
            "emotion": str(row["Emotion"]).strip().lower(),
            "audio_path": audio_path,
        })

    return list(dialogues.values())


def load_meld(split: str, audio_dir: Optional[Path] = None) -> List[dict]:
    paths = CFG.paths("meld")
    return load_meld_split(paths[split], audio_dir)


# ── EmoryNLP ──────────────────────────────────────────────────────────────────

def load_emorynlp_split(json_path: Path) -> List[dict]:
    with open(json_path) as f:
        raw = json.load(f)

    dialogues = []
    # EmoryNLP: {"episodes": [{"scenes": [{"utterances": [...]}]}]}
    for ep_idx, episode in enumerate(raw.get("episodes", [])):
        for sc_idx, scene in enumerate(episode.get("scenes", [])):
            d_id = f"emorynlp_ep{ep_idx}_sc{sc_idx}"
            utterances = []
            for u_idx, utt in enumerate(scene.get("utterances", [])):
                utterances.append({
                    "utterance_id": f"{d_id}_u{u_idx}",
                    "speaker": str(utt.get("speakers", ["Unknown"])[0]).strip(),
                    "text": str(utt.get("transcript", "")).strip(),
                    "emotion": str(utt.get("emotion", "neutral")).strip().lower(),
                    "audio_path": None,  # EmoryNLP has no audio
                })
            if utterances:
                dialogues.append({
                    "dataset": "emorynlp",
                    "dialogue_id": d_id,
                    "utterances": utterances,
                })

    return dialogues


def load_emorynlp(split: str) -> List[dict]:
    paths = CFG.paths("emorynlp")
    return load_emorynlp_split(paths[split])


# ── Unified loader ────────────────────────────────────────────────────────────

def load_dataset(dataset: str, split: str) -> List[dict]:
    if dataset == "meld":
        audio_dir = Path(CFG.data_dir) / "audio" / "meld" if CFG.use_audio else None
        return load_meld(split, audio_dir)
    if dataset == "emorynlp":
        return load_emorynlp(split)
    raise ValueError(f"Unsupported dataset: {dataset}")


def save_dialogues(dialogues: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dialogues, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(dialogues)} dialogues → {out_path}")


def load_dialogues(path: Path) -> List[dict]:
    with open(path) as f:
        return json.load(f)
