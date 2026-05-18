"""
Extract utterance-level audio embeddings using HuBERT.

Audio embeddings are stored as utterance node features in the dialogue graph.
They are NOT used to construct acoustic_similar edges (too noisy / speaker-identity
confounded). Instead they are ablated via config: use_audio_feature=False.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from config import CFG


def _load_model():
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(CFG.audio_model)
    model = HubertModel.from_pretrained(CFG.audio_model)
    model.eval()
    return extractor, model


def extract_audio_embedding(
    audio_path: str,
    extractor,
    model,
    device: str = "cpu",
) -> np.ndarray:
    import librosa
    waveform, sr = librosa.load(audio_path, sr=16_000, mono=True)
    inputs = extractor(waveform, sampling_rate=16_000, return_tensors="pt",
                       padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean-pool over time dimension → (hidden_size,)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    return embedding


def build_audio_cache(
    dialogues,
    cache_path: Path,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Extract and cache HuBERT embeddings for all utterances that have audio.
    Returns dict: utterance_id → embedding (np.ndarray).
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded audio cache ({len(cache)} entries) from {cache_path}")
        return cache

    extractor, model = _load_model()
    model = model.to(device)
    cache: Dict[str, np.ndarray] = {}

    for dlg in dialogues:
        for utt in dlg["utterances"]:
            uid = utt["utterance_id"]
            apath = utt.get("audio_path")
            if apath and Path(apath).exists():
                try:
                    emb = extract_audio_embedding(apath, extractor, model, device)
                    cache[uid] = emb
                except Exception as e:
                    print(f"  [WARN] audio encoding failed for {uid}: {e}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"Audio cache saved: {len(cache)} entries → {cache_path}")
    return cache


def zero_audio_embedding() -> np.ndarray:
    return np.zeros(CFG.audio_embedding_dim, dtype=np.float32)


def get_audio_embedding(
    utterance_id: str,
    audio_cache: Optional[Dict[str, np.ndarray]],
) -> np.ndarray:
    if not CFG.use_audio_feature or audio_cache is None:
        return zero_audio_embedding()
    return audio_cache.get(utterance_id, zero_audio_embedding())
