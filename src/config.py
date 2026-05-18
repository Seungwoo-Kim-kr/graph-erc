from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    # ── Datasets ───────────────────────────────────────────────────────────
    datasets: List[str] = field(default_factory=lambda: ["meld", "emorynlp"])
    # IEMOCAP is optional; add "iemocap" to this list when available

    # ── Preprocessing ──────────────────────────────────────────────────────
    context_window_k: int = 3       # previous utterances fed as context
    same_speaker_history_m: int = 2  # same-speaker history utterances

    # ── Audio ──────────────────────────────────────────────────────────────
    use_audio: bool = True
    audio_model: str = "facebook/hubert-base-ls960"
    audio_embedding_dim: int = 768
    audio_cache_dir: str = "data/audio_cache"

    # ── Text embedding ─────────────────────────────────────────────────────
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_embedding_dim: int = 384

    # ── Retrieval ──────────────────────────────────────────────────────────
    bm25_top_k: int = 5
    dense_top_k: int = 5

    # ── LLM inference ──────────────────────────────────────────────────────
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 512
    temperature: float = 0.0
    llm_batch_size: int = 1         # increase if API supports batching

    # ── Evaluation ─────────────────────────────────────────────────────────
    judge_model: str = "gpt-4o-mini"

    # ── Serialization format ───────────────────────────────────────────────
    # "nl" = natural language  |  "triple" = structured triple
    serialization_format: str = "nl"

    # ── Emotion labels ─────────────────────────────────────────────────────
    MELD_EMOTIONS: List[str] = field(default_factory=lambda: [
        "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
    ])

    # EmoryNLP original labels
    EMORYNLP_EMOTIONS: List[str] = field(default_factory=lambda: [
        "mad", "joyful", "peaceful", "neutral", "sad", "scared", "powerful", "surprised"
    ])

    # EmoryNLP → MELD-compatible mapping (used in cross-dataset analysis only)
    EMORYNLP_TO_UNIFIED: dict = field(default_factory=lambda: {
        "mad":       "anger",
        "joyful":    "joy",
        "peaceful":  "joy",
        "powerful":  "joy",
        "neutral":   "neutral",
        "sad":       "sadness",
        "scared":    "fear",
        "surprised": "surprise",
    })

    # ── Ablation flags ─────────────────────────────────────────────────────
    use_same_speaker_history: bool = True
    use_emotion_shift: bool = True
    use_audio_feature: bool = True
    use_graph: bool = True           # False → Text-only RAG mode

    def emotion_list(self, dataset: str) -> List[str]:
        if dataset == "meld":
            return self.MELD_EMOTIONS
        if dataset == "emorynlp":
            return self.EMORYNLP_EMOTIONS
        raise ValueError(f"Unknown dataset: {dataset}")

    def paths(self, dataset: str):
        base = self.data_dir / "raw" / dataset
        return {
            "train": base / ("train_sent_emo.csv" if dataset == "meld" else "train.json"),
            "dev":   base / ("dev_sent_emo.csv"   if dataset == "meld" else "dev.json"),
            "test":  base / ("test_sent_emo.csv"  if dataset == "meld" else "test.json"),
        }


# Singleton used throughout the project
CFG = Config()
