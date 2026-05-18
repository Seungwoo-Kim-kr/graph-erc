"""
Baseline retrievers: BM25 and Dense (sentence-BERT).

Both retrievers are scoped to utterances BEFORE the target utterance
within the same dialogue (no cross-dialogue retrieval in default mode).
"""

from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import CFG


# ── BM25 ──────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def retrieve_bm25(
    query: str,
    corpus_utts: List[dict],
    top_k: int = None,
) -> List[dict]:
    """
    Retrieve top-k utterances from corpus_utts using BM25.

    Parameters
    ----------
    query       : target utterance text
    corpus_utts : list of utterance dicts (context utterances)
    top_k       : number to retrieve (defaults to CFG.bm25_top_k)
    """
    top_k = top_k or CFG.bm25_top_k
    if not corpus_utts:
        return []

    tokenized = [_tokenize(u["text"]) for u in corpus_utts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(
        zip(corpus_utts, scores), key=lambda x: x[1], reverse=True
    )
    return [utt for utt, _ in ranked[:top_k]]


# ── Dense retrieval ───────────────────────────────────────────────────────────

_dense_model: Optional[SentenceTransformer] = None


def _get_dense_model() -> SentenceTransformer:
    global _dense_model
    if _dense_model is None:
        _dense_model = SentenceTransformer(CFG.text_model)
    return _dense_model


def retrieve_dense(
    query: str,
    corpus_utts: List[dict],
    top_k: int = None,
) -> List[dict]:
    """
    Retrieve top-k utterances using cosine similarity of sentence embeddings.
    """
    top_k = top_k or CFG.dense_top_k
    if not corpus_utts:
        return []

    model = _get_dense_model()
    texts = [u["text"] for u in corpus_utts]
    corpus_embs = model.encode(texts, normalize_embeddings=True)
    query_emb = model.encode([query], normalize_embeddings=True)[0]

    scores = corpus_embs @ query_emb  # cosine sim (both normalized)
    ranked = sorted(
        zip(corpus_utts, scores), key=lambda x: x[1], reverse=True
    )
    return [utt for utt, _ in ranked[:top_k]]


# ── Formatted evidence for prompts ────────────────────────────────────────────

def format_retrieved_utts(retrieved: List[dict]) -> str:
    if not retrieved:
        return "None"
    lines = []
    for utt in retrieved:
        lines.append(f"- {utt['speaker']}: \"{utt['text']}\"")
    return "\n".join(lines)
