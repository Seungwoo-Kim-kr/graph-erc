# Graph-RAG for Emotion Recognition in Conversation (ERC)

A **graph-structured Retrieval-Augmented Generation** approach to Emotion Recognition in Conversation (ERC). Instead of flat text retrieval, dialogue is modelled as a directed heterogeneous graph—capturing speaker turns, temporal order, same-speaker history, and emotion-shift signals—and serialised evidence is fed to an LLM for emotion prediction with grounded explanations.

---

## Table of Contents

- [Motivation](#motivation)
- [Project Structure](#project-structure)
- [Methods](#methods)
- [Results](#results)
- [Setup](#setup)
- [Usage](#usage)
- [Ablation Study](#ablation-study)
- [Faithfulness Evaluation](#faithfulness-evaluation)
- [Datasets](#datasets)

---

## Motivation

Existing ERC approaches either use the entire dialogue as a flat prompt (expensive, noisy) or rely on keyword-based retrieval that ignores relational structure. Dialogue has rich graph structure—speaker identity, temporal flow, cross-speaker references, and emotion dynamics—that a graph representation can encode explicitly.

This project asks: **can structured graph retrieval improve both accuracy and grounded reasoning over simpler retrieval baselines?**

---

## Project Structure

```
graph_emotion_project/
├── src/
│   ├── config.py            # Centralised configuration (CFG singleton)
│   ├── data_loader.py       # MELD / EmoryNLP dataset loaders
│   ├── preprocessing.py     # Instance builder (context window, filtering)
│   ├── graph_builder.py     # Dialogue graph construction (nodes + edges)
│   ├── graph_retriever.py   # Graph-walk evidence retrieval
│   ├── retriever.py         # BM25 and dense (SBERT) retrieval
│   ├── audio_encoder.py     # HuBERT audio feature extraction
│   ├── prompt_builder.py    # Prompt templates for all 6 methods
│   ├── llm_runner.py        # OpenAI inference + response parsing
│   ├── evaluator.py         # Accuracy / Macro-F1 / per-class F1
│   ├── grounding_evaluator.py  # LLM-as-judge faithfulness scoring
│   ├── serializer.py        # Graph → NL / triple serialisation
│   └── utils.py             # Seeding, directory helpers
│
├── run_experiment.py        # Main experiment runner
├── run_ablation.py          # Ablation study runner
├── run_faithfulness.py      # Faithfulness evaluation runner
├── enrich_predictions.py    # Add evidence text to saved predictions
│
├── notebooks/
│   └── final_experiment.ipynb   # End-to-end walkthrough notebook
│
├── outputs/
│   ├── tables/              # Evaluation metrics (JSON)
│   ├── grounding/           # Faithfulness judge results
│   └── case_studies/        # Qualitative examples
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Methods

Six methods are implemented and compared:

| Method | Description |
|---|---|
| `llm_only` | Target utterance only, no context |
| `full_context_llm` | Entire preceding dialogue as flat context |
| `bm25_rag` | BM25-retrieved utterances as evidence |
| `dense_rag` | Sentence-BERT dense retrieval |
| `text_only_rag` | Context-window utterances (no graph) |
| **`graph_rag`** | **Graph-structured evidence retrieval** (proposed) |

### Dialogue Graph

Each dialogue is encoded as a directed heterogeneous graph:

**Node types**
- `speaker:<name>` — speaker identity node
- `utt:<id>` — utterance node (carries text + optional HuBERT audio embedding)
- `emotion:<label>` — emotion class node (target utterance emotion is masked)

**Edge types**

| Edge | Description |
|---|---|
| `speaker_utterance` | speaker → utterance |
| `temporal_previous` | utt_{t-1} → utt_t |
| `utterance_emotion` | utt → emotion (masked for the target) |
| `same_speaker_prev` | Previous utterance by the same speaker |
| `reply_context` | Cross-speaker preceding utterances |
| `emotion_shift` | Same-speaker pair where emotion changed |

> **Leakage prevention**: the target utterance's emotion node and any emotion-shift edge involving it are never constructed.

---

## Results

Evaluated on the **MELD test set** (7-class emotion classification).

| Method | Accuracy | Macro-F1 | Weighted-F1 |
|---|:---:|:---:|:---:|
| LLM Only | 0.618 | 0.485 | 0.621 |
| Full Context LLM | 0.574 | 0.491 | 0.586 |
| BM25-RAG | 0.526 | 0.453 | 0.539 |
| Dense-RAG | 0.529 | 0.454 | 0.541 |
| Text-Only-RAG | 0.530 | 0.462 | 0.542 |
| **Graph-RAG** | **0.591** | **0.500** | **0.602** |

Graph-RAG achieves the **best Macro-F1 (0.500)** among retrieval-based methods and outperforms all RAG baselines on every metric. It closes ~60% of the gap to the LLM-only upper bound while using only local graph evidence.

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/Seungwoo-Kim-kr/graph-erc.git
cd graph-erc
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# edit .env and set your OpenAI API key
```

### 3. Download datasets

**MELD**
```bash
mkdir -p data/raw/meld
# Download train/dev/test CSVs from https://github.com/declare-lab/MELD
# Place: data/raw/meld/train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv
```

**EmoryNLP**
```bash
mkdir -p data/raw/emorynlp
# Download from https://github.com/emorynlp/emotion-detection
# Place: data/raw/emorynlp/train.json, dev.json, test.json
```

---

## Usage

### Run full experiment (all 6 methods)

```bash
python run_experiment.py --dataset meld --split test
```

### Run a single method

```bash
python run_experiment.py --dataset meld --split test --method graph_rag
```

### Use structured triple serialisation instead of NL

```bash
python run_experiment.py --dataset meld --split test --method graph_rag --serial triple
```

### Evaluate faithfulness only (on saved predictions)

```bash
python run_experiment.py --dataset meld --split test --eval-only
```

### Run on EmoryNLP

```bash
python run_experiment.py --dataset emorynlp --split test
```

---

## Ablation Study

Ablation results on the MELD test set (removing one graph component at a time):

| Variant | Accuracy | Macro-F1 |
|---|:---:|:---:|
| Graph-RAG (full) | 0.591 | 0.500 |
| w/o Same-Speaker History | 0.494 | 0.438 |
| w/o Emotion-Shift Edges | 0.564 | 0.488 |
| w/o Audio Features | 0.574 | 0.495 |

Same-speaker history is the most impactful component (−9.7 pp accuracy when removed).

```bash
# Reproduce
python run_ablation.py --dataset meld --split test --max-instances 500
```

---

## Faithfulness Evaluation

Faithfulness is measured by an LLM-as-judge (GPT-4o-mini) that scores whether each prediction's explanation is grounded in the retrieved evidence rather than hallucinated.

```bash
python run_faithfulness.py --dataset meld --split test
```

Judged results are saved to `outputs/grounding/`.

---

## Datasets

| Dataset | Classes | Train | Dev | Test |
|---|:---:|:---:|:---:|:---:|
| [MELD](https://github.com/declare-lab/MELD) | 7 | 9,989 | 1,109 | 2,610 |
| [EmoryNLP](https://github.com/emorynlp/emotion-detection) | 8 | 7,551 | 954 | 984 |

Raw data is **not included** in this repo. Download links above.

---

## Requirements

- Python ≥ 3.10
- OpenAI API key (GPT-4o-mini used for inference and judging)
- GPU recommended for audio feature extraction (HuBERT), but CPU works

Key dependencies: `openai`, `networkx`, `sentence-transformers`, `transformers`, `torch`, `rank-bm25`, `scikit-learn`

---

## Citation

If you use this code, please cite the datasets:

```bibtex
@inproceedings{poria2019meld,
  title={MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations},
  author={Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Naik, Gautam and Cambria, Erik and Mihalcea, Rada},
  booktitle={ACL},
  year={2019}
}

@inproceedings{zahiri2018emotion,
  title={Emotion Detection on TV Show Transcripts with Sequence-Based Convolutional Neural Networks},
  author={Zahiri, Sayyed M and Choi, Jinho D},
  booktitle={AAAI},
  year={2018}
}
```
