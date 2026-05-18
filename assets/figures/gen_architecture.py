"""Generate pipeline architecture diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path

OUT = Path(__file__).parent

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis("off")

def box(ax, x, y, w, h, text, color, fontsize=9.5, textcolor="white", style="round,pad=0.1"):
    b = FancyBboxPatch((x, y), w, h, boxstyle=style,
                       facecolor=color, edgecolor="white", linewidth=1.5, zorder=3)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, fontweight="bold", zorder=4,
            wrap=True, multialignment="center")

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#555555",
                                lw=1.8, mutation_scale=14), zorder=2)

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(6.5, 6.65, "Graph-RAG Pipeline for Emotion Recognition in Conversation",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#222222")

# ── Row 1: Input ─────────────────────────────────────────────────────────────
box(ax, 0.3, 5.2, 2.6, 0.9, "Raw Dialogue\n(MELD / EmoryNLP)", "#546E7A", fontsize=9)

# ── Row 2: Feature extraction ────────────────────────────────────────────────
box(ax, 0.3, 3.6, 1.2, 1.0, "Text\nEncoder\n(SBERT)", "#1565C0", fontsize=8.5)
box(ax, 1.7, 3.6, 1.2, 1.0, "Audio\nEncoder\n(HuBERT)", "#1565C0", fontsize=8.5)

# ── Row 2: Graph builder ──────────────────────────────────────────────────────
box(ax, 3.5, 3.3, 2.8, 1.6,
    "Dialogue Graph Builder\n─────────────────\n"
    "Nodes: speaker · utt · emotion\n"
    "Edges: temporal · same-speaker\n"
    "reply-context · emotion-shift",
    "#2E7D32", fontsize=8.2)

# ── Row 3: Graph retriever ───────────────────────────────────────────────────
box(ax, 3.5, 1.8, 2.8, 1.1, "Graph Retriever\n(relation-path walk)", "#558B2F", fontsize=9)

# ── Row 2: Baselines ─────────────────────────────────────────────────────────
box(ax, 6.8, 4.5, 1.7, 0.75, "BM25\nRetriever",   "#6A1B9A", fontsize=8.5)
box(ax, 8.7, 4.5, 1.7, 0.75, "Dense\nRetriever",  "#6A1B9A", fontsize=8.5)
box(ax, 10.6, 4.5, 1.7, 0.75, "Full\nContext",    "#6A1B9A", fontsize=8.5)

# ── Serializer ───────────────────────────────────────────────────────────────
box(ax, 3.5, 0.7, 2.8, 0.8, "Evidence Serializer  (NL / Triple)", "#00695C", fontsize=8.5)

# ── LLM ──────────────────────────────────────────────────────────────────────
box(ax, 7.5, 1.5, 4.2, 1.5,
    "LLM  (GPT-4o-mini)\n──────────────────\nPrompt = context + evidence\n→ emotion + explanation",
    "#B71C1C", fontsize=8.8)

# ── Output ───────────────────────────────────────────────────────────────────
box(ax, 7.5, 0.3, 4.2, 0.85, "Prediction + Grounded Explanation", "#37474F", fontsize=9)

# ── Arrows ───────────────────────────────────────────────────────────────────
# Raw dialogue → text/audio encoders
arrow(ax, 1.6,  5.2,  0.9,  4.6)
arrow(ax, 1.6,  5.2,  2.3,  4.6)
# Encoders → graph builder
arrow(ax, 0.9,  3.6,  3.5,  4.1)
arrow(ax, 2.3,  3.6,  3.5,  4.1)
# Raw dialogue → baselines
arrow(ax, 2.9,  5.65, 7.65, 5.25)
arrow(ax, 2.9,  5.65, 9.55, 5.25)
arrow(ax, 2.9,  5.65, 11.45,5.25)
# Graph builder → retriever
arrow(ax, 4.9,  3.3,  4.9,  2.9)
# Retriever → serializer
arrow(ax, 4.9,  1.8,  4.9,  1.5)
# Serializer → LLM
arrow(ax, 6.3,  1.1,  7.5,  2.1)
# Baselines → LLM
arrow(ax, 7.65, 4.5,  8.5,  3.0)
arrow(ax, 9.55, 4.5,  9.6,  3.0)
arrow(ax, 11.45,4.5, 10.6,  3.0)
# LLM → output
arrow(ax, 9.6,  1.5,  9.6,  1.15)

# Legend
legend_patches = [
    FancyBboxPatch((0, 0), 1, 0.4, facecolor="#1565C0", label="Feature Extraction"),
    FancyBboxPatch((0, 0), 1, 0.4, facecolor="#2E7D32", label="Graph Components"),
    FancyBboxPatch((0, 0), 1, 0.4, facecolor="#6A1B9A", label="Baseline Retrievers"),
    FancyBboxPatch((0, 0), 1, 0.4, facecolor="#B71C1C", label="LLM Inference"),
]
ax.legend(handles=legend_patches, loc="lower left", fontsize=8.5,
          framealpha=0.9, bbox_to_anchor=(0.0, 0.0))

fig.tight_layout()
fig.savefig(OUT / "architecture.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved architecture.png")
