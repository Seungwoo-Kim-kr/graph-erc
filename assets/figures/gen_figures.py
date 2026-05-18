"""Generate result figures for README."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

# ── 1. Method comparison bar chart ───────────────────────────────────────────

methods   = ["LLM\nOnly", "Full\nContext", "BM25\nRAG", "Dense\nRAG", "Text\nRAG", "Graph\nRAG"]
accuracy  = [0.618, 0.574, 0.526, 0.529, 0.530, 0.591]
macro_f1  = [0.485, 0.491, 0.453, 0.454, 0.462, 0.500]
wtd_f1    = [0.621, 0.586, 0.539, 0.541, 0.542, 0.602]

x = np.arange(len(methods))
w = 0.26

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - w, accuracy, w, label="Accuracy",    color="#4C72B0", alpha=0.9)
bars2 = ax.bar(x,     macro_f1, w, label="Macro-F1",   color="#DD8452", alpha=0.9)
bars3 = ax.bar(x + w, wtd_f1,   w, label="Weighted-F1", color="#55A868", alpha=0.9)

# Highlight Graph-RAG
for bars in [bars1, bars2, bars3]:
    bars[-1].set_edgecolor("black")
    bars[-1].set_linewidth(1.8)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylim(0.38, 0.70)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Method Comparison on MELD Test Set", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# Annotate Graph-RAG bars
for bar in [bars1[-1], bars2[-1], bars3[-1]]:
    ax.annotate(f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

fig.tight_layout()
fig.savefig(OUT / "method_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved method_comparison.png")


# ── 2. Ablation bar chart ─────────────────────────────────────────────────────

abl_labels = ["Full\nGraph-RAG", "w/o Same-\nSpeaker", "w/o Emotion-\nShift", "w/o Audio\nFeatures"]
abl_acc    = [0.591, 0.494, 0.564, 0.574]
abl_f1     = [0.500, 0.438, 0.488, 0.495]

x2 = np.arange(len(abl_labels))

fig, ax = plt.subplots(figsize=(8, 4.5))
bars_a = ax.bar(x2 - 0.18, abl_acc, 0.34, label="Accuracy", color="#4C72B0", alpha=0.9)
bars_b = ax.bar(x2 + 0.18, abl_f1,  0.34, label="Macro-F1", color="#DD8452", alpha=0.9)

bars_a[0].set_edgecolor("black"); bars_a[0].set_linewidth(1.8)
bars_b[0].set_edgecolor("black"); bars_b[0].set_linewidth(1.8)

for bars in [bars_a, bars_b]:
    for bar in bars:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

ax.set_xticks(x2)
ax.set_xticklabels(abl_labels, fontsize=11)
ax.set_ylim(0.35, 0.66)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Ablation Study on MELD Test Set", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT / "ablation_study.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved ablation_study.png")
