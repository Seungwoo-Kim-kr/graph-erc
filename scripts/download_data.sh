#!/usr/bin/env bash
# Download MELD and EmoryNLP datasets into data/raw/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── MELD ─────────────────────────────────────────────────────────────────────
MELD_DIR="$ROOT/data/raw/meld"
mkdir -p "$MELD_DIR"

echo "[MELD] Downloading..."
BASE="https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD"
for split in train dev test; do
    fname="${split}_sent_emo.csv"
    if [ ! -f "$MELD_DIR/$fname" ]; then
        curl -fsSL "$BASE/$fname" -o "$MELD_DIR/$fname"
        echo "  Downloaded $fname"
    else
        echo "  $fname already exists, skipping"
    fi
done

# ── EmoryNLP ─────────────────────────────────────────────────────────────────
EMORY_DIR="$ROOT/data/raw/emorynlp"
mkdir -p "$EMORY_DIR"

echo "[EmoryNLP] Downloading..."
BASE="https://raw.githubusercontent.com/emorynlp/emotion-detection/master/data"
for split in train dev test; do
    fname="${split}.json"
    if [ ! -f "$EMORY_DIR/$fname" ]; then
        curl -fsSL "$BASE/$fname" -o "$EMORY_DIR/$fname"
        echo "  Downloaded $fname"
    else
        echo "  $fname already exists, skipping"
    fi
done

echo ""
echo "Done. Data is in data/raw/"
