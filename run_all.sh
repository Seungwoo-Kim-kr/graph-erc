#!/bin/bash
set -e
cd /Users/seungwookim/Desktop/graph_emotion_project
source venv/bin/activate

echo "========================================"
echo " EXPERIMENT START: $(date)"
echo "========================================"

echo ""
echo "[1/3] MELD - Constrained prompt (6 methods, 500 samples)"
python run_experiment.py --dataset meld --split test --max-instances 500 --skip-faithfulness

echo ""
echo "[2/3] MELD - Faithfulness evaluation (constrained)"
python run_faithfulness.py --dataset meld --split test --prompt-mode constrained --max-samples 500

echo ""
echo "[3/3] MELD - Ablation study (constrained)"
python run_ablation.py --dataset meld --split test --max-instances 500 --max-samples 500

echo ""
echo "========================================"
echo " ALL DONE: $(date)"
echo "========================================"
