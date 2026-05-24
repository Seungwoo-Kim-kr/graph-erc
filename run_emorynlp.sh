#!/bin/bash
set -e
cd /Users/seungwookim/Desktop/graph_emotion_project
source venv/bin/activate

echo "========================================"
echo " EmoryNLP EXPERIMENT START: $(date)"
echo "========================================"

echo ""
echo "[1/2] EmoryNLP - Inference (6 methods, 500 samples)"
python run_experiment.py --dataset emorynlp --split test --max-instances 500 --skip-faithfulness

echo ""
echo "[2/2] EmoryNLP - Faithfulness evaluation"
python run_faithfulness.py --dataset emorynlp --split test --prompt-mode constrained --max-samples 500

echo ""
echo "========================================"
echo " ALL DONE: $(date)"
echo "========================================"
