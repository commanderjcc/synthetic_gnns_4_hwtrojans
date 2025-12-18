#!/usr/bin/env bash
# Sequential runner: runs matrix_runner.py for combinations of ratios, hidden sizes and models
# Sleeps 2 seconds between runs to allow GPU/OS to settle.

set -u

PYTHON=${PYTHON:-python}
MIN_NODES=200
MAX_NODES=2000
NUM_GRAPHS=8000
VAL_GRAPHS=1000
TEST_GRAPHS=1000
BATCH_SIZE=16
EPOCHS=5
MAX_STEPS=3000
DATASET_MODE=disk
DEVICE=cuda
OUT_DIR=results
PREGENDATA_ROOT=pregen_data
SEED=42

mkdir -p "$OUT_DIR"

RATIOS=(0.25 0.5 0.75)
HIDDENS=(96 48 24)
MODELS=(GCN GAT H2GNN GraphSAGE)

for ratio in "${RATIOS[@]}"; do
  pregen_dir="$PREGENDATA_ROOT/r${ratio}"
  for hidden in "${HIDDENS[@]}"; do
    for model in "${MODELS[@]}"; do
      out_csv="$OUT_DIR/result_${model}_h${hidden}_r${ratio}.csv"
      echo "Running model=${model} hidden=${hidden} ratio=${ratio} -> ${out_csv}"
      uv run matrix_runner.py \
        --min-nodes $MIN_NODES --max-nodes $MAX_NODES \
        --trojan-ratios $ratio \
        --num-graphs $NUM_GRAPHS --val-graphs $VAL_GRAPHS --test-graphs $TEST_GRAPHS \
        --batch-size $BATCH_SIZE --epochs $EPOCHS \
        --max-steps $MAX_STEPS --val-every-steps 50 \
        --dataset-mode $DATASET_MODE --pregen-dir $pregen_dir \
        --device $DEVICE --out-csv $out_csv \
        --model $model --hidden-size $hidden --seed $SEED

      # small pause to let processes terminate and free resources
      sleep 5
    done
  done
done

echo "All runs submitted." 
