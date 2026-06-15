export LLAMA3_TOKENIZER_PATH=/p/vast1/pretrain/ting_root/tokenizers/original/tokenizer.model
export WANDB_ENTITY=tomg-group-umd
export LINGUA_DATA_ROOT=/p/vast1/pretrain/ting_root/lingua_datasets/fineweb_edu_10bt

export LINGUA_DUMP_DIR=$OUTPUT_DIR
export WANDB_DIR=$OUTPUT_DIR

if [ -z "$1" ]; then
    echo "Usage: $0 <config> [overrides...]"
    exit 1
fi

CONFIG="$1"
shift

torchrun \
    --nnodes "$FLUX_JOB_NNODES" \
    --node-rank "$RANK" \
    --nproc-per-node 4 \
    --rdzv-endpoint "$MASTER_ADDR:$MASTER_PORT" \
    -m apps.main.train config="$CONFIG" "$@"
