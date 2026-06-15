export LLAMA3_TOKENIZER_PATH=/p/vast1/pretrain/ting_root/tokenizers/original/tokenizer.model
export WANDB_ENTITY=tomg-group-umd
export LINGUA_DATA_ROOT=/p/vast1/pretrain/ting_root/lingua_datasets/fineweb_edu_10bt

export LINGUA_DUMP_DIR=$OUTPUT_DIR
export WANDB_DIR=$OUTPUT_DIR

if [ -z "$1" ]; then
    echo "Usage: $0 <config>"
    exit 1
fi

torchrun --nproc-per-node 4 -m apps.main.train config="$1"
