python setup/download_tokenizer.py llama3 /p/vast1/pretrain/ting_root/tokenizers/

# example path needed
export LLAMA3_TOKENIZER_PATH=/p/vast1/pretrain/ting_root/tokenizers/original/tokenizer.model
export WANDB_ENTITY=tomg-group-umd
export LINGUA_DUMP_DIR=/usr/workspace/mcleish1/loss-spikes-project/lingua/runs/7b_debug
export LINGUA_DATA_ROOT=/p/vast1/pretrain/ting_root/lingua_datasets/fineweb_edu_10bt

torchrun --nproc-per-node 4 -m apps.main.train config=apps/main/configs/base_spike.yaml steps=100 > out.txt 2>&1


python /usr/workspace/mcleish1/llnl-tools/launch_tuo.py \
    --env_act_style=conda_activate \
    --rccl_installdir=/collab/usr/global/tools/rccl/$SYS_TYPE/rocm-6.4.1/install/lib \
    --output_dir=/usr/workspace/mcleish1/loss-spikes-project/lingua/runs \
    --rocm_version=6.4.2 \
    --run_name=7b_loss_spike_1 \
    --nodes=1 \
    --minutes=1440 \
    --repetitions=1 \
    --launch_once_per_node=true \
    --pass_run_name=false \
    --custom_invocation='bash sean_shells/launcher.sh apps/main/configs/base_spike.yaml' --bank=effml
