# python setup/download_prepare_hf_data.py fineweb_edu 8 --data_dir /p/vast1/pretrain/ting_root/lingua_datasets --seed 42 --nchunks 4

python setup/download_prepare_hf_data.py fineweb_edu_10bt 8 --data_dir /p/vast1/pretrain/ting_root/lingua_datasets/fineweb_edu_10bt --seed 42 --nchunks 4

export WANDB_ENTITY=smcleish
export LINGUA_DUMP_DIR=/usr/workspace/mcleish1/loss-spikes-project/lingua/runs
export LINGUA_DATA_ROOT=/p/vast1/pretrain/ting_root/lingua_datasets/fineweb_edu_10bt

torchrun --nproc-per-node 4 -m apps.main.train config=apps/main/configs/debug.yaml steps=100 > out.txt 2>&1