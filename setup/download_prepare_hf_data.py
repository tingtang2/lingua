# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import shlex
import time
import subprocess
import requests
from huggingface_hub import snapshot_download


def run_command(command, env=None, use_bash=False):
    print(f"Running: {command}")
    if use_bash:
        subprocess.run(["bash", "-o", "pipefail", "-c", command], check=True, env=env)
    else:
        subprocess.run(command, shell=True, check=True, env=env)


def download_dataset(repo_id, local_dir, allow_patterns):
    print(f"Downloading dataset from {repo_id}...")
    max_retries = 5
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                resume_download=True,
                max_workers=16,  # Don't hesitate to increase this number to lower the download time
            )
            break
        except requests.exceptions.ReadTimeout:
            if attempt < max_retries - 1:
                print(f"Timeout occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise
    print(f"Dataset downloaded to {local_dir}")


def parquet_to_jsonl(dataset, work_dir, src_dir, tgt_dir, ntasks=64):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                src_dir,
                file_progress=True,
                doc_progress=True,
                glob_pattern="**/*.parquet",
            ),
            JsonlWriter(
                tgt_dir,
                output_filename=dataset + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=ntasks,
        logging_dir=os.path.join(work_dir, "datatrove"),
    )
    pipeline_exec.run()


def setup_terashuf(work_dir):
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir


def get_terashuf_tmp_dir(src_dir, shuffle_tmp_dir=None):
    if shuffle_tmp_dir is not None:
        tmp_dir = shuffle_tmp_dir
    else:
        tmp_dir = os.environ.get("TMPDIR", os.path.join(src_dir, "terashuf_tmp"))

    tmp_dir = os.path.abspath(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def main(dataset, memory, data_dir, seed=42, nchunks=32, shuffle_tmp_dir=None):
    # Configuration
    repo_id = {
        "fineweb_edu": "HuggingFaceFW/fineweb-edu",
        "fineweb_edu_10bt": "HuggingFaceFW/fineweb-edu",
        "dclm_baseline_1.0": "mlfoundations/dclm-baseline-1.0",
        "dclm_baseline_1.0_10prct": "mlfoundations/dclm-baseline-1.0",
    }[dataset]
    data_dir = os.path.abspath(data_dir)
    src_dir = os.path.join(data_dir, dataset)
    out_dir = f"{src_dir}_shuffled"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = src_dir  # Directory of this Python file
    prefix = f"{dataset}.chunk."
    orig_extension = {
        "fineweb_edu": ".jsonl",
        "fineweb_edu_10bt": ".jsonl",
        "dclm_baseline_1.0": ".jsonl.zst",
        "dclm_baseline_1.0_10prct": ".jsonl.zst",
    }[dataset]
    reader_command = {
        "fineweb_edu": 'cat "$1"',
        "fineweb_edu_10bt": 'cat "$1"',
        "dclm_baseline_1.0": 'zstdcat "$1" && echo',
        "dclm_baseline_1.0_10prct": 'zstdcat "$1" && echo',
    }[dataset]
    allow_patterns = {
        "fineweb_edu": None,
        "fineweb_edu_10bt": "sample/10BT/*",
        "dclm_baseline_1.0": "*.jsonl.zst",
        "dclm_baseline_1.0_10prct": "global-shard_01_of_10/*.jsonl.zst",
    }[dataset]
    suffix = ".jsonl"
    k_validation = 10000  # Number of lines to take from each chunk for validation

    # Setup terashuf
    terashuf_dir = setup_terashuf(work_dir)

    # Download dataset
    download_dataset(repo_id, src_dir, allow_patterns)

    if "fineweb" in dataset:
        parquet_to_jsonl(dataset, work_dir, src_dir, src_dir)

    shuffle_tmp_dir = get_terashuf_tmp_dir(src_dir, shuffle_tmp_dir)
    print(f"Using terashuf temporary directory: {shuffle_tmp_dir}")

    shuffle_env = os.environ.copy()
    shuffle_env["MEMORY"] = f"{memory}"
    shuffle_env["SEED"] = f"{seed}"
    shuffle_env["TMPDIR"] = shuffle_tmp_dir

    # Run the original shuffling and splitting command
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    run_command(
        f"ulimit -n 100000 && "
        f"find {shlex.quote(src_dir)} -type f -name '*{orig_extension}' -print0 | "
        f"xargs -0 -I {{}} sh -c '{reader_command}' _ {{}} | "
        f"{shlex.quote(terashuf_executable)} | "
        f"split -n r/{nchunks} -d --suffix-length 2 --additional-suffix {shlex.quote(suffix)} - {shlex.quote(os.path.join(out_dir, prefix))}",
        env=shuffle_env,
        use_bash=True,
    )

    # Create validation set and remove lines from chunks
    validation_file = f"{out_dir}/{dataset}.val{suffix}"
    for i in range(nchunks):
        chunk_file = f"{out_dir}/{prefix}{i:02d}{suffix}"
        run_command(f"head -n {k_validation} {chunk_file} >> {validation_file}")
        run_command(f"sed -i '1,{k_validation}d' {chunk_file}")

    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("memory", type=float, default=8)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nchunks", type=int, default=32)
    parser.add_argument(
        "--shuffle_tmp_dir",
        type=str,
        default=None,
        help="Directory used by terashuf for temporary files. Defaults to $TMPDIR or <data_dir>/<dataset>/terashuf_tmp.",
    )

    args = parser.parse_args()

    main(
        args.dataset,
        args.memory,
        args.data_dir,
        args.seed,
        args.nchunks,
        args.shuffle_tmp_dir,
    )
