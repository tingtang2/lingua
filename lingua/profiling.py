# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import contextlib
from dataclasses import dataclass
import os
from pathlib import Path
import torch
import torch.distributed
import logging

from lingua.distributed import get_global_rank, get_is_master

import wandb


@dataclass
class ProfilerArgs:
    run: bool = False
    trace_folder: str = "profiling"
    mem_warmup: int = 100
    mem_steps: int = 2
    profile_warmup: int = 102
    profile_steps: int = 2


logger = logging.getLogger()


def perfetto_to_html(json_file, html_file):
    import viztracer
    import gzip
    import string

    root = os.path.dirname(viztracer.__file__)
    sub = {}
    json_file = gzip.open(json_file) if ".gz" in str(json_file) else open(json_file)
    with open(
        os.path.join(root, "html/trace_viewer_embedder.html"), encoding="utf-8"
    ) as f:
        tmpl = f.read()
    with open(os.path.join(root, "html/trace_viewer_full.html"), encoding="utf-8") as f:
        sub["trace_viewer_full"] = f.read()
    with json_file as j:
        content = j.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        sub["json_data"] = content.replace("</script>", "<\\/script>")  # type: ignore
    with open(html_file, "w+", encoding="utf-8") as output_file:
        output_file.write(string.Template(tmpl).substitute(sub))


class TorchProfiler:
    def __init__(self, output_dir: str, config: ProfilerArgs) -> None:
        self.output_dir = Path(output_dir)
        self.config = config
        self.profiler = None

    def __enter__(self):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.config.profile_warmup,
                warmup=0,
                active=self.config.profile_steps,
                repeat=1,
            ),
            on_trace_ready=self._on_trace,
            profile_memory=True,
            record_shapes=True,
            with_stack=False,
            with_flops=True,
        )
        self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.profiler is not None
        return self.profiler.__exit__(exc_type, exc_val, exc_tb)

    def step(self):
        assert self.profiler is not None
        self.profiler.step()

    def _on_trace(self, prof: torch.profiler.profiler.profile) -> None:
        filename = self.output_dir / f"profile_rank{get_global_rank()}.pt.trace.json"
        prof.export_chrome_trace(str(filename))
        if get_is_master() and wandb.run is not None:
            html_path = str(filename).replace(".json", ".html")
            perfetto_to_html(filename, html_path)
            wandb.log({"profile_trace": wandb.Html(html_path)})


@contextlib.contextmanager
def maybe_run_profiler(dump_dir, module, config: ProfilerArgs):
    # get user defined profiler settings

    if config.run:
        trace_dir = os.path.join(dump_dir, config.trace_folder)

        logger.info(f"Profiling active.  Traces will be saved at {trace_dir}")

        if get_is_master() and not os.path.exists(trace_dir):
            os.makedirs(trace_dir)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        with TorchProfiler(trace_dir, config) as profiler:
            yield profiler

    else:
        yield None
