# Copyright (c) Meta Platforms, Inc. and affiliates.

from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
import math

import logging
from typing import Optional

from torch import nn
from torch.optim import AdamW, SGD, Optimizer, lr_scheduler

logger = logging.getLogger()


@dataclass
class LMHeadOptimArgs:
    type: Optional[str] = None
    lr: Optional[float] = None
    momentum: float = 0.9


@dataclass
class OptimArgs:
    lr: float = 3e-4
    weight_decay: float = 0.1
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.95
    clip: float = 1.0
    lm_head: LMHeadOptimArgs = field(default_factory=LMHeadOptimArgs)

    scheduler: str = "cosine"
    warmup: int = 2000
    lr_min_ratio: float = 0.1
    cycle_length: float = 1.0
    cosine_theta: float = 1.0
    annealing_step: int = 1000
    decay_fraction: float = 0.1

    exp_factor: float = 0.5


def lr_linear(step: int, warmup: int, n_steps: int, min_ratio: float) -> float:
    if step < warmup:
        lr = float(step) / warmup
    elif step <= n_steps:
        s = float(step - warmup) / (n_steps - warmup)
        lr = s * min_ratio + (1 - s)
    else:
        lr = min_ratio
    return lr


def lr_inv_sqrt(step: int, warmup: int, exp_factor: float, min_ratio: float) -> float:
    if step < warmup:
        lr = float(step) / warmup
    else:
        lr = max((warmup**exp_factor) / (step**exp_factor), min_ratio)
    return lr


def lr_cosine(
    step: int,
    warmup: int,
    n_steps: int,
    cycle_length: float,
    theta: float,
    min_ratio: float,
) -> float:
    sign = ((step // (n_steps*cycle_length)) % 2) * -2 + 1
    if step < warmup:
        lr = float(step) / warmup
    elif step <= n_steps:
        s = float(step - warmup) / (n_steps - warmup)
        lr = min_ratio + 0.5 * (1 - min_ratio) * (
            sign * math.cos(math.pi * s**theta / cycle_length) + 1
        )
    else:
        lr = min_ratio
    return lr

def lr_wsd(
    step: int,
    warmup: int,
    n_steps: int,
    decay_fraction: float,
    cycle_length: float,
    min_ratio: float,
) -> float:
    """
    UNDERSTANDING WARMUP-STABLE-DECAY LEARNING RATES: A RIVER VALLEY LOSS LANDSCAPE PERSPECTIVE
    https://arxiv.org/pdf/2410.05192
    """
    cycle_num = step // int(n_steps * cycle_length) + 1
    curr_n_steps = int(n_steps * cycle_length) * cycle_num
    decay_length = int(curr_n_steps * decay_fraction)
    if step == n_steps:
        cycle_num -= 1
        curr_n_steps = n_steps
    
    if step < warmup:
        lr = float(step) / warmup
    elif step <= curr_n_steps - decay_length:
        lr = 1.0
    elif step > curr_n_steps - decay_length and step <= curr_n_steps:
        # Linear interpolation gives similar results
        # slope = -(1.0 - min_ratio) / decay_length
        # intercept = min_ratio + ((1.0 - min_ratio) * curr_n_steps) / decay_length
        # lr = slope * step + intercept

        step_in_decay = step - (curr_n_steps - decay_length)
        progress = step_in_decay / decay_length  
        lr = 1 / (progress * (1/min_ratio) + (1 - progress))
    else:
        lr = min_ratio

    return lr


def build_lr_fn(args: OptimArgs, n_steps: int):
    if args.scheduler == "constant":
        lr_fn = lambda x: 1.0
    elif args.scheduler == "linear":
        lr_fn = partial(
            lr_linear, warmup=args.warmup, n_steps=n_steps, min_ratio=args.lr_min_ratio
        )
    elif args.scheduler == "inv_sqrt":
        lr_fn = partial(
            lr_inv_sqrt,
            warmup=args.warmup,
            exp_factor=args.exp_factor,
            min_ratio=args.lr_min_ratio,
        )
    elif args.scheduler == "cosine":
        lr_fn = partial(
            lr_cosine,
            warmup=args.warmup,
            n_steps=n_steps,
            cycle_length=args.cycle_length,
            theta=args.cosine_theta,
            min_ratio=args.lr_min_ratio,
        )
    elif args.scheduler == "wsd":
        assert args.decay_fraction < args.cycle_length
        lr_fn = partial(
            lr_wsd,
            warmup=args.warmup,
            n_steps=n_steps,
            decay_fraction=args.decay_fraction,
            cycle_length=args.cycle_length,
            min_ratio=args.lr_min_ratio,
        )
    else:
        raise NotImplementedError(f"Unknown scheduler: {args.scheduler}")
    return lr_fn


class MultiOptimizer(Optimizer):
    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        if not self.optimizers:
            raise ValueError("MultiOptimizer requires at least one optimizer")

        all_params = []
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                all_params.extend(group["params"])
        super().__init__(all_params, defaults={})

        # Reuse the underlying param group dicts so schedulers update the real optimizers.
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]
        self.defaults = dict(self.optimizers[0].defaults)
        self._refresh_state()

    def _refresh_state(self):
        merged_state = defaultdict(dict)
        for optimizer in self.optimizers:
            merged_state.update(optimizer.state)
        self.state = merged_state

    def step(self, closure=None):
        loss = None
        for idx, optimizer in enumerate(self.optimizers):
            curr_loss = optimizer.step(closure if idx == 0 else None)
            if loss is None:
                loss = curr_loss
        self._refresh_state()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict):
        optimizer_states = state_dict.get("optimizers")
        if optimizer_states is None:
            raise ValueError("Expected MultiOptimizer state_dict to contain 'optimizers'")
        if len(optimizer_states) != len(self.optimizers):
            raise ValueError(
                f"Expected {len(self.optimizers)} optimizer states, got {len(optimizer_states)}"
            )
        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(optimizer_state)
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]
        self._refresh_state()


def get_optimizer_for_checkpoint(optimizer):
    return optimizer.optimizers if hasattr(optimizer, "optimizers") else optimizer


def get_lm_head_override(args: OptimArgs):
    lm_head_args = getattr(args, "lm_head", None)
    if lm_head_args is None or lm_head_args.type is None:
        return None

    opt_type = lm_head_args.type.lower()
    if opt_type not in {"adamw", "sgd"}:
        raise ValueError(
            f"Unknown LM-head optimizer type: {lm_head_args.type}. Expected 'adamw' or 'sgd'."
        )
    return lm_head_args


def split_named_params_for_lm_head(model: nn.Module):
    base_params = []
    lm_head_params = []
    lm_head_names = []

    for name, param in model.named_parameters():
        if name.startswith("output."):
            lm_head_params.append(param)
            lm_head_names.append(name)
        else:
            base_params.append(param)

    return base_params, lm_head_params, lm_head_names


def build_single_optimizer(params, args: OptimArgs, opt_type: str = "adamw", lr: Optional[float] = None, momentum: Optional[float] = None):
    opt_type = opt_type.lower()
    lr = args.lr if lr is None else lr

    if opt_type == "adamw":
        return AdamW(
            params,
            lr=lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=args.epsilon,
            fused=True,  # Faster optim.step but can throw errors
        )
    if opt_type == "sgd":
        return SGD(
            params,
            lr=lr,
            momentum=args.beta1 if momentum is None else momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unknown optimizer type: {opt_type}")


def build_optimizer(model: nn.Module, args: OptimArgs, n_steps: int):
    logger.info("Starting build of optimizer...")
    lm_head_args = get_lm_head_override(args)
    optimizer = None

    if lm_head_args is not None:
        base_params, lm_head_params, lm_head_names = split_named_params_for_lm_head(model)
        if not lm_head_params:
            logger.warning(
                "LM-head optimizer override requested, but no separate output.* parameters were found. "
                "If weight tying is enabled, the LM head shares embedding weights and cannot be optimized separately."
            )
        else:
            lm_head_lr = args.lr if lm_head_args.lr is None else lm_head_args.lr
            base_optimizer = build_single_optimizer(base_params, args, opt_type="adamw", lr=args.lr)
            lm_head_optimizer = build_single_optimizer(
                lm_head_params,
                args,
                opt_type=lm_head_args.type,
                lr=lm_head_lr,
                momentum=lm_head_args.momentum,
            )
            optimizer = MultiOptimizer([base_optimizer, lm_head_optimizer])
            logger.info(
                "Using a separate LM-head optimizer: type=%s lr=%s momentum=%s params=%d (%s)",
                lm_head_args.type,
                lm_head_lr,
                lm_head_args.momentum,
                sum(param.numel() for param in lm_head_params),
                ", ".join(lm_head_names),
            )

    if optimizer is None:
        optimizer = build_single_optimizer(model.parameters(), args, opt_type="adamw", lr=args.lr)

    # scheduler
    lr_fn = build_lr_fn(args, n_steps)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_fn
    )  # lr_scheduler.LambdaLR(optimizer, lr_fn)

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler
