# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
import logging
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)

from lingua.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
    TiedLinear,
    cross_entropy,
)

logger = logging.getLogger(__name__)


def create_causal_mask(seqlen, attn_impl, sliding_window):
    if attn_impl == "sdpa" and sliding_window is not None:
        q_idx = torch.arange(seqlen)[:, None]
        kv_idx = torch.arange(seqlen)[None, :]
        return (q_idx >= kv_idx) & (q_idx - kv_idx < sliding_window)
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


@dataclass
class LMTransformerArgs(BaseTransformerArgs):

    seed: int = 42

    vocab_size: int = -1
    weight_tying: bool = False
    z_loss: bool = False
    mu_centering: bool = False
    vocab_chunk_size: Optional[int] = None

    sliding_window: Optional[int] = None


class LMTransformer(BaseTransformer):
    def __init__(self, args: LMTransformerArgs):
        super().__init__(args)
        self.weight_tying = args.weight_tying
        self.sliding_window = args.sliding_window
        self.z_loss = args.z_loss
        self.mu_centering = args.mu_centering
        self.vocab_chunk_size = args.vocab_chunk_size

        assert args.vocab_size > 0

        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        if args.weight_tying:
            self.output = TiedLinear(self.tok_embeddings)
        else:
            self.output = nn.Linear(
                args.dim,
                args.vocab_size,
                bias=False,
            )

    def output_weight(self):
        return self.tok_embeddings.weight if self.weight_tying else self.output.weight

    def chunked_output_loss(self, hidden, target, return_stats: bool = False):
        weight = self.output_weight()
        if isinstance(weight, DTensor):
            # FSDP shards the LM-head weight across ranks, but each rank still owns a different
            # batch. Gather the full weight and chunk over vocab locally to avoid materializing
            # the full logits tensor while keeping the loss correct.
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            weight = weight.to_local() if world_size == 1 else weight.full_tensor()
            weight = weight.to(dtype=hidden.dtype)

        chunk_size = self.vocab_chunk_size
        if chunk_size is None or chunk_size <= 0:
            raise ValueError(f"vocab_chunk_size must be a positive integer, got {chunk_size}")
        chunk_size = min(chunk_size, weight.shape[0])

        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        labels = target.reshape(-1)
        ignore_index = -100
        valid = labels != ignore_index
        has_valid = bool(valid.any().item())

        log_z = None
        target_logits = None
        total_logit_sum = None
        total_logit_count = 0

        for start in range(0, weight.shape[0], chunk_size):
            end = min(start + chunk_size, weight.shape[0])
            chunk_logits = nn.functional.linear(flat_hidden, weight[start:end]).float()

            if return_stats:
                if total_logit_sum is None:
                    total_logit_sum = chunk_logits.detach().sum(dtype=torch.float64)
                else:
                    total_logit_sum = total_logit_sum + chunk_logits.detach().sum(dtype=torch.float64)
                total_logit_count += chunk_logits.numel()

            chunk_log_z = torch.logsumexp(chunk_logits, dim=-1)
            log_z = chunk_log_z if log_z is None else torch.logaddexp(log_z, chunk_log_z)

            if target_logits is None:
                target_logits = torch.zeros_like(chunk_log_z)

            if has_valid:
                in_chunk = valid & (labels >= start) & (labels < end)
                if bool(in_chunk.any().item()):
                    rows = in_chunk.nonzero(as_tuple=False).squeeze(-1)
                    contribution = torch.zeros_like(chunk_log_z)
                    contribution[rows] = chunk_logits[rows, labels[rows] - start]
                    target_logits = target_logits + contribution

        if log_z is None or target_logits is None:
            raise RuntimeError("Failed to compute chunked vocab loss because no vocab chunks were processed.")

        if not has_valid:
            loss = log_z.sum() * 0.0
        else:
            base_loss = (log_z[valid] - target_logits[valid]).mean()
            if self.z_loss:
                base_loss = base_loss + 1e-4 * log_z[valid].square().mean()
            loss = base_loss

        stats = {}
        if return_stats:
            stats["logits_mean"] = (total_logit_sum / total_logit_count).to(torch.float32)
        return loss, stats

    def apply_mu_centering(self):
        with torch.no_grad():
            weight = self.output_weight()
            if isinstance(weight, DTensor):
                if any(isinstance(p, Shard) and p.dim != 0 for p in weight.placements):
                    raise NotImplementedError(
                        f"mu_centering expects output weights sharded along dim 0, got {weight.placements}"
                    )
                local_weight = weight.to_local()
                local_sum = local_weight.float().sum(dim=0)
                local_count = torch.tensor(
                    local_weight.shape[0],
                    device=local_weight.device,
                    dtype=torch.long,
                )
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(local_sum)
                    torch.distributed.all_reduce(local_count)
                if local_count.item() == 0:
                    return
                global_mean = (local_sum / local_count).to(local_weight.dtype)
                local_weight.sub_(global_mean)
            else:
                mean = weight.float().mean(dim=0).to(weight.dtype)
                weight.sub_(mean)

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
        return_stats: bool = False,
    ):
        bsz, seqlen = token_values.shape

        h = self.tok_embeddings(token_values)

        mask = (
            mask
            if mask is not None
            else create_causal_mask(seqlen, attn_impl, self.sliding_window)
        )

        h = super().forward(h, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)

        h = self.norm(h)
        if target is not None and self.vocab_chunk_size is not None:
            loss, stats = self.chunked_output_loss(h, target, return_stats=return_stats)
            if return_stats:
                return loss, stats
            return loss

        logits = self.output(h)
        if target is not None:
            loss = cross_entropy(logits, target, z_loss=self.z_loss)
            if return_stats:
                return loss, {"logits_mean": logits.float().mean().detach()}
            return loss
        else:
            return logits

    def reset_parameters(self, init_std=None):
        # Either use fixed base std or sqrt model dim
        super().reset_parameters()
        init_std = init_std or (self.dim ** (-0.5))
        self.norm.reset_parameters()
        nn.init.trunc_normal_(
            self.tok_embeddings.weight,
            mean=0.0,
            std=init_std,
            a=-3 * init_std,
            b=3 * init_std,
        )
        if not self.weight_tying:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: LMTransformerArgs):
    group_plan: Tuple[int, bool] = []

    # Grouping and output seperately
    group_plan.append(("tok_embeddings", False))

    # Grouping by layers
    for i in range(model_args.n_layers):
        group_plan.append((f"layers.{i}", False))

    group_plan.append(("output", True))

    return group_plan


# Optional and only used for model/tensor parallelism when tp_size > 1
def tp_parallelize(model, tp_mesh, model_args: LMTransformerArgs, distributed_args):
    assert model_args.dim % distributed_args.tp_size == 0
    assert model_args.vocab_size % distributed_args.tp_size == 0
    assert model_args.n_heads % distributed_args.tp_size == 0
    assert (model_args.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.n_heads % (model_args.n_kv_heads or 1) == 0

    # Embedding layer tp
    main_plan = {}
    main_plan["tok_embeddings"] = ColwiseParallel(
        input_layouts=Replicate(), output_layouts=Shard(1)
    )
    main_plan["norm"] = SequenceParallel()
    main_plan["output"] = ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Replicate()
    )

    parallelize_module(
        model,
        tp_mesh,
        main_plan,
    )

    # Attention layers tp
    for layer in model.layers:
        layer_plan = {}

        layer_plan["attention"] = PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        )
        layer_plan["attention_norm"] = SequenceParallel()
        layer_plan["attention.wq"] = ColwiseParallel()
        layer_plan["attention.wk"] = ColwiseParallel()
        layer_plan["attention.wv"] = ColwiseParallel()
        layer_plan["attention.wo"] = RowwiseParallel(output_layouts=Shard(1))

        # Feedforward layers tp
        layer_plan["feed_forward"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        layer_plan["ffn_norm"] = SequenceParallel()
        layer_plan["feed_forward.w1"] = ColwiseParallel()
        layer_plan["feed_forward.w3"] = ColwiseParallel()
        layer_plan["feed_forward.w2"] = RowwiseParallel(output_layouts=Shard(1))

        parallelize_module(
            layer,
            tp_mesh,
            layer_plan,
        )

        # Adjusting the number of heads and kv heads according to the tp size
        attn_layer = layer.attention
        attn_layer.n_heads = attn_layer.n_heads // distributed_args.tp_size
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // distributed_args.tp_size
