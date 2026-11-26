"""Paper-aligned GPT construction for S5 Lean experiments."""

from __future__ import annotations

from typing import Any, Mapping

from .upstream_runtime import GPT, GPTConfig


def build_gpt_config(model_config: Mapping[str, Any]) -> GPTConfig:
    keep_fraction = model_config.get("activation_keep_fraction")
    if keep_fraction is not None:
        keep_fraction = float(keep_fraction)
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError("activation_keep_fraction must be in (0, 1]")
        if keep_fraction == 1.0:
            keep_fraction = None

    return GPTConfig(
        block_size=int(model_config["block_size"]),
        vocab_size=int(model_config["vocab_size"]),
        n_layer=int(model_config["n_layer"]),
        n_head=int(model_config["n_head"]),
        d_head=int(model_config["d_head"]),
        d_model=int(model_config["d_model"]),
        d_mlp=int(model_config["d_mlp"]),
        d_pos_emb=(
            int(model_config["d_pos_emb"])
            if model_config.get("d_pos_emb") is not None
            else None
        ),
        dropout=float(model_config.get("dropout", 0.0)),
        bias=bool(model_config.get("bias", True)),
        ln_bias=bool(model_config.get("ln_bias", True)),
        rms_norm=bool(model_config.get("rms_norm", True)),
        activation_type=str(model_config.get("activation_type", "gelu")),
        residual_activation_type=str(
            model_config.get("residual_activation_type", "identity")
        ),
        grad_checkpointing=bool(model_config.get("grad_checkpointing", True)),
        enable_bigram_table=bool(model_config.get("enable_bigram_table", True)),
        learnable_bigram_table=bool(
            model_config.get("learnable_bigram_table", True)
        ),
        tied_unembed=bool(model_config.get("tied_unembed", False)),
        sink=bool(model_config.get("sink", True)),
        flash=bool(model_config.get("flash", True)),
        afrac=keep_fraction,
        afrac_loctypes=str(
            model_config.get(
                "activation_sparse_locations",
                "attn_in,attn_out,mlp_in,mlp_out,mlp_neuron,attn_v,attn_k,attn_q",
            )
        ),
    )


def build_model(model_config: Mapping[str, Any]) -> GPT:
    return GPT(build_gpt_config(model_config))


def parameter_counts(model: GPT) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "non_embedding": int(model.get_num_params(non_embedding=True)),
    }
