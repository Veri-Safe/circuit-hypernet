from __future__ import annotations

import copy
import math
import types
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


NATIVE_HF_NUMERICS_BACKEND = "qwen3_native_hf_fa2_numerics_v1"
NATIVE_HF_INTEGRATION_BACKEND = "qwen3_native_hf_fa2_integration_v2"
NATIVE_HF_CPU_ROPE_BACKEND = "qwen3_native_hf_fa2_cpu_rope_v3"
NATIVE_HF_REUSED_ROPE_BACKEND = "qwen3_native_hf_fa2_reused_rope_v4"


def _native_input_projection_weight(weight: torch.Tensor) -> torch.Tensor:
    """Restore HF Linear[out, in] from TL [head, in, head_dim]."""

    return weight.permute(0, 2, 1).reshape(-1, int(weight.shape[1])).contiguous()


def _native_output_projection_weight(weight: torch.Tensor) -> torch.Tensor:
    """Restore HF Linear[out, in] from TL [head, head_dim, model]."""

    return weight.reshape(-1, int(weight.shape[-1])).T.contiguous()


def _native_rotary_initialization_device(
    numerics_backend: str,
    target_device: str | torch.device,
) -> torch.device:
    # Native HF constructs Qwen3RotaryEmbedding before model.to(cuda), so its
    # inverse frequencies are computed on CPU and then copied to the GPU.
    if numerics_backend == NATIVE_HF_CPU_ROPE_BACKEND:
        return torch.device("cpu")
    return torch.device(target_device)


def _has_active_hook_point(hook_point: Any) -> bool:
    return bool(
        getattr(hook_point, "fwd_hooks", ()) or getattr(hook_point, "bwd_hooks", ())
    )


def _attention_has_internal_observer_hooks(attention: Any) -> bool:
    return any(
        _has_active_hook_point(getattr(attention, name, None))
        for name in (
            "hook_q",
            "hook_k",
            "hook_v",
            "hook_rot_q",
            "hook_rot_k",
            "hook_z",
            "hook_result",
        )
    )


def _mlp_has_internal_observer_hooks(mlp: Any) -> bool:
    return any(
        _has_active_hook_point(getattr(mlp, name, None))
        for name in ("hook_pre", "hook_pre_linear", "hook_mid", "hook_post")
    )


def _native_hf_rms_norm_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    """Match transformers 4.55 Qwen3 RMSNorm while retaining TL hook points."""

    input_dtype = x.dtype
    x_float = x.to(torch.float32)
    variance = x_float.pow(2).mean(-1, keepdim=True)
    inverse_rms = self.hook_scale(torch.rsqrt(variance + self.eps))
    normalized = self.hook_normalized((x_float * inverse_rms).to(input_dtype))
    if self.w.device != normalized.device:
        self.to(normalized.device)
    return self.w * normalized


def _native_qk_norm(
    value: torch.Tensor,
    *,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    input_dtype = value.dtype
    value_float = value.to(torch.float32)
    variance = value_float.pow(2).mean(-1, keepdim=True)
    normalized = value_float * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)


def _native_hf_flash_attention_forward(
    self: Any,
    query_input: torch.Tensor,
    key_input: torch.Tensor,
    value_input: torch.Tensor,
    past_kv_cache_entry: Any = None,
    additive_attention_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run frozen-QK direct effects with Qwen3's native HF numerical path."""

    noop_additive_mask = additive_attention_mask is None or bool(
        torch.all(additive_attention_mask == 0).item()
    )
    noop_attention_mask = attention_mask is None or bool(
        torch.all(attention_mask != 0).item()
    )
    unsupported = (
        past_kv_cache_entry is not None
        or not noop_additive_mask
        or not noop_attention_mask
        or position_bias is not None
        or self.cfg.attention_dir != "causal"
        or self.cfg.positional_embedding_type != "rotary"
        or self.cfg.attn_scores_soft_cap > 0
        or self.cfg.load_in_4bit
        or self.cfg.use_attn_result
        or (
            _attention_has_internal_observer_hooks(self)
            and not bool(getattr(self, "_nvarc_diagnostic_observer_hooks", False))
        )
    )
    if unsupported:
        return self._nvarc_original_forward(
            query_input,
            key_input,
            value_input,
            past_kv_cache_entry=past_kv_cache_entry,
            additive_attention_mask=additive_attention_mask,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )

    max_chunk_tokens = int(getattr(self, "_nvarc_attention_chunk_tokens", 32_768))
    should_chunk = (
        query_input.ndim == 3
        and int(query_input.shape[0]) > 1
        and int(query_input.shape[0]) * int(query_input.shape[1]) > max_chunk_tokens
    )
    if not should_chunk:
        return _native_hf_flash_attention_forward_unchunked(
            self, query_input, key_input, value_input
        )

    rows_per_chunk = max(
        1,
        min(
            int(query_input.shape[0]),
            max_chunk_tokens // int(query_input.shape[1]),
        ),
    )
    outputs = [
        _native_hf_flash_attention_forward_unchunked(
            self, query_chunk, key_chunk, value_chunk
        )
        for query_chunk, key_chunk, value_chunk in zip(
            query_input.split(rows_per_chunk, dim=0),
            key_input.split(rows_per_chunk, dim=0),
            value_input.split(rows_per_chunk, dim=0),
            strict=True,
        )
    ]
    return torch.cat(outputs, dim=0)


def _native_hf_flash_attention_forward_unchunked(
    self: Any,
    query_input: torch.Tensor,
    key_input: torch.Tensor,
    value_input: torch.Tensor,
) -> torch.Tensor:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    batch, sequence_length, _ = query_input.shape
    n_heads = int(self.cfg.n_heads)
    n_kv_heads = int(self.cfg.n_key_value_heads or n_heads)
    d_head = int(self.cfg.d_head)

    query = F.linear(query_input, self._nvarc_native_w_q).view(
        batch, sequence_length, n_heads, d_head
    )
    key = F.linear(key_input, self._nvarc_native_w_k).view(
        batch, sequence_length, n_kv_heads, d_head
    )
    value = F.linear(value_input, self._nvarc_native_w_v).view(
        batch, sequence_length, n_kv_heads, d_head
    )
    query = self.hook_q(query)
    key = self.hook_k(key)
    value = self.hook_v(value)
    query = _native_qk_norm(
        query,
        weight=self.q_norm.w,
        eps=float(self.q_norm.eps),
    )
    key = _native_qk_norm(
        key,
        weight=self.k_norm.w,
        eps=float(self.k_norm.eps),
    )

    position_ids = torch.arange(sequence_length, device=query.device).unsqueeze(0)
    cos, sin = self._nvarc_native_rotary_embedding(query, position_ids)
    query_heads, key_heads = apply_rotary_pos_emb(
        query.transpose(1, 2),
        key.transpose(1, 2),
        cos,
        sin,
    )
    query = self.hook_rot_q(query_heads.transpose(1, 2))
    key = self.hook_rot_k(key_heads.transpose(1, 2))

    # CodeCircuit freezes the attention pattern. Detaching Q/K gives the same
    # gradient boundary while preserving the native FlashAttention2 forward.
    if self._nvarc_native_attention_backend in {
        NATIVE_HF_INTEGRATION_BACKEND,
        NATIVE_HF_CPU_ROPE_BACKEND,
        NATIVE_HF_REUSED_ROPE_BACKEND,
    }:
        from transformers.integrations.flash_attention import flash_attention_forward

        z, _ = flash_attention_forward(
            self,
            query.transpose(1, 2).detach(),
            key.transpose(1, 2).detach(),
            value.transpose(1, 2),
            None,
            dropout=0.0,
            scaling=1.0 / math.sqrt(d_head),
            sliding_window=None,
        )
    else:
        from transformers.modeling_flash_attention_utils import (
            _flash_attention_forward,
        )

        z = _flash_attention_forward(
            query.detach(),
            key.detach(),
            value,
            None,
            query_length=sequence_length,
            is_causal=True,
            dropout=0.0,
            position_ids=position_ids,
            softmax_scale=1.0 / math.sqrt(d_head),
            use_top_left_mask=bool(self._nvarc_native_use_top_left_mask),
            target_dtype=value.dtype,
            implementation="flash_attention_2",
        )
    z = self.hook_z(z)
    return F.linear(
        z.reshape(batch, sequence_length, n_heads * d_head),
        self._nvarc_native_w_o,
    )


def _native_hf_gated_mlp_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    if _mlp_has_internal_observer_hooks(self):
        return self._nvarc_original_forward(x)
    if self.W_gate.device != x.device:
        x = x.to(self.W_gate.device)

    max_chunk_tokens = int(getattr(self, "_nvarc_mlp_chunk_tokens", 4096))
    rows_per_chunk = int(x.shape[0])
    if x.ndim == 3 and int(x.shape[0]) * int(x.shape[1]) > max_chunk_tokens:
        rows_per_chunk = max(
            1,
            min(int(x.shape[0]), max_chunk_tokens // int(x.shape[1])),
        )

    def chunk_forward(chunk: torch.Tensor) -> torch.Tensor:
        gate = self.hook_pre(F.linear(chunk, self._nvarc_native_w_gate))
        up = self.hook_pre_linear(F.linear(chunk, self._nvarc_native_w_up))
        hidden = self.hook_post(self.act_fn(gate) * up)
        return F.linear(hidden, self._nvarc_native_w_down)

    outputs = []
    for chunk in x.split(rows_per_chunk, dim=0):
        if torch.is_grad_enabled() and chunk.requires_grad:
            outputs.append(checkpoint(chunk_forward, chunk, use_reentrant=False))
        else:
            outputs.append(chunk_forward(chunk))
    return torch.cat(outputs, dim=0)


def _native_hf_unembed_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, self._nvarc_native_weight)


def _install_native_rms_norms(model: Any) -> None:
    norms = [model.ln_final]
    for block in model.blocks:
        norms.extend((block.ln1, block.ln2))
    for norm in norms:
        if hasattr(norm, "_nvarc_native_hf_original_forward"):
            continue
        norm._nvarc_native_hf_original_forward = norm.forward
        norm.forward = types.MethodType(_native_hf_rms_norm_forward, norm)


def _install_native_attention(
    model: Any,
    hf_config: Any,
    *,
    numerics_backend: str,
    hf_rotary_embedding: Any | None,
) -> None:
    from transformers.modeling_flash_attention_utils import (
        flash_attn_supports_top_left_mask,
    )
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    if numerics_backend == NATIVE_HF_REUSED_ROPE_BACKEND:
        if hf_rotary_embedding is None:
            raise ValueError("Reused-RoPE backend requires the loaded HF rotary module")
        rotary = copy.deepcopy(hf_rotary_embedding).to(device=model.cfg.device)
    else:
        rotary = Qwen3RotaryEmbedding(
            hf_config,
            device=_native_rotary_initialization_device(
                numerics_backend,
                model.cfg.device,
            ),
        ).to(device=model.cfg.device)
    model._nvarc_native_rotary_embedding = rotary
    use_top_left_mask = bool(flash_attn_supports_top_left_mask())
    attention_config = copy.deepcopy(hf_config)
    attention_config._attn_implementation = "flash_attention_2"

    for layer, block in enumerate(model.blocks):
        attn = block.attn
        if hasattr(attn, "_nvarc_original_forward"):
            raise RuntimeError("Native-HF attention must be installed on an unpatched bridge")
        raw_k = attn._W_K if hasattr(attn, "_W_K") else attn.W_K
        raw_v = attn._W_V if hasattr(attn, "_W_V") else attn.W_V
        attn._nvarc_original_forward = attn.forward
        attn._nvarc_attention_chunk_tokens = 32_768
        attn._nvarc_native_w_q = _native_input_projection_weight(attn.W_Q)
        attn._nvarc_native_w_k = _native_input_projection_weight(raw_k)
        attn._nvarc_native_w_v = _native_input_projection_weight(raw_v)
        attn._nvarc_native_w_o = _native_output_projection_weight(attn.W_O)
        object.__setattr__(attn, "_nvarc_native_rotary_embedding", rotary)
        attn._nvarc_native_use_top_left_mask = use_top_left_mask
        attn._nvarc_native_attention_backend = numerics_backend
        if numerics_backend in {
            NATIVE_HF_INTEGRATION_BACKEND,
            NATIVE_HF_CPU_ROPE_BACKEND,
            NATIVE_HF_REUSED_ROPE_BACKEND,
        }:
            attn.config = attention_config
            attn.is_causal = True
            attn.layer_idx = layer
        attn.forward = types.MethodType(_native_hf_flash_attention_forward, attn)


def _install_native_mlp(model: Any) -> None:
    for block in model.blocks:
        mlp = getattr(block.mlp, "old_mlp", block.mlp)
        if hasattr(mlp, "_nvarc_original_forward"):
            raise RuntimeError("Native-HF MLP must be installed on an unpatched bridge")
        mlp._nvarc_original_forward = mlp.forward
        mlp._nvarc_mlp_chunk_tokens = 4096
        mlp._nvarc_native_w_gate = mlp.W_gate.T.contiguous()
        mlp._nvarc_native_w_up = mlp.W_in.T.contiguous()
        mlp._nvarc_native_w_down = mlp.W_out.T.contiguous()
        mlp.forward = types.MethodType(_native_hf_gated_mlp_forward, mlp)


def _install_native_unembed(model: Any) -> None:
    unembed = getattr(model.unembed, "old_unembed", model.unembed)
    unembed._nvarc_native_hf_original_forward = unembed.forward
    unembed._nvarc_native_weight = unembed.W_U.T.contiguous()
    unembed.forward = types.MethodType(_native_hf_unembed_forward, unembed)


def install_native_hf_numerics(
    model: Any,
    *,
    hf_config: Any,
    numerics_backend: str = NATIVE_HF_NUMERICS_BACKEND,
    hf_rotary_embedding: Any | None = None,
) -> dict[str, str]:
    """Install an isolated HF-numerical candidate on a configured TL model."""

    if numerics_backend not in {
        NATIVE_HF_NUMERICS_BACKEND,
        NATIVE_HF_INTEGRATION_BACKEND,
        NATIVE_HF_CPU_ROPE_BACKEND,
        NATIVE_HF_REUSED_ROPE_BACKEND,
    }:
        raise ValueError(f"Unsupported native-HF numerics backend: {numerics_backend}")
    if bool(getattr(hf_config, "attention_bias", False)):
        raise ValueError("Native-HF numerics currently requires bias-free Qwen3 attention")
    _install_native_rms_norms(model)
    _install_native_attention(
        model,
        hf_config,
        numerics_backend=numerics_backend,
        hf_rotary_embedding=hf_rotary_embedding,
    )
    _install_native_mlp(model)
    _install_native_unembed(model)
    if numerics_backend == NATIVE_HF_REUSED_ROPE_BACKEND:
        attention_identity = "qwen3_hf_reused_loaded_rope_fa2_frozen_qk_v4"
    elif numerics_backend == NATIVE_HF_CPU_ROPE_BACKEND:
        attention_identity = "qwen3_hf_cpu_initialized_rope_fa2_frozen_qk_v3"
    elif numerics_backend == NATIVE_HF_INTEGRATION_BACKEND:
        attention_identity = "qwen3_hf_flash_attention_integration_frozen_qk_v2"
    else:
        attention_identity = "qwen3_hf_flinear_rope_fa2_frozen_qk_v1"
    return {
        "numerics_backend": numerics_backend,
        "normalization": "qwen3_hf_rsqrt_with_frozen_inverse_rms_v1",
        "direct_effect_attention": attention_identity,
        "direct_effect_mlp": "qwen3_hf_flinear_chunked_gated_v1",
        "unembed": "qwen3_hf_flinear_v1",
    }
