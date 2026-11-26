from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import time

import importlib.metadata
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformer_lens.utilities.addmm import batch_addmm

from s3circuit.graph_extract import ensure_codecircuit_on_path

from .identity import semantic_sha256, sha256_file
from .hf_numerics_bridge import (
    NATIVE_HF_CPU_ROPE_BACKEND,
    NATIVE_HF_INTEGRATION_BACKEND,
    NATIVE_HF_NUMERICS_BACKEND,
    NATIVE_HF_REUSED_ROPE_BACKEND,
    install_native_hf_numerics,
)
from .storage import write_json_atomic

TRANSFORMER_LENS_VERSION = "2.17.0"
BRIDGE_FORMAT = "nvarc_qwen3_tl_bridge_v1"
TL_SDPA_NUMERICS_BACKEND = "tl_sdpa_direct_effect_numerics_v1"

_ATTRIBUTION_SETUP_GATE = ContextVar("attribution_setup_memory_gate", default=None)


@contextmanager
def attribution_setup_gate(factory):
    """Gate only transient attribution setup, not the long VJP/CPU stages."""
    token = _ATTRIBUTION_SETUP_GATE.set(factory)
    try:
        yield
    finally:
        _ATTRIBUTION_SETUP_GATE.reset(token)


@dataclass(frozen=True)
class LocalQwenBridge:
    model: Any
    tokenizer: Any
    manifest: dict[str, Any]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _require_transformer_lens_version() -> str:
    actual = importlib.metadata.version("transformer-lens")
    if actual != TRANSFORMER_LENS_VERSION:
        raise RuntimeError(
            f"TransformerLens version changed: {actual} != {TRANSFORMER_LENS_VERSION}"
        )
    return actual


def _validate_hf_config(config: Any, *, n_ctx: int) -> None:
    expected = {
        "model_type": "qwen3",
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 16,
        "tie_word_embeddings": True,
        "rope_theta": 5_000_000.0,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": getattr(config, key, None)}
        for key, expected_value in expected.items()
        if getattr(config, key, None) != expected_value
    }
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if architectures != ("Qwen3ForCausalLM",):
        mismatches["architectures"] = {
            "expected": ["Qwen3ForCausalLM"],
            "actual": list(architectures),
        }
    if int(n_ctx) != 16_384:
        mismatches["n_ctx"] = {"expected": 16_384, "actual": int(n_ctx)}
    if mismatches:
        raise ValueError(f"Local Qwen3 checkpoint contract changed: {mismatches}")


def _validate_tl_config(config: Any, hf_config: Any, *, n_ctx: int) -> None:
    expected = {
        "d_model": int(hf_config.hidden_size),
        "d_mlp": int(hf_config.intermediate_size),
        "n_layers": int(hf_config.num_hidden_layers),
        "n_heads": int(hf_config.num_attention_heads),
        "n_key_value_heads": int(hf_config.num_key_value_heads),
        "d_head": int(hf_config.head_dim),
        "d_vocab": int(hf_config.vocab_size),
        "n_ctx": int(n_ctx),
        "normalization_type": "RMS",
        "positional_embedding_type": "rotary",
        "rotary_base": int(hf_config.rope_theta),
        "rotary_dim": int(hf_config.head_dim),
        "gated_mlp": True,
        "final_rms": True,
        "use_qk_norm": True,
        "default_prepend_bos": False,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": getattr(config, key, None)}
        for key, expected_value in expected.items()
        if getattr(config, key, None) != expected_value
    }
    if float(config.eps) != float(hf_config.rms_norm_eps):
        mismatches["eps"] = {
            "expected": float(hf_config.rms_norm_eps),
            "actual": float(config.eps),
        }
    if mismatches:
        raise ValueError(
            f"TransformerLens Qwen3 config conversion changed: {mismatches}"
        )


def qwen3_conversion_map(n_layers: int = 36) -> tuple[dict[str, Any], ...]:
    """Return the exact source-to-destination map used by TL 2.17's Qwen3 converter."""

    rows: list[dict[str, Any]] = [
        {"source": "model.embed_tokens.weight", "destination": "embed.W_E"}
    ]
    for layer in range(n_layers):
        prefix = f"model.layers.{layer}"
        destination = f"blocks.{layer}"
        rows.extend(
            (
                {
                    "source": f"{prefix}.input_layernorm.weight",
                    "destination": f"{destination}.ln1.w",
                },
                {
                    "source": f"{prefix}.self_attn.q_proj.weight",
                    "destination": f"{destination}.attn.W_Q",
                },
                {
                    "source": f"{prefix}.self_attn.k_proj.weight",
                    "destination": f"{destination}.attn._W_K",
                },
                {
                    "source": f"{prefix}.self_attn.v_proj.weight",
                    "destination": f"{destination}.attn._W_V",
                },
                {
                    "source": f"{prefix}.self_attn.q_norm.weight",
                    "destination": f"{destination}.attn.q_norm.w",
                },
                {
                    "source": f"{prefix}.self_attn.k_norm.weight",
                    "destination": f"{destination}.attn.k_norm.w",
                },
                {
                    "source": None,
                    "destination": f"{destination}.attn.b_Q",
                    "synthetic": "zeros",
                },
                {
                    "source": None,
                    "destination": f"{destination}.attn._b_K",
                    "synthetic": "zeros",
                },
                {
                    "source": None,
                    "destination": f"{destination}.attn._b_V",
                    "synthetic": "zeros",
                },
                {
                    "source": f"{prefix}.self_attn.o_proj.weight",
                    "destination": f"{destination}.attn.W_O",
                },
                {
                    "source": None,
                    "destination": f"{destination}.attn.b_O",
                    "synthetic": "zeros",
                },
                {
                    "source": f"{prefix}.post_attention_layernorm.weight",
                    "destination": f"{destination}.ln2.w",
                },
                {
                    "source": f"{prefix}.mlp.up_proj.weight",
                    "destination": f"{destination}.mlp.W_in",
                },
                {
                    "source": f"{prefix}.mlp.gate_proj.weight",
                    "destination": f"{destination}.mlp.W_gate",
                },
                {
                    "source": None,
                    "destination": f"{destination}.mlp.b_in",
                    "synthetic": "zeros",
                },
                {
                    "source": f"{prefix}.mlp.down_proj.weight",
                    "destination": f"{destination}.mlp.W_out",
                },
                {
                    "source": None,
                    "destination": f"{destination}.mlp.b_out",
                    "synthetic": "zeros",
                },
            )
        )
    rows.extend(
        (
            {"source": "model.norm.weight", "destination": "ln_final.w"},
            {"source": "lm_head.weight", "destination": "unembed.W_U"},
            {"source": None, "destination": "unembed.b_U", "synthetic": "zeros"},
        )
    )
    return tuple(rows)


def _conversion_manifest(
    *,
    checkpoint: Path,
    tokenizer_path: Path,
    hf_config: Any,
    tl_config: Any,
    state_dict: dict[str, torch.Tensor],
    transformer_lens_version: str,
) -> dict[str, Any]:
    conversion = qwen3_conversion_map(int(hf_config.num_hidden_layers))
    expected_keys = {row["destination"] for row in conversion}
    actual_keys = set(state_dict)
    if actual_keys != expected_keys:
        raise ValueError(
            "Qwen3 conversion key coverage changed: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    rows = []
    for mapping in conversion:
        tensor = state_dict[mapping["destination"]]
        rows.append(
            {
                **mapping,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        )
    checkpoint_files = sorted(checkpoint.glob("*.safetensors")) + sorted(
        checkpoint.glob("pytorch_model*.bin")
    )
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint weight shards found under {checkpoint}")
    value = {
        "format": BRIDGE_FORMAT,
        "transformer_lens_version": transformer_lens_version,
        "checkpoint": str(checkpoint),
        "tokenizer": str(tokenizer_path),
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "checkpoint_weight_shards": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in checkpoint_files
        },
        "hf_config": _json_safe(hf_config.to_dict()),
        "tl_config": _json_safe(tl_config.to_dict()),
        "conversion": rows,
        "weight_processing": {
            "fold_ln": False,
            "center_writing_weights": False,
            "center_unembed": False,
            "fold_value_biases": False,
            "refactor_factored_attn_matrices": False,
        },
        "position_zero_error_policy": "preserve_real_sft_token",
    }
    value["semantic_sha256"] = semantic_sha256(value)
    return value


def _compute_attribution_components_preserving_position_zero(
    transcoders: Any,
    mlp_inputs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Mirror CodeCircuit's PLT components without its dummy-BOS zeroing."""

    reconstruction = torch.zeros_like(mlp_inputs)
    encoder_vectors = []
    decoder_vectors = []
    sparse_activations = []
    for layer, transcoder in enumerate(transcoders):
        sparse, active_encoders = transcoder.encode_sparse(
            mlp_inputs[layer], zero_first_pos=False
        )
        reconstruction[layer], active_decoders = _decode_sparse_deterministic(
            transcoder, sparse
        )
        encoder_vectors.append(active_encoders)
        decoder_vectors.append(active_decoders)
        sparse_activations.append(sparse)
        # The lists own these tensors until concatenation; do not retain the
        # last layer through the local variables after clearing a list.
        del active_encoders, active_decoders
    activation_matrix = torch.stack(sparse_activations).coalesce()
    feature_count = int(activation_matrix._nnz())
    encoder_vecs = torch.cat(encoder_vectors, dim=0)
    encoder_vectors.clear()
    decoder_vecs = torch.cat(decoder_vectors, dim=0)
    decoder_vectors.clear()
    return {
        "activation_matrix": activation_matrix,
        "reconstruction": reconstruction,
        "encoder_vecs": encoder_vecs,
        "decoder_vecs": decoder_vecs,
        "encoder_to_decoder_map": torch.arange(feature_count, device=mlp_inputs.device),
        "decoder_locations": activation_matrix.indices()[:2],
    }


def _decode_sparse_deterministic(
    transcoder: Any, sparse_activations: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode sorted sparse features without CUDA atomic accumulation."""

    if sparse_activations.layout != torch.sparse_coo:
        raise TypeError("NVARC transcoder decoding requires sparse COO activations")
    sparse = sparse_activations.coalesce()
    if sparse.ndim != 2:
        raise ValueError("NVARC transcoder activations must be [position, feature]")
    position_indices, feature_indices = sparse.indices()
    if position_indices.numel() > 1 and bool(
        torch.any(position_indices[1:] < position_indices[:-1]).item()
    ):
        raise ValueError("Coalesced transcoder activations are not position-sorted")
    if getattr(transcoder, "W_skip", None) is not None:
        raise ValueError("Deterministic sparse decoding does not support skip transcoders")

    values = sparse.values()
    decoder_vectors = transcoder._get_decoder_vectors(feature_indices.detach().cpu())
    scaled_decoders = decoder_vectors * values[:, None]
    position_counts = torch.bincount(
        position_indices, minlength=int(sparse.shape[0])
    )
    reconstruction = torch.segment_reduce(
        scaled_decoders,
        "sum",
        lengths=position_counts,
        axis=0,
        unsafe=False,
    )
    reconstruction = reconstruction + transcoder.b_dec
    return reconstruction, scaled_decoders


def _replacement_model_class() -> type:
    ensure_codecircuit_on_path()
    from circuit_tracer.attribution.context import AttributionContext
    from circuit_tracer.replacement_model import ReplacementModel

    class NvarcReplacementModel(ReplacementModel):
        @torch.no_grad()
        def setup_attribution(self, inputs: str | torch.Tensor) -> Any:
            factory = _ATTRIBUTION_SETUP_GATE.get()
            started = time.monotonic()
            with factory() if factory is not None else nullcontext():
                acquired = time.monotonic()
                context = self._setup_attribution_impl(inputs)
                if factory is not None:
                    # Release unused, formerly interleaved encoder/decoder
                    # allocation blocks before the peer enters its peak.
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                self._nvarc_setup_memory_metrics = {
                    "wait_seconds": acquired - started,
                    "build_seconds": time.monotonic() - acquired,
                }
                return context

        @torch.no_grad()
        def _setup_attribution_impl(self, inputs: str | torch.Tensor) -> Any:
            if isinstance(inputs, str):
                raise TypeError(
                    "NVARC attribution requires frozen pretokenized IDs; string input is forbidden"
                )
            tokens = inputs.squeeze()
            if tokens.ndim != 1:
                raise ValueError("NVARC attribution tokens must be one-dimensional")
            mlp_in_cache, mlp_in_hooks, _ = self.get_caching_hooks(
                lambda name: self.feature_input_hook in name
            )
            mlp_out_cache, mlp_out_hooks, _ = self.get_caching_hooks(
                lambda name: self.feature_output_hook in name
            )
            logits = self.run_with_hooks(tokens, fwd_hooks=mlp_in_hooks + mlp_out_hooks)
            mlp_inputs = torch.cat(list(mlp_in_cache.values()), dim=0)
            mlp_in_cache.clear()
            mlp_outputs = torch.cat(list(mlp_out_cache.values()), dim=0)
            mlp_out_cache.clear()
            components = _compute_attribution_components_preserving_position_zero(
                self.transcoders, mlp_inputs
            )
            reconstruction = components["reconstruction"]
            error_vectors = mlp_outputs - reconstruction
            token_vectors = self.W_E[tokens].detach()
            context = AttributionContext(
                activation_matrix=components["activation_matrix"],
                logits=logits,
                error_vectors=error_vectors,
                token_vectors=token_vectors,
                decoder_vecs=components["decoder_vecs"],
                encoder_vecs=components["encoder_vecs"],
                encoder_to_decoder_map=components["encoder_to_decoder_map"],
                decoder_locations=components["decoder_locations"],
            )
            if bool(getattr(self, "_nvarc_capture_attribution_components", False)):
                context._nvarc_attribution_components = {
                    "mlp_inputs": mlp_inputs.detach(),
                    "mlp_outputs": mlp_outputs.detach(),
                    "transcoder_reconstruction": reconstruction.detach(),
                    "error_vectors": error_vectors.detach(),
                }
            return context

    return NvarcReplacementModel


def _frozen_qk_sdpa_attention_forward(
    self: Any,
    query_input: torch.Tensor,
    key_input: torch.Tensor,
    value_input: torch.Tensor,
    past_kv_cache_entry: Any = None,
    additive_attention_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Memory-efficient direct-effect attention for frozen-pattern attribution."""

    original_additive_attention_mask = additive_attention_mask
    original_attention_mask = attention_mask
    noop_additive_attention_mask = additive_attention_mask is None or bool(
        torch.all(additive_attention_mask == 0).item()
    )
    noop_attention_mask = attention_mask is None or bool(
        torch.all(attention_mask != 0).item()
    )

    unsupported = (
        past_kv_cache_entry is not None
        or not noop_additive_attention_mask
        or not noop_attention_mask
        or position_bias is not None
        or self.cfg.attention_dir != "causal"
        or self.cfg.positional_embedding_type != "rotary"
        or self.cfg.attn_scores_soft_cap > 0
        or self.cfg.load_in_4bit
    )
    if unsupported:
        return self._nvarc_original_forward(
            query_input,
            key_input,
            value_input,
            past_kv_cache_entry=past_kv_cache_entry,
            additive_attention_mask=original_additive_attention_mask,
            attention_mask=original_attention_mask,
            position_bias=position_bias,
        )
    additive_attention_mask = None
    attention_mask = None

    max_chunk_tokens = int(getattr(self, "_nvarc_attention_chunk_tokens", 32_768))
    should_chunk = (
        query_input.ndim >= 2
        and int(query_input.shape[0]) > 1
        and int(query_input.shape[0]) * int(query_input.shape[1]) > max_chunk_tokens
    )
    if should_chunk:
        if _attention_has_internal_hooks(self):
            return self._nvarc_original_forward(
                query_input,
                key_input,
                value_input,
                past_kv_cache_entry=past_kv_cache_entry,
                additive_attention_mask=original_additive_attention_mask,
                attention_mask=original_attention_mask,
                position_bias=position_bias,
            )
        rows_per_chunk = max(
            1,
            min(
                int(query_input.shape[0]), max_chunk_tokens // int(query_input.shape[1])
            ),
        )
        outputs = []
        for query_chunk, key_chunk, value_chunk in zip(
            query_input.split(rows_per_chunk, dim=0),
            key_input.split(rows_per_chunk, dim=0),
            value_input.split(rows_per_chunk, dim=0),
            strict=True,
        ):
            outputs.append(
                _frozen_qk_sdpa_attention_forward_unchunked(
                    self, query_chunk, key_chunk, value_chunk
                )
            )
        return torch.cat(outputs, dim=0)

    return _frozen_qk_sdpa_attention_forward_unchunked(
        self, query_input, key_input, value_input
    )


def _frozen_qk_sdpa_attention_forward_unchunked(
    self: Any,
    query_input: torch.Tensor,
    key_input: torch.Tensor,
    value_input: torch.Tensor,
) -> torch.Tensor:
    """Run the frozen-pattern SDPA path for one already-sized batch chunk."""

    q, k, v = self.calculate_qkv_matrices(query_input, key_input, value_input)
    q = self.hook_rot_q(self.apply_rotary(q, 0, None))
    k = self.hook_rot_k(self.apply_rotary(k, 0, None))

    q_heads = q.transpose(1, 2).detach().to(dtype=v.dtype)
    k_heads = k.transpose(1, 2).detach().to(dtype=v.dtype)
    v_heads = v.transpose(1, 2)
    enable_gqa = (
        not self.cfg.ungroup_grouped_query_attention
        and getattr(self.cfg, "n_key_value_heads", None) is not None
        and int(self.cfg.n_key_value_heads) != int(self.cfg.n_heads)
    )
    z_heads = F.scaled_dot_product_attention(
        q_heads,
        k_heads,
        v_heads,
        dropout_p=0.0,
        is_causal=True,
        scale=1.0 / float(self.attn_scale),
        enable_gqa=enable_gqa,
    )
    z = self.hook_z(z_heads.transpose(1, 2))
    if self.cfg.use_attn_result:
        output_weight = self.W_O
        if self.b_O.device != output_weight.device:
            output_weight = output_weight.to(self.b_O.device)
        if self.b_O.device != z.device:
            z = z.to(self.b_O.device)
        result = self.hook_result(torch.einsum("bphd,hdm->bphm", z, output_weight))
        return result.sum(dim=2) + self.b_O

    weight = self.W_O.reshape(self.cfg.n_heads * self.cfg.d_head, self.cfg.d_model).T
    if self.b_O.device != weight.device:
        weight = weight.to(self.b_O.device)
    if self.b_O.device != z.device:
        z = z.to(self.b_O.device)
    return F.linear(
        z.reshape(z.shape[0], z.shape[1], self.cfg.d_head * self.cfg.n_heads),
        weight,
        self.b_O,
    )


def _install_memory_efficient_direct_effect_attention(model: Any) -> None:
    for block in model.blocks:
        attn = block.attn
        if hasattr(attn, "_nvarc_original_forward"):
            continue
        attn._nvarc_original_forward = attn.forward
        attn._nvarc_attention_chunk_tokens = 32_768
        attn.forward = types.MethodType(_frozen_qk_sdpa_attention_forward, attn)


def _has_active_hook_point(hook_point: Any) -> bool:
    return bool(
        getattr(hook_point, "fwd_hooks", ()) or getattr(hook_point, "bwd_hooks", ())
    )


def _gated_mlp_has_internal_hooks(mlp: Any) -> bool:
    return any(
        _has_active_hook_point(getattr(mlp, name, None))
        for name in ("hook_pre", "hook_pre_linear", "hook_mid", "hook_post")
    )


def _attention_has_internal_hooks(attn: Any) -> bool:
    return any(
        _has_active_hook_point(getattr(attn, name, None))
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


def _chunked_gated_mlp_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    """Memory-efficient GatedMLP forward for exact direct-effect row batches."""

    unsupported = (
        not all(hasattr(self, name) for name in ("W_gate", "W_in", "W_out"))
        or self.cfg.is_layer_norm_activation()
        or _gated_mlp_has_internal_hooks(self)
    )
    if unsupported:
        return self._nvarc_original_forward(x)
    if self.W_gate.device != x.device:
        x = x.to(self.W_gate.device)

    max_chunk_tokens = int(getattr(self, "_nvarc_mlp_chunk_tokens", 4096))
    if x.ndim != 3 or x.shape[0] * x.shape[1] <= max_chunk_tokens:
        return self._nvarc_original_forward(x)

    rows_per_chunk = max(1, min(int(x.shape[0]), max_chunk_tokens // int(x.shape[1])))

    def chunk_forward(chunk: torch.Tensor) -> torch.Tensor:
        pre_act = torch.matmul(chunk, self.W_gate)
        pre_linear = torch.matmul(chunk, self.W_in)
        post_act = (self.act_fn(pre_act) * pre_linear) + self.b_in
        return batch_addmm(self.b_out, self.W_out, post_act)

    chunks = []
    for chunk in x.split(rows_per_chunk, dim=0):
        if torch.is_grad_enabled() and chunk.requires_grad:
            chunks.append(checkpoint(chunk_forward, chunk, use_reentrant=False))
        else:
            chunks.append(chunk_forward(chunk))
    return torch.cat(chunks, dim=0)


def _install_memory_efficient_direct_effect_mlp(model: Any) -> None:
    for block in model.blocks:
        mlp = getattr(block.mlp, "old_mlp", block.mlp)
        if hasattr(mlp, "_nvarc_original_forward"):
            continue
        if not all(hasattr(mlp, name) for name in ("W_gate", "W_in", "W_out")):
            continue
        mlp._nvarc_original_forward = mlp.forward
        mlp._nvarc_mlp_chunk_tokens = 4096
        mlp.forward = types.MethodType(_chunked_gated_mlp_forward, mlp)


def build_local_qwen3_replacement_model(
    *,
    checkpoint: str | Path,
    tokenizer_path: str | Path,
    transcoders: Any,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    n_ctx: int = 16_384,
    manifest_path: str | Path | None = None,
    numerics_backend: str = TL_SDPA_NUMERICS_BACKEND,
) -> LocalQwenBridge:
    """Convert the exact local HF checkpoint through TL 2.17 and attach transcoders."""

    transformer_lens_version = _require_transformer_lens_version()
    if numerics_backend not in {
        TL_SDPA_NUMERICS_BACKEND,
        NATIVE_HF_NUMERICS_BACKEND,
        NATIVE_HF_INTEGRATION_BACKEND,
        NATIVE_HF_CPU_ROPE_BACKEND,
        NATIVE_HF_REUSED_ROPE_BACKEND,
    }:
        raise ValueError(f"Unsupported Qwen3 bridge numerics backend: {numerics_backend}")
    ensure_codecircuit_on_path()
    from transformer_lens import loading_from_pretrained as loading
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    checkpoint_path = Path(checkpoint).resolve()
    tokenizer_root = Path(tokenizer_path).resolve()
    if not checkpoint_path.is_dir() or not tokenizer_root.is_dir():
        raise FileNotFoundError("Local checkpoint or tokenizer directory is missing")
    hf_config = AutoConfig.from_pretrained(
        str(checkpoint_path), local_files_only=True, trust_remote_code=False
    )
    _validate_hf_config(hf_config, n_ctx=n_ctx)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_root), use_fast=True, local_files_only=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        torch_dtype=dtype,
        local_files_only=True,
        trust_remote_code=False,
    )
    hf_model.eval()
    tl_config = loading.get_pretrained_model_config(
        str(checkpoint_path),
        hf_cfg=hf_config.to_dict(),
        fold_ln=False,
        device=device,
        n_devices=1,
        default_prepend_bos=False,
        dtype=dtype,
        n_ctx=n_ctx,
        local_files_only=True,
        trust_remote_code=False,
    )
    _validate_tl_config(tl_config, hf_config, n_ctx=n_ctx)
    state_dict = loading.get_pretrained_state_dict(
        str(checkpoint_path),
        tl_config,
        hf_model=hf_model,
        dtype=dtype,
        local_files_only=True,
        trust_remote_code=False,
    )
    manifest = _conversion_manifest(
        checkpoint=checkpoint_path,
        tokenizer_path=tokenizer_root,
        hf_config=hf_config,
        tl_config=tl_config,
        state_dict=state_dict,
        transformer_lens_version=transformer_lens_version,
    )
    model_class = _replacement_model_class()
    model = model_class(
        tl_config,
        tokenizer=tokenizer,
        move_to_device=False,
        default_padding_side="left",
    )
    model.load_and_process_state_dict(
        state_dict,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        refactor_factored_attn_matrices=False,
    )
    model.move_model_modules_to_device()
    model._configure_replacement_model(transcoders)
    if numerics_backend in {
        NATIVE_HF_NUMERICS_BACKEND,
        NATIVE_HF_INTEGRATION_BACKEND,
        NATIVE_HF_CPU_ROPE_BACKEND,
        NATIVE_HF_REUSED_ROPE_BACKEND,
    }:
        manifest.update(
            install_native_hf_numerics(
                model,
                hf_config=hf_config,
                numerics_backend=numerics_backend,
                hf_rotary_embedding=(
                    hf_model.model.rotary_emb
                    if numerics_backend == NATIVE_HF_REUSED_ROPE_BACKEND
                    else None
                ),
            )
        )
    else:
        _install_memory_efficient_direct_effect_attention(model)
        _install_memory_efficient_direct_effect_mlp(model)
        manifest["numerics_backend"] = TL_SDPA_NUMERICS_BACKEND
        manifest["direct_effect_attention"] = "frozen_qk_sdpa_batch_chunked_v1"
        manifest["direct_effect_mlp"] = "checkpointed_chunked_gated_v1"
    model.eval()
    del state_dict
    del hf_model
    manifest["transcoder_reconstruction"] = "sorted_segment_reduce_v1"
    manifest["semantic_sha256"] = semantic_sha256(manifest)
    if manifest_path is not None:
        write_json_atomic(manifest_path, manifest)
    return LocalQwenBridge(model=model, tokenizer=tokenizer, manifest=manifest)
