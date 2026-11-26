"""End-to-end S5 Query/Circuit -> Generated-LoRA runtime.

The frozen S5 GPT is intentionally kept behind a small adapter here.  The
existing S2 PG and Generated-LoRA hook are reused for the mathematically
generic part, while all S5-specific tokenization, fusion, and checkpoint
identity handling stays local to this package.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from s2_hypernet.nvarc_pg.generated_lora import GeneratedLoRAController
from ..modeling import build_model
from .circuit_encoders import (
    CandidateABEncoder,
    CandidateAEncoder,
    CandidateBEncoder,
    append_after_valid_query,
)
from .circuit_projection import (
    VECTOR397_DIM,
    compact_circuit_state_dict,
    validate_retained_indices,
)
from .graph_condition import FeatureDefinitionBank, GraphBatch, S5GraphEncoder, graph_group_mask
from .config import (
    BASE_CHECKPOINT_SHA256,
    BASE_CONFIG_SHA256,
    BASE_STEP,
    S5BaseIdentity,
    base_content_identity_matches,
    semantic_sha256,
)
from .parameter_generator import (
    S5ParameterGenerator,
    build_s5_circuit_fusion_parameter_generator,
    build_s5_parameter_generator,
)
from .raw_weight import (
    S5RawParameterGenerator,
    S5RawQueryEncoderConditioner,
    S5RawWeightCache,
    build_s5_raw_parameter_generator,
)
from .query_encoder import ConditionTypeEmbeddings, QueryEncoder
from .target_layout import S5TargetLayout, build_s5_target_layout
from .losses import (
    base_preservation_hinge,
    causal_loss_stats_per_example,
    effective_delta_l2_per_example,
    scale_generated_lora_state,
)
from .alignment import query_circuit_alignment_stats


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def state_identity(state: Mapping[str, Tensor]) -> dict[str, Any]:
    tensors = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_sha256(value),
        }
        for key, value in sorted(state.items())
    }
    return {"tensors": tensors, "aggregate_sha256": semantic_sha256(tensors)}


def model_state_identity(state: Mapping[str, Tensor]) -> str:
    """Hash a Base model state independently of checkpoint-envelope metadata."""

    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_s5_base_model(
    architecture_config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    expected_identity: S5BaseIdentity | None = None,
    verified_checkpoint_sha256: str | None = None,
    verified_model_state_sha256: str | None = None,
) -> tuple[nn.Module, S5BaseIdentity]:
    """Load and verify the user-selected step-38000 ``best_model.pt`` Base."""

    config_source = Path(architecture_config_path).resolve()
    checkpoint_source = Path(checkpoint_path).resolve()
    if not config_source.is_file() or not checkpoint_source.is_file():
        raise FileNotFoundError("S5 Base architecture config or checkpoint is missing")
    architecture_file_sha = sha256_file(config_source)
    checkpoint_sha = verified_checkpoint_sha256 or sha256_file(checkpoint_source)
    expected = expected_identity or S5BaseIdentity(
        checkpoint_path=str(checkpoint_source),
        checkpoint_sha256=BASE_CHECKPOINT_SHA256,
        config_sha256=BASE_CONFIG_SHA256,
        step=BASE_STEP,
    )
    architecture = json.loads(config_source.read_text(encoding="utf-8"))
    if not isinstance(architecture, Mapping):
        raise ValueError("S5 Base architecture config must be a mapping")
    # The frozen protocol hash is the semantic identity embedded in the
    # resolved config and checkpoint.  It is not the SHA256 of the JSON file
    # bytes (the latter is recorded separately in run_identity.json).
    if str(architecture.get("config_sha256")) != expected.config_sha256:
        raise ValueError(
            "S5 Base config identity mismatch: "
            f"expected {expected.config_sha256}, got {architecture.get('config_sha256')}"
        )
    if checkpoint_sha != expected.checkpoint_sha256:
        raise ValueError(
            "S5 Base checkpoint SHA256 mismatch; refusing a nearby 38000 artifact: "
            f"expected {expected.checkpoint_sha256}, got {checkpoint_sha}"
        )
    # Do not compare ``architecture_config_path`` here.  SiFlow mounts the
    # shared volume under /mnt while local preparation uses /volume; the path
    # is provenance only.  The embedded config identity, checkpoint bytes and
    # model-state identity above are the immutable selection gate.
    model = build_model(architecture["model"])
    checkpoint = torch.load(
        checkpoint_source,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("S5 Base checkpoint envelope is not a mapping")
    if int(checkpoint.get("step", -1)) != BASE_STEP:
        raise ValueError("S5 Base checkpoint step is not 38000")
    if str(checkpoint.get("config_sha256")) != expected.config_sha256:
        raise ValueError("S5 Base checkpoint config identity changed")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("S5 Base checkpoint has no model_state_dict")
    measured_state_sha = verified_model_state_sha256 or model_state_identity(state)
    if measured_state_sha != expected.model_state_sha256:
        raise ValueError(
            "S5 Base model-state identity mismatch: "
            f"expected {expected.model_state_sha256}, got {measured_state_sha}"
        )
    model.load_state_dict(state, strict=True)
    model.config.grad_checkpointing = False
    model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    identity = S5BaseIdentity(
        checkpoint_path=str(checkpoint_source),
        architecture_config_path=str(config_source),
        checkpoint_sha256=checkpoint_sha,
        model_state_sha256=measured_state_sha,
        config_sha256=str(architecture["config_sha256"]),
        step=BASE_STEP,
    )
    # Keep this diagnostic available to callers without changing the frozen
    # identity dataclass: the run writer records it explicitly.
    setattr(model, "_s5_architecture_file_sha256", architecture_file_sha)
    return model, identity


def _unique_parameters(module: nn.Module) -> list[nn.Parameter]:
    seen: set[int] = set()
    result: list[nn.Parameter] = []
    for parameter in module.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        result.append(parameter)
    return result


def _unique_parameters_from_iterable(parameters: Any) -> list[nn.Parameter]:
    """Deduplicate a parameter iterable while keeping trainable tensors only."""

    seen: set[int] = set()
    result: list[nn.Parameter] = []
    for parameter in parameters:
        if not isinstance(parameter, nn.Parameter) or not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        result.append(parameter)
    return result


class S5HyperNet(nn.Module):
    """S5 condition runtime for Circuit, Query-only, Raw, and hybrid routes."""

    CIRCUIT_SOURCES = frozenset({"circuit", "graph", "circuit_raw_append"})
    RAW_WEIGHT_SOURCES = frozenset({"raw_weight", "circuit_raw_append"})

    def __init__(
        self,
        base_model: nn.Module,
        *,
        candidate: str = "A",
        seed: int = 42,
        parameter_generator: S5ParameterGenerator | None = None,
        query_encoder: QueryEncoder | None = None,
        circuit_encoder: nn.Module | None = None,
        feature_definition_bank: FeatureDefinitionBank | None = None,
        type_embeddings: ConditionTypeEmbeddings | None = None,
        layout: S5TargetLayout | None = None,
        condition_source: str = "circuit",
        fusion_mode: str = "append",
        circuit_residual_scale: float = 1.0,
        circuit_global_residual_scale: float = 1.0,
        circuit_block_residual_scale: float = 1.0,
        fusion_injection_blocks: tuple[int, ...] | None = None,
        circuit_lr_multipliers: Mapping[str, float] | None = None,
        raw_weight_cache: S5RawWeightCache | None = None,
        raw_fusion_mode: str = "p2e",
        circuit_input_indices: Sequence[int] | Tensor | None = None,
        validate_base: bool = True,
    ) -> None:
        super().__init__()
        candidate = str(candidate).upper()
        condition_source = str(condition_source).lower()
        if candidate not in {"A", "B", "AB"}:
            raise ValueError("S5HyperNet candidate must be A, B, or AB")
        if condition_source not in {
            "circuit",
            "graph",
            "query_only",
            "raw_weight",
            "circuit_raw_append",
        }:
            raise ValueError(f"unknown S5 condition source: {condition_source}")
        fusion_mode = str(fusion_mode).lower()
        if fusion_mode not in {
            "append",
            "f2",
            "f3i",
            "ab_joint_f3i",
            "split_f3i",
            "dual_f3i",
            "circuit_p2e",
        }:
            raise ValueError(f"unknown S5 fusion mode: {fusion_mode}")
        uses_circuit = condition_source in self.CIRCUIT_SOURCES
        uses_raw_weight = condition_source in self.RAW_WEIGHT_SOURCES
        if not uses_circuit and fusion_mode != "append":
            raise ValueError("F2/F3i fusion requires the Circuit condition source")
        if condition_source != "circuit" and candidate != "A":
            raise ValueError("Query-only, Raw-Weight, and hybrid routes use Candidate-A layout")
        if condition_source == "circuit_raw_append" and fusion_mode != "append":
            raise ValueError("Circuit+Raw hybrid requires append fusion")
        if condition_source == "graph" and fusion_mode != "append":
            raise ValueError("Graph condition currently uses append fusion")
        if condition_source == "graph" and candidate != "A":
            raise ValueError("Graph condition uses the single candidate-A PG layout")
        if fusion_mode in {
            "ab_joint_f3i",
            "split_f3i",
            "dual_f3i",
            "circuit_p2e",
        } and candidate != "AB":
            raise ValueError(f"{fusion_mode} requires candidate=AB")
        if fusion_mode == "ab_joint_f3i" and circuit_residual_scale <= 0.0:
            raise ValueError("circuit_residual_scale must be positive")
        if uses_raw_weight and raw_weight_cache is None:
            raise ValueError("Raw-Weight route requires an S5 raw-weight cache")
        raw_fusion_mode = str(raw_fusion_mode).lower()
        if raw_fusion_mode not in {"p2e", "query_cross"}:
            raise ValueError("raw_fusion_mode must be p2e or query_cross")
        if not uses_raw_weight and raw_fusion_mode != "p2e":
            raise ValueError("raw_fusion_mode=query_cross requires Raw-Weight condition source")
        if condition_source == "circuit_raw_append" and raw_fusion_mode != "p2e":
            raise ValueError("Circuit+Raw hybrid requires Safe-P2E raw fusion")
        self.candidate = candidate
        self.condition_source = condition_source
        self.fusion_mode = fusion_mode
        self.circuit_residual_scale = float(circuit_residual_scale)
        self.circuit_global_residual_scale = float(circuit_global_residual_scale)
        self.circuit_block_residual_scale = float(circuit_block_residual_scale)
        self.raw_fusion_mode = raw_fusion_mode
        self.circuit_input_indices = (
            None
            if circuit_input_indices is None
            else validate_retained_indices(circuit_input_indices)
        )
        self.circuit_input_dim = (
            VECTOR397_DIM
            if self.circuit_input_indices is None
            else len(self.circuit_input_indices)
        )
        self.feature_definition_bank = feature_definition_bank
        self.fusion_injection_blocks = (
            None
            if fusion_injection_blocks is None
            else tuple(int(index) for index in fusion_injection_blocks)
        )
        self.circuit_lr_multipliers = (
            None
            if circuit_lr_multipliers is None
            else {str(key): float(value) for key, value in circuit_lr_multipliers.items()}
        )
        self.layout = layout or build_s5_target_layout()
        if validate_base:
            self.layout.validate_model(base_model)
        self.base_model = base_model
        self.base_model.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.type_embeddings = type_embeddings or ConditionTypeEmbeddings()
            self.query_encoder = query_encoder or QueryEncoder(
                type_embeddings=self.type_embeddings
            )
            # Construct all shared modules before the candidate-specific MLP;
            # otherwise A/B would consume different RNG streams and silently
            # receive different PG initializations.
            if uses_raw_weight:
                if raw_fusion_mode == "p2e":
                    if parameter_generator is not None and not isinstance(
                        parameter_generator, S5RawParameterGenerator
                    ):
                        raise ValueError("Raw Safe-P2E route requires S5RawParameterGenerator")
                    self.parameter_generator = parameter_generator or build_s5_raw_parameter_generator(
                        cache=raw_weight_cache  # type: ignore[arg-type]
                    )
                    raw_conditioner: nn.Module | None = self.parameter_generator.model.raw_weight_conditioner
                else:
                    if parameter_generator is not None and isinstance(
                        parameter_generator, S5RawParameterGenerator
                    ):
                        raise ValueError("Raw Query-cross route requires the ordinary S5 PG")
                    self.parameter_generator = parameter_generator or build_s5_parameter_generator()
                    raw_conditioner = S5RawQueryEncoderConditioner(
                        raw_weight_cache  # type: ignore[arg-type]
                    )
            else:
                raw_conditioner = None
                if fusion_mode == "append":
                    self.parameter_generator = parameter_generator or build_s5_parameter_generator()
                else:
                    generator_mode = (
                        "f3i"
                        if fusion_mode in {"ab_joint_f3i", "split_f3i"}
                        else fusion_mode
                    )
                    generator_token_count = (
                        5
                        if fusion_mode == "split_f3i"
                        else (
                            6
                            if fusion_mode == "ab_joint_f3i"
                            else {"A": 1, "B": 5, "AB": 6}[candidate]
                        )
                    )
                    if parameter_generator is not None and not hasattr(
                        parameter_generator, "fusion_parameters"
                    ):
                        raise ValueError("F2/F3i route requires a Circuit fusion parameter generator")
                    self.parameter_generator = parameter_generator or build_s5_circuit_fusion_parameter_generator(
                        fusion_mode=generator_mode,
                        circuit_token_count=generator_token_count,
                        circuit_residual_scale=circuit_residual_scale,
                        circuit_global_residual_scale=circuit_global_residual_scale,
                        circuit_block_residual_scale=circuit_block_residual_scale,
                        fusion_injection_blocks=fusion_injection_blocks,
                    )
            if not uses_circuit:
                circuit_encoder = None
            elif circuit_encoder is None:
                if condition_source == "graph":
                    circuit_encoder = S5GraphEncoder()
                elif candidate == "A":
                    circuit_encoder = CandidateAEncoder(
                        type_embeddings=self.type_embeddings,
                        retained_indices=self.circuit_input_indices,
                    )
                elif candidate == "B":
                    circuit_encoder = CandidateBEncoder(
                        type_embeddings=self.type_embeddings,
                        retained_indices=self.circuit_input_indices,
                    )
                else:
                    circuit_encoder = CandidateABEncoder(
                        type_embeddings=self.type_embeddings,
                        retained_indices=self.circuit_input_indices,
                    )
            self.circuit_encoder = circuit_encoder

        # Candidate A has no block-addressed circuit token.  The shared
        # ConditionTypeEmbeddings object is intentionally retained so A/B
        # common initialization remains bit-identical, but A must not expose
        # the B-only block table as a trainable DDP parameter.  Leaving it
        # trainable would create a permanently unused parameter with
        # find_unused_parameters=False and stall the reducer on iteration 2.
        if candidate == "A" or not uses_circuit:
            self.type_embeddings.block.requires_grad_(False)
        if not uses_circuit or condition_source == "graph":
            self.type_embeddings.circuit.requires_grad_(False)

        expected_tokens = 19 if condition_source == "graph" else (
            ({"A": 1, "B": 5, "AB": 6}[candidate]) if uses_circuit else 0
        )
        self.circuit_token_count = expected_tokens
        self.raw_weight_conditioner = raw_conditioner
        if self.raw_weight_conditioner is not None and hasattr(
            self.raw_weight_conditioner, "restore_fp32_output_path"
        ):
            self.raw_weight_conditioner.restore_fp32_output_path()
        target_keys = self.layout.target_keys
        self.controller = GeneratedLoRAController(base_model, target_keys=target_keys)
        if self.fusion_mode != "append":
            validator = getattr(self.parameter_generator, "validate_fusion_step0_initialization", None)
            if validator is None:
                raise RuntimeError("Circuit fusion generator does not expose step-0 validation")
            validator()

    def train(self, mode: bool = True) -> "S5HyperNet":
        super().train(mode)
        # The Base is a frozen inference component even while the generators
        # are in train mode; dropout is frozen by the protocol.
        self.base_model.eval()
        return self

    def close(self) -> None:
        self.controller.close()

    def enable_gradient_checkpointing(self) -> None:
        self.parameter_generator.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        self.parameter_generator.disable_gradient_checkpointing()

    def _encode_circuit_condition(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        vector397: Tensor | None = None,
        *,
        graph_batch: GraphBatch | None = None,
        circuit_present: Tensor | None = None,
        block_permutation: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode Query and Circuit separately for non-append fusion routes."""

        if self.condition_source == "graph":
            if graph_batch is None or self.feature_definition_bank is None:
                raise ValueError("Graph route requires graph_batch and feature_definition_bank")
            if vector397 is not None or block_permutation is not None:
                raise ValueError("Graph route takes graph_batch, not vector397/block permutation")
            if graph_batch.graph_count != query_ids.shape[0]:
                raise ValueError("graph_batch count must match query batch")
        elif vector397 is None or vector397.ndim != 2 or vector397.shape[0] != query_ids.shape[0]:
            raise ValueError(
                "Circuit route requires a condition tensor with shape "
                f"[B,{self.circuit_input_dim}]"
            )
        elif vector397.shape[1] not in {
            self.circuit_input_dim,
            VECTOR397_DIM if self.circuit_input_indices is not None else self.circuit_input_dim,
        }:
            raise ValueError(
                "Circuit route condition width does not match the configured "
                f"{self.circuit_input_dim}-D projection"
            )
        if circuit_present is None:
            circuit_present = torch.ones(
                query_ids.shape[0], device=query_ids.device, dtype=torch.bool
            )
        else:
            circuit_present = circuit_present.to(device=query_ids.device, dtype=torch.bool)
        if circuit_present.shape != (query_ids.shape[0],):
            raise ValueError("circuit_present must have shape [B]")
        query_tokens = self.query_encoder(query_ids, query_mask)
        with torch.autocast(device_type=query_ids.device.type, enabled=False):
            if self.condition_source == "graph":
                circuit_tokens = self.circuit_encoder(graph_batch, self.feature_definition_bank)
            else:
                circuit_values = vector397.to(dtype=torch.float32)
            if self.condition_source != "graph" and self.candidate == "A":
                circuit_tokens = self.circuit_encoder(circuit_values)
            elif self.condition_source != "graph":
                circuit_tokens = self.circuit_encoder(
                    circuit_values,
                    block_permutation=block_permutation,
                )
        circuit_tokens = circuit_tokens.to(
            device=query_tokens.device, dtype=query_tokens.dtype
        )
        circuit_mask = circuit_present[:, None].expand(
            -1, self.circuit_token_count
        )
        if self.condition_source == "graph":
            circuit_mask = circuit_mask & graph_group_mask(graph_batch).to(query_ids.device)
        return (
            query_tokens,
            query_mask.to(device=query_ids.device, dtype=torch.bool),
            circuit_tokens,
            circuit_mask,
        )

    def condition_memory(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        vector397: Tensor | None = None,
        *,
        graph_batch: GraphBatch | None = None,
        circuit_present: Tensor | None = None,
        block_permutation: Tensor | None = None,
        raw_weight_control: str = "matched",
        raw_weight_layer_permutation: tuple[int, ...] | None = None,
        raw_weight_module_permutation: tuple[int, ...] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if query_ids.ndim != 2 or query_mask.shape != query_ids.shape:
            raise ValueError("query_ids/query_mask must have shape [B,T]")
        if self.condition_source in {"query_only", "raw_weight"}:
            query_tokens = self.query_encoder(query_ids, query_mask)
            if self.condition_source == "raw_weight" and self.raw_fusion_mode == "query_cross":
                if not isinstance(self.raw_weight_conditioner, S5RawQueryEncoderConditioner):
                    raise RuntimeError("Raw Query-cross conditioner is unavailable")
                query_tokens = self.raw_weight_conditioner(
                    query_tokens,
                    query_mask,
                    control=raw_weight_control,
                    layer_permutation=raw_weight_layer_permutation,
                    module_permutation=raw_weight_module_permutation,
                )
            return query_tokens, query_mask.to(device=query_ids.device, dtype=torch.bool)
        query_tokens, query_mask, circuit_tokens, circuit_mask = self._encode_circuit_condition(
            query_ids,
            query_mask,
            vector397,
            graph_batch=graph_batch,
            circuit_present=circuit_present,
            block_permutation=block_permutation,
        )
        if self.fusion_mode == "split_f3i":
            return append_after_valid_query(
                query_tokens,
                query_mask,
                circuit_tokens[:, :1],
                circuit_mask[:, :1],
            )
        if self.fusion_mode != "append":
            # Joint F3i/F2 keep Circuit tokens on a separate PG path; the
            # generate_lora call passes them explicitly to the fusion module.
            return query_tokens, query_mask
        return append_after_valid_query(
            query_tokens,
            query_mask,
            circuit_tokens,
            circuit_mask,
        )

    def generate_lora(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        vector397: Tensor | None = None,
        *,
        graph_batch: GraphBatch | None = None,
        circuit_present: Tensor | None = None,
        block_permutation: Tensor | None = None,
        gate_multiplier: float = 1.0,
        return_memory: bool = False,
        raw_weight_control: str = "matched",
        raw_weight_layer_permutation: tuple[int, ...] | None = None,
        raw_weight_module_permutation: tuple[int, ...] | None = None,
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], Tensor, Tensor]:
        if self.condition_source not in self.RAW_WEIGHT_SOURCES and (
            raw_weight_control != "matched"
            or raw_weight_layer_permutation is not None
            or raw_weight_module_permutation is not None
        ):
            raise ValueError("raw-weight controls require the Raw-Weight route")
        if self.condition_source in self.CIRCUIT_SOURCES and self.fusion_mode != "append":
            if vector397 is None:
                raise ValueError("Circuit fusion requires vector397")
            query_tokens, query_mask, circuit_tokens, circuit_mask = self._encode_circuit_condition(
                query_ids,
                query_mask,
                vector397,
                graph_batch=graph_batch,
                circuit_present=circuit_present,
                block_permutation=block_permutation,
            )
            if self.fusion_mode == "split_f3i":
                memory, memory_mask = append_after_valid_query(
                    query_tokens,
                    query_mask,
                    circuit_tokens[:, :1],
                    circuit_mask[:, :1],
                )
                circuit_tokens = circuit_tokens[:, 1:]
                circuit_mask = circuit_mask[:, 1:]
            else:
                memory, memory_mask = query_tokens, query_mask
        else:
            memory, memory_mask = self.condition_memory(
                query_ids,
                query_mask,
                vector397,
                graph_batch=graph_batch,
                circuit_present=circuit_present,
                block_permutation=block_permutation,
                raw_weight_control=raw_weight_control,
                raw_weight_layer_permutation=raw_weight_layer_permutation,
                raw_weight_module_permutation=raw_weight_module_permutation,
            )
            circuit_tokens = None
            circuit_mask = None
        if self.condition_source in self.RAW_WEIGHT_SOURCES and self.raw_fusion_mode == "p2e":
            raw_state = self.parameter_generator(
                memory,
                condition_attention_mask=memory_mask,
                raw_weight_control=raw_weight_control,
                raw_weight_layer_permutation=raw_weight_layer_permutation,
                raw_weight_module_permutation=raw_weight_module_permutation,
            )
        else:
            if self.fusion_mode == "append":
                raw_state = self.parameter_generator(
                    memory,
                    condition_attention_mask=memory_mask,
                )
            else:
                raw_state = self.parameter_generator(
                    memory,
                    condition_attention_mask=memory_mask,
                    circuit_tokens=circuit_tokens,
                    circuit_mask=circuit_mask,
                )
        state = scale_generated_lora_state(raw_state, gate_multiplier)
        if return_memory:
            return state, memory, memory_mask
        return state

    @staticmethod
    def _logits_from_output(output: Any) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
            return output[0]
        logits = getattr(output, "logits", None)
        if isinstance(logits, Tensor):
            return logits
        raise TypeError("S5 Base forward did not return logits")

    def base_logits(self, input_ids: Tensor) -> Tensor:
        return self._logits_from_output(self.base_model(input_ids))

    def supervised_loss(
        self,
        *,
        query_ids: Tensor,
        query_mask: Tensor,
        input_ids: Tensor,
        labels: Tensor,
        vector397: Tensor | None = None,
        graph_batch: GraphBatch | None = None,
        frozen_base_ce: Tensor,
        gate_multiplier: float,
        circuit_present: Tensor | None = None,
        block_permutation: Tensor | None = None,
        alignment_mode: str = "none",
        alignment_temperature: float = 0.1,
        alignment_projection_width: int = 128,
        task_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if self.condition_source != "circuit" and alignment_mode != "none":
            raise ValueError("Query-only/Raw-Weight routes do not expose Circuit alignment")
        if self.fusion_mode != "append" and alignment_mode != "none":
            raise ValueError("F2/F3i primary routes do not expose alignment loss")
        generated = self.generate_lora(
            query_ids,
            query_mask,
            vector397,
            graph_batch=graph_batch,
            circuit_present=circuit_present,
            block_permutation=block_permutation,
            gate_multiplier=gate_multiplier,
            return_memory=alignment_mode != "none",
        )
        alignment: dict[str, Tensor] | None = None
        if alignment_mode == "none":
            if not isinstance(generated, dict):
                raise RuntimeError("non-alignment S5 generation returned condition memory")
            state = generated
        else:
            if not isinstance(generated, tuple) or len(generated) != 3:
                raise RuntimeError("alignment S5 generation omitted condition memory")
            state, memory, _memory_mask = generated
            alignment = query_circuit_alignment_stats(
                memory,
                query_mask,
                circuit_token_count=self.circuit_token_count,
                mode=alignment_mode,
                temperature=alignment_temperature,
                projection_width=alignment_projection_width,
                task_ids=task_ids,
            )
        with self.controller.activate(state):
            logits = self._logits_from_output(self.base_model(input_ids))
        loss_sum, event_ce, token_count = causal_loss_stats_per_example(logits, labels)
        hinge = base_preservation_hinge(event_ce, frozen_base_ce)
        effective_delta = effective_delta_l2_per_example(state)
        zero = event_ce.sum() * 0.0
        return {
            "loss_sum": loss_sum.sum(),
            "per_event_loss_sum": loss_sum,
            "per_event_ce": event_ce,
            "base_preservation_hinge": hinge,
            "effective_delta_l2": effective_delta,
            "token_count": token_count.sum(),
            "per_event_token_count": token_count,
            "logits": logits,
            "state": state,
            "alignment_loss_sum": (
                zero if alignment is None else alignment["loss_sum"]
            ),
            "alignment_event_count": (
                torch.zeros((), device=event_ce.device, dtype=torch.long)
                if alignment is None
                else alignment["event_count"]
            ),
            "alignment_positive_similarity_sum": (
                zero if alignment is None else alignment["positive_similarity_sum"]
            ),
            "alignment_negative_similarity_sum": (
                zero if alignment is None else alignment["negative_similarity_sum"]
            ),
            "alignment_negative_similarity_count": (
                torch.zeros((), device=event_ce.device, dtype=torch.long)
                if alignment is None
                else alignment["negative_similarity_count"]
            ),
            "alignment_retrieval_correct_sum": (
                zero if alignment is None else alignment["retrieval_correct_sum"]
            ),
            "alignment_same_task_negative_similarity_sum": (
                zero
                if alignment is None
                else alignment["same_task_negative_similarity_sum"]
            ),
            "alignment_same_task_negative_similarity_count": (
                torch.zeros((), device=event_ce.device, dtype=torch.long)
                if alignment is None
                else alignment["same_task_negative_similarity_count"]
            ),
        }

    def forward(
        self,
        *,
        query_ids: Tensor,
        query_mask: Tensor,
        input_ids: Tensor,
        labels: Tensor,
        vector397: Tensor | None = None,
        graph_batch: GraphBatch | None = None,
        frozen_base_ce: Tensor,
        gate_multiplier: float,
        circuit_present: Tensor | None = None,
        block_permutation: Tensor | None = None,
        alignment_mode: str = "none",
        alignment_temperature: float = 0.1,
        alignment_projection_width: int = 128,
        task_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Expose the training path through ``forward`` for DDP correctness.

        Calling ``DistributedDataParallel.module.supervised_loss`` directly
        bypasses DDP's forward bookkeeping.  The training entrypoint invokes
        this method through the DDP wrapper instead.
        """

        return self.supervised_loss(
            query_ids=query_ids,
            query_mask=query_mask,
            input_ids=input_ids,
            labels=labels,
            vector397=vector397,
            graph_batch=graph_batch,
            frozen_base_ce=frozen_base_ce,
            gate_multiplier=gate_multiplier,
            circuit_present=circuit_present,
            block_permutation=block_permutation,
            alignment_mode=alignment_mode,
            alignment_temperature=alignment_temperature,
            alignment_projection_width=alignment_projection_width,
            task_ids=task_ids,
        )

    def common_state_dict(self) -> OrderedDict[str, Tensor]:
        """Return the candidate-independent initialization state only once."""

        state: OrderedDict[str, Tensor] = OrderedDict()
        state["type_embeddings.query"] = self.type_embeddings.query.detach().cpu().clone()
        state["type_embeddings.circuit"] = self.type_embeddings.circuit.detach().cpu().clone()
        state["type_embeddings.block"] = self.type_embeddings.block.detach().cpu().clone()
        for key, value in self.query_encoder.state_dict().items():
            if key.startswith("type_embeddings."):
                continue
            state[f"query_encoder.{key}"] = value.detach().cpu().clone()
        for key, value in self.parameter_generator.state_dict().items():
            if key.startswith("model.raw_weight_conditioner."):
                continue
            state[f"parameter_generator.{key}"] = value.detach().cpu().clone()
        return state

    def circuit_state_dict(self) -> OrderedDict[str, Tensor]:
        if self.circuit_encoder is None:
            return OrderedDict()
        return OrderedDict(
            (key, value.detach().cpu().clone())
            for key, value in self.circuit_encoder.state_dict().items()
            if not key.startswith("type_embeddings.")
        )

    def trainable_state_dict(self) -> OrderedDict[str, Tensor]:
        state = self.common_state_dict()
        state.update((f"circuit_encoder.{key}", value) for key, value in self.circuit_state_dict().items())
        if self.raw_weight_conditioner is not None:
            state.update(
                (f"raw_weight_conditioner.{key}", value.detach().cpu().clone())
                for key, value in self.raw_weight_conditioner.state_dict().items()
            )
        return state

    def load_common_state_dict(
        self,
        state: Mapping[str, Tensor],
        *,
        allow_mode_specific_fusion_init: bool = False,
    ) -> None:
        expected = self.common_state_dict()
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        if allow_mode_specific_fusion_init:
            missing = [
                key
                for key in missing
                if not key.startswith("parameter_generator.model.fusion_")
            ]
            extra = [
                key
                for key in extra
                if not key.startswith("parameter_generator.model.fusion_")
            ]
        if missing or extra:
            raise ValueError(f"common S5 init keys changed: missing={missing[:3]} extra={extra[:3]}")
        self.type_embeddings.query.data.copy_(state["type_embeddings.query"])
        self.type_embeddings.circuit.data.copy_(state["type_embeddings.circuit"])
        self.type_embeddings.block.data.copy_(state["type_embeddings.block"])
        query_state = {
            key[len("query_encoder.") :]: value
            for key, value in state.items()
            if key.startswith("query_encoder.")
        }
        query_state.update(
            {
                "type_embeddings.query": state["type_embeddings.query"],
                "type_embeddings.circuit": state["type_embeddings.circuit"],
                "type_embeddings.block": state["type_embeddings.block"],
            }
        )
        pg_state = {
            key[len("parameter_generator.") :]: value
            for key, value in state.items()
            if key.startswith("parameter_generator.")
        }
        if any(key.startswith("model.raw_weight_conditioner.") for key in pg_state):
            raise ValueError("common state must not contain raw-weight conditioner keys")
        self.query_encoder.load_state_dict(query_state, strict=True)
        missing, unexpected = self.parameter_generator.load_state_dict(pg_state, strict=False)
        raw_conditioner = getattr(self, "raw_weight_conditioner", None)
        allowed_missing = set()
        if raw_conditioner is not None and any(
            key.startswith("model.raw_weight_conditioner.")
            for key in self.parameter_generator.state_dict()
        ):
            allowed_missing.update(
                f"model.raw_weight_conditioner.{key}"
                for key in raw_conditioner.state_dict()
            )
        if allow_mode_specific_fusion_init:
            allowed_missing.update(
                key
                for key in self.parameter_generator.state_dict()
                if key.startswith("model.fusion_")
            )
        if unexpected or set(missing) != allowed_missing:
            raise ValueError(
                f"parameter-generator common keys changed: missing={missing[:3]} unexpected={unexpected[:3]}"
            )

    def load_trainable_state_dict(self, state: Mapping[str, Tensor]) -> None:
        common = {
            key: value
            for key, value in state.items()
            if not key.startswith("circuit_encoder.")
            and not key.startswith("raw_weight_conditioner.")
        }
        self.load_common_state_dict(common)
        circuit = {
            key[len("circuit_encoder.") :]: value
            for key, value in state.items()
            if key.startswith("circuit_encoder.")
        }
        if self.condition_source in self.CIRCUIT_SOURCES:
            self.load_circuit_state_dict(circuit)
        elif circuit:
            raise ValueError("non-Circuit route checkpoint contains Circuit keys")
        raw = {
            key[len("raw_weight_conditioner.") :]: value
            for key, value in state.items()
            if key.startswith("raw_weight_conditioner.")
        }
        if self.condition_source in self.RAW_WEIGHT_SOURCES:
            if self.raw_weight_conditioner is None or set(raw) != set(self.raw_weight_conditioner.state_dict()):
                raise ValueError("Raw-Weight checkpoint keys changed")
            self.raw_weight_conditioner.load_state_dict(raw, strict=True)
        elif raw:
            raise ValueError("non-Raw route checkpoint contains raw-weight keys")

    def load_circuit_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Load a candidate init, slicing a legacy 397-D state if needed."""

        if self.circuit_encoder is None:
            raise RuntimeError("Circuit route has no circuit encoder")
        circuit = (
            compact_circuit_state_dict(state, self.circuit_input_indices)
            if self.circuit_input_indices is not None
            else OrderedDict(state)
        )
        expected = self.circuit_state_dict()
        if set(circuit) != set(expected):
            raise ValueError("candidate-specific S5 circuit init keys changed")
        missing, unexpected = self.circuit_encoder.load_state_dict(
            circuit, strict=False
        )
        if unexpected or any(
            not key.startswith("type_embeddings.") for key in missing
        ):
            raise ValueError(
                "candidate-specific S5 circuit init tensors do not match the "
                "configured projection"
            )

    def parameter_groups(
        self,
        *,
        pg_lr: float,
        query_lr_multiplier: float = 0.5,
        circuit_lr_multiplier: float = 2.0,
        circuit_lr_multipliers: Mapping[str, float] | None = None,
        raw_encoder_lr: float | None = None,
        raw_output_lr: float | None = None,
        weight_decay: float = 0.0,
    ) -> list[dict[str, Any]]:
        partitions = self._parameter_partitions()
        groups = [
            {
                "params": partitions["pg"],
                "lr": float(pg_lr),
                "weight_decay": float(weight_decay),
                "name": "pg",
            },
            {
                "params": partitions["query"],
                "lr": float(pg_lr) * float(query_lr_multiplier),
                "weight_decay": float(weight_decay),
                "name": "query",
            },
        ]
        lr_map = self.circuit_lr_multipliers if circuit_lr_multipliers is None else {
            str(key): float(value) for key, value in circuit_lr_multipliers.items()
        }
        if lr_map is not None and any(
            partitions.get(name)
            for name in ("circuit_global", "circuit_block", "circuit_fusion")
        ):
            for name, key in (
                ("circuit_global", "global"),
                ("circuit_block", "block"),
                ("circuit_fusion", "fusion"),
            ):
                if partitions.get(name):
                    if key not in lr_map:
                        raise ValueError(f"Circuit LR map is missing {key!r}")
                    groups.append(
                        {
                            "params": partitions[name],
                            "lr": float(pg_lr) * float(lr_map[key]),
                            "weight_decay": float(weight_decay),
                            "name": name,
                        }
                    )
        elif partitions.get("circuit"):
            groups.append(
                {
                    "params": partitions["circuit"],
                    "lr": float(pg_lr) * float(circuit_lr_multiplier),
                    "weight_decay": float(weight_decay),
                    "name": "circuit",
                }
            )
        elif partitions.get("circuit_global"):
            groups.append(
                {
                    "params": partitions["circuit_global"],
                    "lr": float(pg_lr) * float(circuit_lr_multiplier),
                    "weight_decay": float(weight_decay),
                    "name": "circuit_global",
                }
            )
        if partitions.get("raw_encoder"):
            if raw_encoder_lr is None or raw_output_lr is None:
                raise ValueError("Raw-Weight optimizer requires raw_encoder_lr/raw_output_lr")
            groups.extend(
                [
                    {
                        "params": partitions["raw_encoder"],
                        "lr": float(raw_encoder_lr),
                        "weight_decay": float(weight_decay),
                        "name": "raw_encoder",
                    },
                    {
                        "params": partitions["raw_output"],
                        "lr": float(raw_output_lr),
                        "weight_decay": float(weight_decay),
                        "name": "raw_output",
                    },
                ]
            )
        grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
        trainable_ids = [id(parameter) for parameter in self.parameters() if parameter.requires_grad]
        if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != set(trainable_ids):
            raise RuntimeError(
                "S5 optimizer groups do not cover each trainable parameter exactly once"
            )
        return groups

    def _parameter_partitions(self) -> dict[str, list[nn.Parameter]]:
        """Partition shared type embeddings by semantic Query/Circuit ownership."""

        circuit_type_ids = {
            id(parameter)
            for parameter in (
                self.type_embeddings.circuit,
                self.type_embeddings.block,
            )
            if parameter.requires_grad
        }
        query_params = [
            parameter
            for parameter in _unique_parameters(self.query_encoder)
            if id(parameter) not in circuit_type_ids
        ]
        query_ids = {id(item) for item in query_params}
        circuit_params = (
            [
                parameter
                for parameter in _unique_parameters(self.circuit_encoder)
                if id(parameter) not in query_ids
            ]
            if self.circuit_encoder is not None
            else []
        )
        fusion_params = []
        fusion_getter = getattr(self.parameter_generator, "fusion_parameters", None)
        if fusion_getter is not None:
            fusion_params = list(fusion_getter())
        circuit_global_params: list[nn.Parameter] = []
        circuit_block_params: list[nn.Parameter] = []
        global_getter = getattr(self.circuit_encoder, "global_parameters", None)
        block_getter = getattr(self.circuit_encoder, "block_parameters", None)
        if global_getter is not None and block_getter is not None:
            circuit_global_params = [
                parameter for parameter in _unique_parameters_from_iterable(global_getter())
                if id(parameter) not in query_ids
            ]
            circuit_block_params = [
                parameter for parameter in _unique_parameters_from_iterable(block_getter())
                if id(parameter) not in query_ids
            ]
            circuit_params = _unique_parameters_from_iterable(
                [*circuit_global_params, *circuit_block_params]
            )
        circuit_ids = {id(parameter) for parameter in circuit_params}
        circuit_params.extend(
            parameter for parameter in fusion_params if id(parameter) not in circuit_ids
        )
        raw_params = _unique_parameters(self.raw_weight_conditioner) if self.raw_weight_conditioner is not None else []
        raw_ids = {id(parameter) for parameter in raw_params}
        fusion_ids = {id(parameter) for parameter in fusion_params}
        pg_params = [
            parameter
            for parameter in _unique_parameters(self.parameter_generator)
            if id(parameter) not in raw_ids and id(parameter) not in fusion_ids
        ]
        if self.raw_weight_conditioner is None:
            if self.condition_source == "query_only":
                return {"pg": pg_params, "query": query_params}
            return {
                "pg": pg_params,
                "query": query_params,
                "circuit": circuit_params,
                "circuit_global": circuit_global_params,
                "circuit_block": circuit_block_params,
                "circuit_fusion": fusion_params,
            }
        output_ids = {id(parameter) for parameter in _unique_parameters(self.raw_weight_conditioner.output_projection)}
        result = {
            "pg": pg_params,
            "query": query_params,
            "raw_encoder": [parameter for parameter in raw_params if id(parameter) not in output_ids],
            "raw_output": [parameter for parameter in raw_params if id(parameter) in output_ids],
        }
        if self.condition_source == "circuit_raw_append":
            # Candidate-A has one global Circuit token.  Give that branch the
            # same audited semantic name used by global AB encoders, while
            # preserving the historical ``circuit`` name for Circuit-only A.
            result["circuit_global"] = circuit_params
        return result

    def validate_step0_initialization(self) -> None:
        """Assert the exact zero-B Generated-LoRA starting contract."""

        output_b = self.parameter_generator.model.proj_out_B
        if not bool(torch.equal(output_b.weight.detach(), torch.zeros_like(output_b.weight))):
            raise ValueError("S5 PG B output weight is not zero at initialization")
        if not bool(torch.equal(output_b.bias.detach(), torch.zeros_like(output_b.bias))):
            raise ValueError("S5 PG B output bias is not zero at initialization")
        if self.fusion_mode != "append":
            validator = getattr(self.parameter_generator, "validate_fusion_step0_initialization", None)
            if validator is None:
                raise ValueError("S5 Circuit fusion generator does not expose step-0 validation")
            validator()
        if self.raw_weight_conditioner is not None:
            output = self.raw_weight_conditioner.output_projection.weight.detach()
            if not bool(torch.equal(output, torch.zeros_like(output))):
                raise ValueError("S5 Raw-Weight output projection is not zero at initialization")

    def group_gradient_norms(self) -> dict[str, float]:
        groups = self._parameter_partitions()
        result: dict[str, float] = {}
        for name, parameters in groups.items():
            values = [
                parameter.grad.detach().float().pow(2).sum()
                for parameter in parameters
                if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            ]
            result[name] = float(torch.sqrt(torch.stack(values).sum()).item()) if values else 0.0
        return result

    def group_parameter_counts(self) -> dict[str, int]:
        """Return exact trainable scalar counts for every audited partition."""

        partitions = self._parameter_partitions()
        if any(partitions.get(name) for name in ("circuit_global", "circuit_block", "circuit_fusion")):
            partitions = {
                name: parameters
                for name, parameters in partitions.items()
                if name != "circuit"
            }
        return {
            name: int(sum(parameter.numel() for parameter in parameters))
            for name, parameters in partitions.items()
        }


def save_common_initialization(
    destination: str | Path,
    model: S5HyperNet,
    *,
    base_identity: S5BaseIdentity | Mapping[str, Any],
    seed: int,
    layout_sha256: str | None = None,
) -> dict[str, Any]:
    """Write a canonical common-init tensor snapshot and identity manifest."""

    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "common_state.pt"
    manifest_path = root / "manifest.json"
    if state_path.exists() or manifest_path.exists():
        return validate_common_initialization(
            root,
            expected_base_identity=base_identity,
            expected_seed=seed,
            expected_layout_sha256=layout_sha256,
        )
    state = model.common_state_dict()
    identity = state_identity(state)
    torch.save(state, state_path)
    base_record = (
        dict(base_identity) if isinstance(base_identity, Mapping) else base_identity.__dict__.copy()
    )
    manifest = {
        "format": "s5_hypernet_common_initialization_v1",
        "status": "complete",
        "protocol_id": "s5_hypernet_vector397_ab_fusion_v1",
        "seed": int(seed),
        "layout_sha256": layout_sha256 or model.layout.layout_sha256,
        "base_identity": base_record,
        "state_path": state_path.name,
        "state_sha256": sha256_file(state_path),
        **identity,
    }
    manifest["manifest_sha256"] = semantic_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validate_common_initialization(
        root,
        expected_base_identity=base_identity,
        expected_seed=seed,
        expected_layout_sha256=layout_sha256 or model.layout.layout_sha256,
    )


def save_candidate_initialization(
    destination: str | Path,
    model: S5HyperNet,
    *,
    base_identity: S5BaseIdentity | Mapping[str, Any],
    seed: int,
    layout_sha256: str | None = None,
) -> dict[str, Any]:
    """Write the reproducible candidate-specific circuit encoder snapshot."""

    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "circuit_state.pt"
    manifest_path = root / "manifest.json"
    if state_path.exists() or manifest_path.exists():
        return validate_candidate_initialization(
            root,
            expected_base_identity=base_identity,
            expected_seed=seed,
            expected_candidate=model.candidate,
            expected_layout_sha256=layout_sha256,
        )
    state = model.circuit_state_dict()
    identity = state_identity(state)
    torch.save(state, state_path)
    base_record = (
        dict(base_identity)
        if isinstance(base_identity, Mapping)
        else base_identity.__dict__.copy()
    )
    manifest = {
        "format": "s5_hypernet_candidate_initialization_v1",
        "status": "complete",
        "protocol_id": "s5_hypernet_vector397_ab_fusion_v1",
        "candidate": model.candidate,
        "seed": int(seed),
        "layout_sha256": layout_sha256 or model.layout.layout_sha256,
        "base_identity": base_record,
        "state_path": state_path.name,
        "state_sha256": sha256_file(state_path),
        **identity,
    }
    manifest["manifest_sha256"] = semantic_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validate_candidate_initialization(
        root,
        expected_base_identity=base_identity,
        expected_seed=seed,
        expected_candidate=model.candidate,
        expected_layout_sha256=layout_sha256 or model.layout.layout_sha256,
    )


def validate_candidate_initialization(
    root: str | Path,
    *,
    expected_base_identity: Mapping[str, Any] | None = None,
    expected_seed: int | None = None,
    expected_candidate: str | None = None,
    expected_layout_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed if a candidate-specific init artifact was changed."""

    base = resolve_common_initialization_root(root)
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.pop("manifest_sha256", None)
    if recorded != semantic_sha256(manifest):
        raise ValueError("candidate-init manifest identity changed")
    manifest["manifest_sha256"] = recorded
    if manifest.get("format") != "s5_hypernet_candidate_initialization_v1":
        raise ValueError("candidate-init format identity changed")
    if manifest.get("status") != "complete":
        raise ValueError("candidate-init is not complete")
    if manifest.get("protocol_id") != "s5_hypernet_vector397_ab_fusion_v1":
        raise ValueError("candidate-init protocol identity changed")
    if expected_seed is not None and int(manifest.get("seed", -1)) != int(expected_seed):
        raise ValueError("candidate-init seed identity changed")
    if expected_candidate is not None and str(manifest.get("candidate")) != str(expected_candidate).upper():
        raise ValueError("candidate-init candidate identity changed")
    if expected_layout_sha256 is not None and manifest.get("layout_sha256") != expected_layout_sha256:
        raise ValueError("candidate-init target layout identity changed")
    if expected_base_identity is not None:
        actual_base = dict(manifest.get("base_identity", {}))
        if not base_content_identity_matches(actual_base, expected_base_identity):
            raise ValueError("candidate-init Base content identity changed")
    state_path = base / str(manifest["state_path"])
    if sha256_file(state_path) != manifest["state_sha256"]:
        raise ValueError("candidate-init state checksum changed")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if state_identity(state)["aggregate_sha256"] != manifest["aggregate_sha256"]:
        raise ValueError("candidate-init tensor identity changed")
    return manifest


def resolve_common_initialization_root(path: str | Path) -> Path:
    """Normalize a config value that may name a directory or manifest file."""

    value = Path(path).expanduser().resolve()
    if value.exists() and value.is_file():
        return value.parent
    if value.suffix.lower() == ".json":
        return value.parent
    return value


def validate_common_initialization(
    root: str | Path,
    *,
    expected_base_identity: Mapping[str, Any] | None = None,
    expected_seed: int | None = None,
    expected_layout_sha256: str | None = None,
) -> dict[str, Any]:
    base = resolve_common_initialization_root(root)
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest.pop("manifest_sha256", None)
    if recorded != semantic_sha256(manifest):
        raise ValueError("common-init manifest identity changed")
    manifest["manifest_sha256"] = recorded
    if manifest.get("format") != "s5_hypernet_common_initialization_v1":
        raise ValueError("common-init format identity changed")
    if manifest.get("status") != "complete":
        raise ValueError("common-init is not complete")
    if manifest.get("protocol_id") != "s5_hypernet_vector397_ab_fusion_v1":
        raise ValueError("common-init protocol identity changed")
    if expected_seed is not None and int(manifest.get("seed", -1)) != int(expected_seed):
        raise ValueError("common-init seed identity changed")
    if expected_layout_sha256 is not None and manifest.get("layout_sha256") != expected_layout_sha256:
        raise ValueError("common-init target layout identity changed")
    if expected_base_identity is not None:
        actual_base = dict(manifest.get("base_identity", {}))
        if not base_content_identity_matches(actual_base, expected_base_identity):
            raise ValueError("common-init Base content identity changed")
    state_path = base / str(manifest["state_path"])
    if sha256_file(state_path) != manifest["state_sha256"]:
        raise ValueError("common-init state checksum changed")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if state_identity(state)["aggregate_sha256"] != manifest["aggregate_sha256"]:
        raise ValueError("common-init tensor identity changed")
    return manifest


__all__ = [
    "S5HyperNet",
    "load_s5_base_model",
    "model_state_identity",
    "resolve_common_initialization_root",
    "save_candidate_initialization",
    "save_common_initialization",
    "sha256_file",
    "state_identity",
    "tensor_sha256",
    "validate_candidate_initialization",
    "validate_common_initialization",
]
