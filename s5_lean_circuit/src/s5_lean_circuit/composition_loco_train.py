"""Distributed trainer for the frozen composition leave-one-out SFT round."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F

from .circuit.model_io import load_tokenizer
from .composition_loco import (
    ATOMIC_SAMPLES_PER_TASK,
    ATOMIC_TASKS,
    COMPOSITION_TASKS,
    STAGE_A_BASE_IDENTITY,
    GLOBAL_BATCH_SIZE,
    TOTAL_STEPS,
    VALIDATION_STEPS,
    CompositionFoldDataset,
    load_eval_bank,
    sha256_file,
)
from .composition_loco_config import (
    CIRCUIT_SWEEP40_PROTOCOL_ID,
    CompositionLOCOConfig,
    DIRECT_LR32_PROTOCOL_ID,
    MANUAL160_PROTOCOL_ID,
    QUERY_CIRCUIT_PROTOCOL_ID,
    QUERY_WEIGHT_PROTOCOL_ID,
    WEIGHT_CIRCUIT_PROTOCOL_ID,
    NESTED_QUERY_WEIGHT_PROTOCOL_ID,
    LR_HALF_PROTOCOL_ID,
    RECREATED_INPUTS_PROTOCOL_ID,
    ORIGINAL_LR_RECREATED_PROTOCOL_ID,
    DIRECT_RETENTION32_PROTOCOL_ID,
    QUERY_RETENTION20_PROTOCOL_ID,
    RAW_RETENTION20_PROTOCOL_ID,
    load_config,
)
from .composition_loco_circuit_retention20_spec import (
    CIRCUIT_RETENTION20_PROTOCOL_ID,
)
from .composition_loco_vector422_spec import VECTOR422_PROTOCOL_ID
from .composition_loco_vector422 import (
    expand_circuit_input_422, load_vector422_eval_bank, vector422_dataset_view,
)
from .composition_loco_direct_lr32_spec import direct_lr32_profile_values
from .composition_loco_circuit_sweep40_spec import (
    circuit_sweep40_optimizer_peak_lrs,
)
from .composition_loco_manual160_spec import profile_values
from .composition_loco_query_weight_spec import profile_values as query_weight_profile_values
from .composition_loco_query_circuit_spec import profile_values as query_circuit_profile_values
from .composition_loco_weight_circuit_spec import profile_values as weight_circuit_profile_values
from .composition_loco_t_nested_and_query_weight_spec import profile_values as nested_query_weight_profile_values
from .composition_loco_retention import (
    DIRECT_RETENTION4_PROTOCOL_ID,
    RETENTION_ATOMIC_TASK_IDS,
    RETENTION_GLOBAL_BATCH_SIZE,
    RETENTION_KD_TEMPERATURE,
    RetentionBatchSampler,
    atomic_kd_loss_sum,
    atomic_response_token_mask,
    build_task_index_pools,
    collate_retention_batch,
    load_atomic_replay_dataset,
    validate_retention_pool_cardinality,
)
from .composition_loco_resume import archive_replay_suffix, load_verified_resume, validate_optimizer_layout
from .composition_loco_cache_repair import inspect_raw_weight_cache
from .hypernet.losses import scale_microbatch_objective_for_ddp
from .hypernet.losses import causal_loss_stats_per_example
from .hypernet.raw_weight import S5RawWeightCache
from .hypernet.config import S5BaseIdentity
from .hypernet.config import base_content_identity_matches
from .hypernet.runtime import S5HyperNet, load_s5_base_model
from .modeling import build_model


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")


def _rank_world() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("composition LOCO distributed training requires CUDA")
        torch.cuda.set_device(local)
        if not dist.is_initialized():
            dist.init_process_group("nccl", init_method="env://")
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    return rank, world, local, device


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _all_reduce(value: Tensor) -> Tensor:
    result = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def _seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _gather_rng_states(world: int) -> list[dict[str, Any]]:
    state = _rng_state()
    if not (dist.is_available() and dist.is_initialized()):
        return [state]
    values: list[dict[str, Any] | None] = [None] * int(world)
    dist.all_gather_object(values, state)
    if any(value is None for value in values):
        raise RuntimeError("composition LOCO RNG gather returned an incomplete list")
    return [value for value in values if value is not None]


def _autocast(device: torch.device, dtype: torch.dtype):
    return torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()


def _logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    value = getattr(output, "logits", None)
    if isinstance(value, Tensor):
        return value
    raise TypeError("S5 model output does not contain logits")


def _tokenizer_special_ids(tokenizer: Any) -> tuple[int, ...]:
    """Read SentencePiece special ids without assuming a wrapper API."""

    values: set[int] = set()
    for name in ("pad_id", "unk_id", "bos_id", "eos_id"):
        getter = getattr(tokenizer, name, None)
        if callable(getter):
            value = int(getter())
            if value >= 0:
                values.add(value)
    return tuple(sorted(values))


def _load_state(path: Path) -> tuple[Mapping[str, Tensor], Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"initial checkpoint is not a mapping: {path}")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        # A plain state_dict is accepted only for Direct-SFT, whose historical
        # best_model.pt artifacts use both envelope spellings.
        if payload and all(isinstance(key, str) for key in payload):
            state = payload
        else:
            raise ValueError(f"initial checkpoint has no model_state_dict: {path}")
    return state, payload


def _verify_initial_checkpoint(path: Path, config: CompositionLOCOConfig) -> None:
    expected = config.initialization.get("checkpoint_sha256")
    if expected is not None and sha256_file(path) != str(expected):
        raise ValueError("composition LOCO initial checkpoint SHA256 changed")


def _load_base_ce(config: CompositionLOCOConfig, dataset: CompositionFoldDataset) -> Tensor:
    source = _resolve(str(config.data["base_ce_path"]))
    sidecar = source.with_suffix(source.suffix + ".json")
    if not source.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"composition LOCO Base-CE artifact is missing: {source}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
        metadata.get("format") != "s5_composition_loco_base_ce_v1"
        or metadata.get("status") != "complete"
        or int(metadata.get("event_count", -1)) != 1_500_000
        or metadata.get("cache_identity_sha256") != dataset.cache_metadata["identity_sha256"]
        or metadata.get("cache_payload_sha256") != dataset.cache_metadata["payload_sha256"]
        or metadata.get("payload_path") != source.name
        or metadata.get("payload_sha256") != sha256_file(source)
        or metadata.get("objective")
        != "per_event_full_sequence_nonpadding_next_token_ce_fp32_v1"
        or not base_content_identity_matches(
            metadata.get("base_identity", {}), _base_identity(config)
        )
    ):
        raise ValueError("composition LOCO Base-CE identity changed")
    values = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(values, Tensor) or values.dtype != torch.float32 or tuple(values.shape) != (1_500_000,):
        raise ValueError("composition LOCO Base-CE payload shape changed")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("composition LOCO Base-CE payload is non-finite")
    return values


@torch.no_grad()
def _mixed_frozen_base_ce(
    core: S5HyperNet,
    batch: Mapping[str, Tensor],
    composition_base_ce: Tensor,
    *,
    device: torch.device,
) -> Tensor:
    """Resolve Base-CE for a mixed composition/atomic Circuit batch.

    The existing 1.5M composition CE artifact is indexed by canonical
    composition rows.  Atomic replay has a different 1.2M index namespace and
    does not have a compatible local artifact in the historical protocol, so
    its frozen Base CE is computed on the atomic sub-batch.  Keeping this
    distinction explicit prevents atomic indices from accidentally selecting
    unrelated composition CE values.
    """

    is_atomic = batch.get("is_atomic")
    composition_indices = batch.get("composition_global_indices")
    if is_atomic is None or composition_indices is None:
        raise ValueError("mixed retention batch lacks source-qualified CE metadata")
    is_atomic = is_atomic.to(device=device, dtype=torch.bool)
    composition_indices = composition_indices.to(device=device, dtype=torch.long)
    size = int(is_atomic.numel())
    result = torch.empty((size,), dtype=torch.float32, device=device)
    composition_rows = torch.nonzero(~is_atomic, as_tuple=False).flatten()
    if composition_rows.numel():
        indices = composition_indices.index_select(0, composition_rows)
        if bool((indices < 0).any()):
            raise ValueError("composition retention row has no canonical CE index")
        result.index_copy_(
            0,
            composition_rows,
            composition_base_ce.index_select(0, indices.detach().cpu())
            .to(device=device, dtype=torch.float32),
        )
    atomic_rows = torch.nonzero(is_atomic, as_tuple=False).flatten()
    if atomic_rows.numel():
        atomic_input = batch["input_ids"].index_select(0, atomic_rows)
        atomic_labels = batch["labels"].index_select(0, atomic_rows)
        with _autocast(device, torch.bfloat16):
            logits = core.base_logits(atomic_input)
        _loss_sum, event_ce, _token_count = causal_loss_stats_per_example(
            logits, atomic_labels
        )
        result.index_copy_(0, atomic_rows, event_ce.detach().float())
    if not bool(torch.isfinite(result).all()):
        raise ValueError("mixed retention frozen Base CE is non-finite")
    return result


def _base_identity(config: CompositionLOCOConfig) -> S5BaseIdentity:
    base = dict(config.base)
    return S5BaseIdentity(
        checkpoint_path=str(_resolve(str(base["checkpoint_path"]))),
        architecture_config_path=str(_resolve(str(base["architecture_config_path"]))),
        checkpoint_sha256=str(base.get("checkpoint_sha256", S5BaseIdentity.__dataclass_fields__["checkpoint_sha256"].default)),
        model_state_sha256=str(base.get("model_state_sha256", S5BaseIdentity.__dataclass_fields__["model_state_sha256"].default)),
        config_sha256=str(base.get("config_sha256", S5BaseIdentity.__dataclass_fields__["config_sha256"].default)),
        step=int(base.get("step", 38_000)),
    )


def _architecture(config: CompositionLOCOConfig) -> dict[str, Any]:
    value = json.loads(_resolve(str(config.base["architecture_config_path"])).read_text(encoding="utf-8"))
    if isinstance(value.get("model"), Mapping):
        return dict(value["model"])
    return dict(value)


def _load_direct(config: CompositionLOCOConfig, device: torch.device) -> nn.Module:
    model = build_model(_architecture(config)).to(device)
    init_path = _resolve(str(config.initialization["checkpoint_path"]))
    _verify_initial_checkpoint(init_path, config)
    state, payload = _load_state(init_path)
    expected_step = int(config.initialization.get("step", -1))
    expected_config_sha = config.initialization.get("checkpoint_config_sha256")
    if expected_config_sha is not None and payload.get("config_sha256") not in {
        None,
        str(expected_config_sha),
    }:
        raise ValueError("Direct-SFT initial checkpoint config identity changed")
    if expected_step > 0 and int(payload.get("step", payload.get("global_step", expected_step))) != expected_step:
        raise ValueError("Direct-SFT initial checkpoint step changed")
    model.load_state_dict(state, strict=True)
    model.config.grad_checkpointing = False
    return model


def _load_hypernet(config: CompositionLOCOConfig, device: torch.device) -> S5HyperNet:
    if config.protocol_id == RAW_RETENTION20_PROTOCOL_ID:
        from .composition_loco_ood_raw_cache import check_recreated_raw_cache, load_raw_hypernet_for_recreated_inputs
        receipt = check_recreated_raw_cache(config, Path.cwd())
        return load_raw_hypernet_for_recreated_inputs(config, device, receipt, root=Path.cwd())
    if config.protocol_id in {
        LR_HALF_PROTOCOL_ID, RECREATED_INPUTS_PROTOCOL_ID, ORIGINAL_LR_RECREATED_PROTOCOL_ID,
    } and config.route == "raw_weight":
        # Fresh recovery-wave initialization requires the explicit rebuilt-cache
        # provenance gate, not the historical training-resume permission.
        from .composition_loco_ood_raw_cache import (
            check_raw_cache_for_ood,
            check_recreated_raw_cache,
            load_raw_hypernet_for_ood,
            load_raw_hypernet_for_recreated_inputs,
        )
        if config.protocol_id in {RECREATED_INPUTS_PROTOCOL_ID, ORIGINAL_LR_RECREATED_PROTOCOL_ID}:
            receipt = check_recreated_raw_cache(config, Path.cwd())
            return load_raw_hypernet_for_recreated_inputs(config, device, receipt, root=Path.cwd())
        receipt = check_raw_cache_for_ood(config, Path.cwd())
        return load_raw_hypernet_for_ood(config, device, receipt, root=Path.cwd())
    base, base_identity = load_s5_base_model(
        _resolve(str(config.base["architecture_config_path"])),
        _resolve(str(config.base["checkpoint_path"])),
        device=device,
        dtype=torch.bfloat16,
        expected_identity=_base_identity(config),
    )
    model_cfg = dict(config.model)
    source = str(model_cfg["condition_source"])
    raw_cache = None
    raw_cache_repair = None
    if source == "raw_weight":
        cache_info = inspect_raw_weight_cache(
            root=Path.cwd(),
            cache_path=model_cfg["raw_weight_cache_path"],
            expected_sha256=str(model_cfg["raw_weight_cache_sha256"]),
            route=config.route,
            resume_path=os.environ.get("S5_LOCO_RESUME"),
            base_identity=asdict(base_identity),
            repair_manifest_path=os.environ.get("S5_LOCO_RAW_CACHE_REPAIR_MANIFEST"),
        )
        raw_cache_repair = cache_info["repair"]
        expected_cache_sha256 = str(cache_info["load_sha256"])
        raw_cache = S5RawWeightCache.load(
            _resolve(str(model_cfg["raw_weight_cache_path"])),
            expected_sha256=expected_cache_sha256,
            expected_base_identity=asdict(base_identity),
        )
    core = S5HyperNet(
        base,
        candidate="A",
        seed=int(config.training["seed"]),
        condition_source=source,
        fusion_mode="append",
        raw_weight_cache=raw_cache,
        raw_fusion_mode=str(model_cfg.get("raw_fusion_mode", "query_cross")),
    )
    core.validate_step0_initialization()
    init_path = _resolve(str(config.initialization["checkpoint_path"]))
    _verify_initial_checkpoint(init_path, config)
    state, payload = _load_state(init_path)
    expected_step = int(config.initialization.get("step", -1))
    checkpoint_source = payload.get("condition_source")
    # The historical P03 model-only checkpoint predates the explicit
    # condition_source field.  Its route is still unambiguous here: the
    # validated LOCO config is circuit_a, and all other checkpoint identity
    # fields (format, protocol, candidate, group, SHA, and step) are checked
    # below.  Do not apply this compatibility rule to another route.
    if checkpoint_source is None and source == "circuit" and config.route == "circuit_a":
        checkpoint_source = source
    if (
        payload.get("format") != "s5_hypernet_formal_checkpoint_v1"
        or payload.get("candidate") != "A"
        or checkpoint_source != source
        or payload.get("group") != model_cfg.get("group")
    ):
        raise ValueError("HyperNet initial checkpoint route identity changed")
    expected_treatment = str(config.data.get("input_treatment", "fixed_log1p_v1"))
    if expected_treatment == "train_zscore_v1":
        if (
            payload.get("input_treatment") != "train_zscore_v1"
            or payload.get("input_treatment_identity")
            != config.data.get("fingerprint_normalizer_artifact_sha256")
        ):
            raise ValueError(
                "z-score composition continuation cannot start from a fixed-input checkpoint"
            )
    if expected_step > 0 and int(payload.get("global_step", expected_step)) != expected_step:
        raise ValueError("HyperNet initial checkpoint step changed")
    core.load_trainable_state_dict(state)
    if config.protocol_id == VECTOR422_PROTOCOL_ID:
        expand_circuit_input_422(core)
    core.to(device)
    if core.raw_weight_conditioner is not None:
        core.raw_weight_conditioner.restore_fp32_output_path()
    if raw_cache_repair is not None:
        # Keep this provenance on the route module without changing the
        # checkpoint/config identity used by the verified continuation.
        core.__dict__["_s5_raw_cache_repair"] = dict(raw_cache_repair)
    return core


def _cosine_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float, floor: float):
    warmup = max(1, int(round(total_steps * warmup_ratio)))

    def factor(step: int) -> float:
        position = int(step) + 1
        if position <= warmup:
            return position / warmup
        progress = min(1.0, (position - warmup) / max(1, total_steps - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _manual160_optimizer_peak_lrs(config: CompositionLOCOConfig) -> dict[str, float]:
    """Resolve the sealed manual160 profile into optimizer-group peak LRs."""

    profile_id = str(config.training.get("manual160_profile_id", ""))
    values = profile_values(config.route, profile_id)
    if config.route == "direct_sft":
        return {"0": float(values["learning_rate"])}
    expected = {
        "pg": float(values["pg_lr"]),
        "query": float(values["pg_lr"]) * float(values["query_lr_multiplier"]),
    }
    if config.route == "circuit_a":
        expected["circuit"] = float(values["pg_lr"]) * float(values["circuit_lr_multiplier"])
    if config.route == "raw_weight":
        expected.update({
            "raw_encoder": float(values["raw_encoder_lr"]),
            "raw_output": float(values["raw_output_lr"]),
        })
    return expected


def _query_weight_optimizer_peak_lrs(config: CompositionLOCOConfig) -> dict[str, float]:
    """Resolve the sealed QUERY_WEIGHT profile into optimizer-group peak LRs."""

    profile_id = str(config.training.get("query_weight_profile_id", ""))
    values = query_weight_profile_values(config.route, profile_id)
    expected = {
        "pg": float(values["pg_lr"]),
        "query": float(values["pg_lr"]) * float(values["query_lr_multiplier"]),
    }
    if config.route == "raw_weight":
        expected.update({
            "raw_encoder": float(values["raw_encoder_lr"]),
            "raw_output": float(values["raw_output_lr"]),
        })
    return expected


def _query_circuit_optimizer_peak_lrs(config: CompositionLOCOConfig) -> dict[str, float]:
    """Resolve the paired Query-only/Circuit-A QUERY_CIRCUIT profile into peak LRs."""

    profile_id = str(config.training.get("query_circuit_profile_id", ""))
    values = query_circuit_profile_values(config.route, profile_id)
    expected = {
        "pg": float(values["pg_lr"]),
        "query": float(values["pg_lr"]) * float(values["query_lr_multiplier"]),
    }
    if config.route == "circuit_a":
        expected["circuit"] = float(values["pg_lr"]) * float(values["circuit_lr_multiplier"])
    return expected


def _weight_circuit_optimizer_peak_lrs(config: CompositionLOCOConfig) -> dict[str, float]:
    """Resolve the WEIGHT_CIRCUIT route/profile into optimizer-group peak LRs."""

    profile_id = str(config.training.get("weight_circuit_profile_id", ""))
    values = weight_circuit_profile_values(config.route, profile_id)
    expected = {
        "pg": float(values["pg_lr"]),
        "query": float(values["pg_lr"]) * float(values["query_lr_multiplier"]),
    }
    if config.route == "circuit_a":
        expected["circuit"] = float(values["pg_lr"]) * float(values["circuit_lr_multiplier"])
    if config.route == "raw_weight":
        expected.update({
            "raw_encoder": float(values["raw_encoder_lr"]),
            "raw_output": float(values["raw_output_lr"]),
        })
    return expected


def _nested_query_weight_optimizer_peak_lrs(config: CompositionLOCOConfig) -> dict[str, float]:
    """Resolve the nested-AND Query/Raw profile into optimizer-group peak LRs."""

    profile_id = str(config.training.get("nested_query_weight_profile_id", ""))
    values = nested_query_weight_profile_values(config.route, profile_id)
    expected = {
        "pg": float(values["pg_lr"]),
        "query": float(values["pg_lr"]) * float(values["query_lr_multiplier"]),
    }
    if config.route == "raw_weight":
        expected.update({
            "raw_encoder": float(values["raw_encoder_lr"]),
            "raw_output": float(values["raw_output_lr"]),
        })
    return expected


def _direct_lr32_optimizer_peak_lrs(config: CompositionLOCOConfig) -> dict[str, float]:
    """Resolve the sealed Direct-LR32 profile into its optimizer peak LR."""

    profile_id = str(config.training.get("direct_lr32_profile_id", ""))
    values = direct_lr32_profile_values(profile_id)
    return {"0": float(values["learning_rate"])}


def _circuit_sweep40_optimizer_peak_lrs(config: CompositionLOCOConfig) -> dict[str, float]:
    """Resolve the sealed Circuit-A sweep40 profile into group peak LRs."""

    profile_id = str(config.training.get("circuit_sweep40_profile_id", ""))
    return circuit_sweep40_optimizer_peak_lrs(profile_id)


def _save_model_checkpoint(
    path: Path,
    *,
    config: CompositionLOCOConfig,
    model: Any,
    core: Any,
    step: int,
) -> dict[str, Any]:
    target = model.module if hasattr(model, "module") else model
    if config.route == "direct_sft":
        state = {key: value.detach().cpu() for key, value in target.state_dict().items()}
    else:
        state = target.trainable_state_dict()
    payload = {
        "format": "s5_composition_loco_checkpoint_v1",
        "protocol_id": config.protocol_id,
        "config_sha256": config.computed_sha256,
        "route": config.route,
        "fold_id": config.fold_id,
        "heldout_task": config.heldout_task,
        "kind": "model_only",
        "global_step": int(step),
        "model_state_dict": state,
    }
    _atomic_torch(path, payload)
    return {
        "path": path.name,
        "global_step": int(step),
        "payload_sha256": sha256_file(path),
        "payload_bytes": path.stat().st_size,
        "config_sha256": config.computed_sha256,
    }


def _visible_score(result: Mapping[str, Any], tasks: Sequence[str]) -> float:
    counts = result.get("task_counts", {})
    values = [
        100.0 * int(counts[task]["correct"]) / int(counts[task]["total"])
        if int(counts[task]["total"])
        else 0.0
        for task in tasks
    ]
    return sum(values) / len(values) if values else 0.0


def _visible_correct_count(result: Mapping[str, Any], tasks: Sequence[str]) -> int:
    counts = result.get("task_counts", {})
    return sum(int(counts[task]["correct"]) for task in tasks)


def _visible_eval_path(eval_dir: Path, step: int) -> Path:
    return eval_dir / f"visible_step{int(step):06d}.json"


def _atomic_eval_path(eval_dir: Path, step: int) -> Path:
    return eval_dir / f"atomic_step{int(step):06d}.json"


def _expected_model_checkpoint(output: Path, step: int) -> dict[str, Any]:
    """Return the identity of the model checkpoint for one validation step."""

    if int(step) <= 0:
        raise ValueError("step zero has no model checkpoint")
    path = output / "checkpoints" / f"model_step{int(step):06d}.pt"
    if not path.is_file():
        raise ValueError(f"validation checkpoint is missing: {path}")
    return {
        "path": path.name,
        "global_step": int(step),
        "payload_sha256": sha256_file(path),
        "payload_bytes": path.stat().st_size,
        "config_sha256": None,
    }


def _validate_visible_evaluation(
    value: Mapping[str, Any],
    *,
    path: Path,
    config: CompositionLOCOConfig,
    output: Path,
    expected_step: int,
    validation_identity: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize one visible-validation artifact for selection.

    Resume must never trust a score merely because a JSON file exists.  This
    checks the fold/task identity, the fixed 500-row-per-task coverage, and the
    model checkpoint hash that produced every positive validation step.
    """

    step = int(expected_step)
    if (
        value.get("format") != "s5_composition_loco_visible_validation_v1"
        or int(value.get("step", -1)) != step
        or value.get("heldout_task") != config.heldout_task
        or value.get("visible_tasks") != list(config.train_tasks)
        or path.name != _visible_eval_path(path.parent, step).name
    ):
        raise ValueError(f"visible validation identity changed at step {step}")
    if validation_identity is not None and value.get("validation_bank_identity") != dict(validation_identity):
        raise ValueError(f"visible validation bank identity changed at step {step}")
    counts = value.get("task_counts")
    if not isinstance(counts, Mapping) or set(counts) != set(config.train_tasks):
        raise ValueError(f"visible validation task coverage changed at step {step}")
    normalized_counts: dict[str, dict[str, int]] = {}
    for task in config.train_tasks:
        row = counts.get(task)
        if not isinstance(row, Mapping):
            raise ValueError(f"visible validation count is malformed at step {step}")
        total = int(row.get("total", -1))
        correct = int(row.get("correct", -1))
        if total != 500 or correct < 0 or correct > total:
            raise ValueError(f"visible validation count changed at step {step}")
        normalized_counts[task] = {"correct": correct, "total": total}
    if sum(row["total"] for row in normalized_counts.values()) != 2_000:
        raise ValueError(f"visible validation total changed at step {step}")

    if step == 0:
        if value.get("checkpoint") is not None:
            raise ValueError("step-zero visible validation must not name a checkpoint")
        checkpoint_record: dict[str, Any] | None = None
    else:
        checkpoint_path = output / "checkpoints" / f"model_step{step:06d}.pt"
        actual = _expected_model_checkpoint(output, step)
        recorded = value.get("checkpoint")
        if recorded is None:
            # Legacy callers can reconstruct the envelope from the file.
            # Full resume additionally supplies the sealed/manifest identity
            # as `checkpoint`, so fresh bytes alone never justify old scores.
            recorded = actual
        if not isinstance(recorded, Mapping):
            raise ValueError(f"visible validation checkpoint is malformed at step {step}")
        if (
            Path(str(recorded.get("path", ""))).name != checkpoint_path.name
            or int(recorded.get("global_step", -1)) != step
            or recorded.get("payload_sha256") != actual["payload_sha256"]
            or (
                recorded.get("payload_bytes") is not None
                and int(recorded["payload_bytes"]) != actual["payload_bytes"]
            )
        ):
            raise ValueError(f"visible validation checkpoint identity changed at step {step}")
        if checkpoint is not None:
            for key in ("path", "global_step", "payload_sha256", "payload_bytes"):
                if checkpoint.get(key) is not None and recorded.get(key) != checkpoint.get(key):
                    raise ValueError(f"visible validation/checkpoint mismatch at step {step}")
        checkpoint_record = dict(actual)
        if recorded.get("config_sha256") is not None:
            checkpoint_record["config_sha256"] = recorded["config_sha256"]

    normalized = {
        "step": step,
        "score": _visible_score({"task_counts": normalized_counts}, config.train_tasks),
        "path": path.name,
        "task_counts": normalized_counts,
        "overall_correct": sum(row["correct"] for row in normalized_counts.values()),
        "overall_total": 2_000,
    }
    if checkpoint_record is not None:
        normalized["checkpoint"] = checkpoint_record
    return normalized


def _atomic_score(result: Mapping[str, Any]) -> float:
    counts = result.get("task_counts", {})
    values = [
        100.0 * int(counts[task]["correct"]) / int(counts[task]["total"])
        if int(counts[task]["total"])
        else 0.0
        for task in ATOMIC_TASKS
    ]
    return sum(values) / len(values) if values else 0.0


def _is_weight_circuit(config: CompositionLOCOConfig) -> bool:
    return config.protocol_id == WEIGHT_CIRCUIT_PROTOCOL_ID


def _is_equal_task_selection(config: CompositionLOCOConfig) -> bool:
    """Whether selection combines four visible and four atomic tasks."""

    return config.protocol_id in {WEIGHT_CIRCUIT_PROTOCOL_ID, NESTED_QUERY_WEIGHT_PROTOCOL_ID}


def _weight_circuit_best_selection(
    visible_records: Sequence[Mapping[str, Any]],
    atomic_records: Sequence[Mapping[str, Any]],
) -> tuple[int | None, float, int | None]:
    """Select the earliest positive checkpoint by equal eight-task count.

    Each record contains 2,000 seen-composition rows or 2,000 atomic rows.
    Counting correct rows (rather than averaging rounded percentages) gives
    the fixed 50/50 group weighting exactly and keeps tie-breaking stable.
    """

    atomic_by_step = {int(row.get("step", -1)): row for row in atomic_records}
    best_step: int | None = None
    best_score = -float("inf")
    best_correct: int | None = None
    for visible in visible_records:
        step = int(visible.get("step", -1))
        if step <= 0:
            continue
        atomic = atomic_by_step.get(step)
        if atomic is None:
            raise ValueError(f"weight_circuit selection is missing atomic validation at step {step}")
        visible_correct = int(visible.get("overall_correct", -1))
        visible_total = int(visible.get("overall_total", -1))
        atomic_correct = int(atomic.get("overall_correct", -1))
        atomic_total = int(atomic.get("overall_total", -1))
        if visible_total != 2_000 or atomic_total != len(ATOMIC_TASKS) * ATOMIC_SAMPLES_PER_TASK:
            raise ValueError(f"weight_circuit selection totals changed at step {step}")
        correct = visible_correct + atomic_correct
        total = visible_total + atomic_total
        if (
            best_step is None
            or best_correct is None
            or correct > best_correct
            or (correct == best_correct and step < best_step)
        ):
            best_step = step
            best_correct = correct
            best_score = 100.0 * correct / total
    return best_step, best_score, best_correct


def _weight_circuit_selection_details(
    visible_records: Sequence[Mapping[str, Any]],
    atomic_records: Sequence[Mapping[str, Any]],
    selected_step: int | None,
) -> dict[str, Any] | None:
    if selected_step is None:
        return None
    visible = next((row for row in visible_records if int(row.get("step", -1)) == selected_step), None)
    atomic = next((row for row in atomic_records if int(row.get("step", -1)) == selected_step), None)
    if visible is None or atomic is None:
        raise ValueError(f"weight_circuit selected step {selected_step} lacks paired validation records")
    visible_correct = int(visible["overall_correct"])
    visible_total = int(visible["overall_total"])
    atomic_correct = int(atomic["overall_correct"])
    atomic_total = int(atomic["overall_total"])
    total_correct = visible_correct + atomic_correct
    total = visible_total + atomic_total
    return {
        "visible_correct": visible_correct,
        "visible_total": visible_total,
        "visible_score": 100.0 * visible_correct / visible_total if visible_total else 0.0,
        "atomic_correct": atomic_correct,
        "atomic_total": atomic_total,
        "atomic_score": 100.0 * atomic_correct / atomic_total if atomic_total else 0.0,
        "selection_correct": total_correct,
        "selection_total": total,
        "selection_score": 100.0 * total_correct / total if total else 0.0,
    }


def _capture_raw_update_probe(core: Any) -> dict[str, Any] | None:
    """Capture only Raw-conditioner groups for a one-update activity audit."""

    if core is None or getattr(core, "raw_weight_conditioner", None) is None:
        return None
    partitions = core._parameter_partitions()
    groups: dict[str, Any] = {}
    for name in ("raw_encoder", "raw_output"):
        parameters = list(partitions.get(name, ()))
        groups[name] = {
            "parameters": parameters,
            "before": [parameter.detach().float().clone() for parameter in parameters],
            "grad_l2": float(
                torch.sqrt(
                    torch.stack(
                        [parameter.grad.detach().float().pow(2).sum() for parameter in parameters if parameter.grad is not None]
                    ).sum()
                ).item()
            ) if any(parameter.grad is not None for parameter in parameters) else 0.0,
        }
    return {"groups": groups}


def _finish_raw_update_probe(
    probe: dict[str, Any],
    *,
    core: Any,
    config: CompositionLOCOConfig,
    output: Path,
    step: int,
) -> None:
    groups_out: dict[str, Any] = {}
    for name, payload in probe["groups"].items():
        parameters = payload["parameters"]
        before = payload["before"]
        before_sq = sum(value.pow(2).sum() for value in before)
        after_sq = sum(parameter.detach().float().pow(2).sum() for parameter in parameters)
        delta_sq = sum(
            (parameter.detach().float() - old).pow(2).sum()
            for parameter, old in zip(parameters, before)
        )
        delta = float(torch.sqrt(delta_sq).item())
        groups_out[name] = {
            "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
            "grad_l2_before_step": float(payload["grad_l2"]),
            "parameter_l2_before_step": float(torch.sqrt(before_sq).item()),
            "parameter_l2_after_step": float(torch.sqrt(after_sq).item()),
            "parameter_delta_l2": delta,
            "updated": bool(delta > 0.0),
        }
    conditioner = getattr(core, "raw_weight_conditioner", None)
    diagnostics = conditioner.diagnostic_metrics() if conditioner is not None else {}
    _atomic_json(
        output / "raw_update_diagnostics.json",
        {
            "format": "s5_composition_loco_raw_update_diagnostics_v1",
            "protocol_id": config.protocol_id,
            "route": config.route,
            "profile_id": config.training.get(
                "raw_retention20_group_id", config.training.get("nested_query_weight_profile_id", config.training.get("weight_circuit_profile_id"))
            ),
            "step": int(step),
            "groups": groups_out,
            "conditioner": diagnostics,
            "raw_conditioner_present": conditioner is not None,
        },
    )


def _validate_atomic_evaluation(
    value: Mapping[str, Any],
    *,
    path: Path,
    config: CompositionLOCOConfig,
    output: Path,
    expected_step: int,
    validation_identity: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one atomic-retention diagnostic without affecting selection."""

    step = int(expected_step)
    if (
        value.get("format") != "s5_composition_loco_atomic_validation_v1"
        or int(value.get("step", -1)) != step
        or value.get("route") != config.route
        or value.get("fold_id") != config.fold_id
        or value.get("heldout_task") != config.heldout_task
        or value.get("atomic_tasks") != list(ATOMIC_TASKS)
        or path.name != _atomic_eval_path(path.parent, step).name
    ):
        raise ValueError(f"atomic validation identity changed at step {step}")
    if validation_identity is not None and value.get("validation_bank_identity") != dict(validation_identity):
        raise ValueError(f"atomic validation bank identity changed at step {step}")
    counts = value.get("task_counts")
    if not isinstance(counts, Mapping) or set(counts) != set(ATOMIC_TASKS):
        raise ValueError(f"atomic validation task coverage changed at step {step}")
    normalized_counts: dict[str, dict[str, int]] = {}
    for task in ATOMIC_TASKS:
        row = counts.get(task)
        if not isinstance(row, Mapping):
            raise ValueError(f"atomic validation count is malformed at step {step}")
        total = int(row.get("total", -1))
        correct = int(row.get("correct", -1))
        if total != ATOMIC_SAMPLES_PER_TASK or correct < 0 or correct > total:
            raise ValueError(f"atomic validation count changed at step {step}")
        normalized_counts[task] = {"correct": correct, "total": total}
    expected_total = len(ATOMIC_TASKS) * ATOMIC_SAMPLES_PER_TASK
    if sum(row["total"] for row in normalized_counts.values()) != expected_total:
        raise ValueError(f"atomic validation total changed at step {step}")
    if int(value.get("overall_correct", -1)) != sum(row["correct"] for row in normalized_counts.values()):
        raise ValueError(f"atomic validation overall correct count changed at step {step}")
    if int(value.get("overall_total", -1)) != expected_total:
        raise ValueError(f"atomic validation overall total changed at step {step}")

    if step == 0:
        if value.get("checkpoint") is not None:
            raise ValueError("step-zero atomic validation must not name a checkpoint")
        checkpoint_record: dict[str, Any] | None = None
    else:
        checkpoint_path = output / "checkpoints" / f"model_step{step:06d}.pt"
        actual = _expected_model_checkpoint(output, step)
        recorded = value.get("checkpoint")
        if recorded is None:
            recorded = actual
        if not isinstance(recorded, Mapping):
            raise ValueError(f"atomic validation checkpoint is malformed at step {step}")
        if (
            Path(str(recorded.get("path", ""))).name != checkpoint_path.name
            or int(recorded.get("global_step", -1)) != step
            or recorded.get("payload_sha256") != actual["payload_sha256"]
            or (
                recorded.get("payload_bytes") is not None
                and int(recorded["payload_bytes"]) != actual["payload_bytes"]
            )
        ):
            raise ValueError(f"atomic validation checkpoint identity changed at step {step}")
        if checkpoint is not None:
            for key in ("path", "global_step", "payload_sha256", "payload_bytes"):
                if checkpoint.get(key) is not None and recorded.get(key) != checkpoint.get(key):
                    raise ValueError(f"atomic validation/checkpoint mismatch at step {step}")
        checkpoint_record = dict(actual)
        if recorded.get("config_sha256") is not None:
            checkpoint_record["config_sha256"] = recorded["config_sha256"]

    normalized = {
        "step": step,
        "score": _atomic_score({"task_counts": normalized_counts}),
        "path": path.name,
        "task_counts": normalized_counts,
        "overall_correct": sum(row["correct"] for row in normalized_counts.values()),
        "overall_total": expected_total,
    }
    if checkpoint_record is not None:
        normalized["checkpoint"] = checkpoint_record
    return normalized


def _restore_selection_state(
    *,
    output: Path,
    eval_dir: Path,
    config: CompositionLOCOConfig,
    resume_payload: Mapping[str, Any],
    start_step: int,
    validation_identity: Mapping[str, Any] | None = None,
    atomic_records: Sequence[Mapping[str, Any]] | None = None,
    expected_validation_steps: Sequence[int] | None = None,
) -> tuple[int | None, float, list[dict[str, Any]], int | None]:
    """Restore selection candidates from a resume or its validation files."""

    allowed_steps = VALIDATION_STEPS if expected_validation_steps is None else tuple(expected_validation_steps)
    expected_steps = tuple(step for step in allowed_steps if step <= int(start_step))
    state = resume_payload.get("selection_state")
    raw_records: Mapping[int, Mapping[str, Any]] = {}
    if state is not None:
        if not isinstance(state, Mapping):
            raise ValueError("composition LOCO resume selection state is malformed")
        if (
            state.get("format") != "s5_composition_loco_selection_state_v1"
            or state.get("config_sha256") != config.computed_sha256
            or state.get("route") != config.route
            or state.get("fold_id") != config.fold_id
            or state.get("heldout_task") != config.heldout_task
            or state.get("visible_tasks") != list(config.train_tasks)
            or (
                validation_identity is not None
                and state.get("validation_bank_identity") != dict(validation_identity)
            )
        ):
            raise ValueError("composition LOCO resume selection identity changed")
        if _is_equal_task_selection(config) and state.get("selection_metric") != "eight_task_equal_weight_correct_count":
            raise ValueError("equal-task resume selection metric changed")
        values = state.get("evaluations")
        if not isinstance(values, list):
            raise ValueError("composition LOCO resume selection evaluations are missing")
        parsed: dict[int, Mapping[str, Any]] = {}
        for item in values:
            if not isinstance(item, Mapping):
                raise ValueError("composition LOCO resume selection record is malformed")
            step = int(item.get("step", -1))
            if step in parsed or step > int(start_step) or step not in allowed_steps:
                raise ValueError("composition LOCO resume selection step is invalid")
            parsed[step] = item
        raw_records = parsed

    if validation_identity is not None:
        resolved_path = output / "resolved_config.json"
        if not resolved_path.is_file():
            raise ValueError("composition LOCO resume has no resolved validation identity")
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        if resolved.get("validation_bank_identity") != dict(validation_identity):
            raise ValueError("composition LOCO resume validation bank identity changed")

    records: list[dict[str, Any]] = []
    for step in expected_steps:
        path = _visible_eval_path(eval_dir, step)
        if not path.is_file():
            raise ValueError(f"visible validation history is missing before resume step {start_step}: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        previous = raw_records.get(step)
        sealed_checkpoint = None if previous is None else previous.get("checkpoint")
        if step > 0 and state is not None:
            if sealed_checkpoint is None and value.get("checkpoint") is None:
                # Legacy validation JSON can omit its checkpoint envelope.
                # Require an independent recorded identity, not a fresh hash
                # of whatever bytes now occupy this step-named model file.
                manifest = output / "checkpoint_manifest.jsonl"
                if not manifest.is_file():
                    raise ValueError(f"resume selection has no recorded checkpoint identity at step {step}")
                candidates = [
                    json.loads(line)
                    for line in manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                candidates = [
                    item for item in candidates
                    if item.get("kind") == "model_only"
                    and item.get("path") == f"model_step{step:06d}.pt"
                    and int(item.get("global_step", -1)) == step
                ]
                if not candidates or any(item != candidates[0] for item in candidates[1:]):
                    raise ValueError(f"resume selection checkpoint manifest missing/ambiguous at step {step}")
                sealed_checkpoint = candidates[0]
            if sealed_checkpoint is not None and (
                not isinstance(sealed_checkpoint, Mapping)
                or not isinstance(sealed_checkpoint.get("payload_sha256"), str)
                or len(sealed_checkpoint["payload_sha256"]) != 64
                or sealed_checkpoint.get("path") != f"model_step{step:06d}.pt"
                or int(sealed_checkpoint.get("global_step", -1)) != step
            ):
                raise ValueError(f"resume sealed checkpoint identity is malformed at step {step}")
        normalized = _validate_visible_evaluation(
            value,
            path=path,
            config=config,
            output=output,
            expected_step=step,
            validation_identity=validation_identity,
            checkpoint=sealed_checkpoint,
        )
        if previous is not None:
            if (
                previous.get("path") != path.name
                or float(previous.get("score", float("nan"))) != normalized["score"]
                or previous.get("task_counts") != normalized["task_counts"]
            ):
                raise ValueError(f"composition LOCO resume selection record disagrees at step {step}")
        records.append(normalized)
    if state is not None and set(raw_records) != set(expected_steps):
        raise ValueError("composition LOCO resume selection history is incomplete")

    if _is_equal_task_selection(config):
        if atomic_records is None:
            raise ValueError("equal-task resume selection requires restored atomic validation records")
        best_step, best_score, best_correct = _weight_circuit_best_selection(records, atomic_records)
    else:
        best_step = None
        best_score = -float("inf")
        best_correct = None
        for record in records:
            step = int(record["step"])
            score = float(record["score"])
            correct = int(record["overall_correct"])
            if step > 0 and (best_step is None or best_correct is None or correct > best_correct):
                best_step = step
                best_score = score
                best_correct = correct
    if state is not None:
        saved_step = state.get("best_step")
        normalized_saved_step = None if saved_step is None else int(saved_step)
        saved_score = float(state.get("best_score", -float("inf")))
        if normalized_saved_step != best_step or saved_score != best_score:
            raise ValueError("composition LOCO resume best-selection state changed")
        saved_correct = state.get("best_correct")
        if saved_correct is None or int(saved_correct) != best_correct:
            raise ValueError("composition LOCO resume best-selection count changed")
    return best_step, best_score, records, best_correct


def _restore_atomic_evaluation_history(
    *,
    output: Path,
    eval_dir: Path,
    config: CompositionLOCOConfig,
    resume_payload: Mapping[str, Any],
    start_step: int,
    validation_identity: Mapping[str, Any],
    expected_steps: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Restore manual160 atomic diagnostics from the sealed resume state."""

    state = resume_payload.get("selection_state")
    if not isinstance(state, Mapping):
        raise ValueError("manual160 resume has no selection state for atomic validation")
    if (
        state.get("atomic_validation_bank_identity") != dict(validation_identity)
        or state.get("atomic_validation_steps")
        != list(VALIDATION_STEPS if expected_steps is None else expected_steps)
    ):
        raise ValueError("manual160 resume atomic-validation identity changed")
    values = state.get("atomic_evaluations")
    if not isinstance(values, list):
        raise ValueError("manual160 resume atomic-validation history is missing")
    allowed_steps = VALIDATION_STEPS if expected_steps is None else tuple(expected_steps)
    expected_steps = tuple(step for step in allowed_steps if step <= int(start_step))
    parsed: dict[int, Mapping[str, Any]] = {}
    for item in values:
        if not isinstance(item, Mapping):
            raise ValueError("manual160 resume atomic-validation record is malformed")
        step = int(item.get("step", -1))
        if step in parsed or step > int(start_step) or step not in allowed_steps:
            raise ValueError("manual160 resume atomic-validation step is invalid")
        parsed[step] = item
    if set(parsed) != set(expected_steps):
        raise ValueError("manual160 resume atomic-validation history is incomplete")
    records: list[dict[str, Any]] = []
    for step in expected_steps:
        path = _atomic_eval_path(eval_dir, step)
        if not path.is_file():
            raise ValueError(f"manual160 atomic validation history is missing before resume step {start_step}: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        normalized = _validate_atomic_evaluation(
            value,
            path=path,
            config=config,
            output=output,
            expected_step=step,
            validation_identity=validation_identity,
            checkpoint=parsed[step].get("checkpoint"),
        )
        previous = parsed[step]
        if (
            previous.get("path") != path.name
            or float(previous.get("score", float("nan"))) != normalized["score"]
            or previous.get("task_counts") != normalized["task_counts"]
        ):
            raise ValueError(f"manual160 resume atomic-validation record disagrees at step {step}")
        records.append(normalized)
    return records


def _load_retention_teacher(config: CompositionLOCOConfig, device: torch.device) -> Any:
    """Create an independent frozen copy of the route-specific atomic endpoint."""
    teacher = _load_direct(config, device) if config.route == "direct_sft" else _load_hypernet(config, device)
    teacher.eval()
    teacher.requires_grad_(False)
    if config.protocol_id == VECTOR422_PROTOCOL_ID:
        if bool(torch.count_nonzero(teacher.circuit_encoder.network[0].weight[:, :25])):
            raise ValueError("Vector422 teacher must retain the original atomic 397-only function")
    return teacher


def train(
    config: CompositionLOCOConfig,
    *,
    max_steps: int | None = None,
    resume_path: str | Path | None = None,
    stage2_smoke: bool = False,
) -> dict[str, Any]:
    rank, world, _local, device = _rank_world()
    is_retention = config.protocol_id in {
        DIRECT_RETENTION4_PROTOCOL_ID,
        DIRECT_RETENTION32_PROTOCOL_ID,
        CIRCUIT_RETENTION20_PROTOCOL_ID,
        VECTOR422_PROTOCOL_ID,
        QUERY_RETENTION20_PROTOCOL_ID,
        RAW_RETENTION20_PROTOCOL_ID,
    }
    training = dict(config.training)
    stage2_permuted = training.get("stage2_permuted_control")
    if stage2_smoke and (stage2_permuted is None or max_steps != 2):
        raise ValueError("stage2_smoke is restricted to two diagnostic updates of the new control")
    if stage2_permuted is not None:
        from .composition_stage2_permuted import validate_config as validate_stage2_permuted
        validate_stage2_permuted(config)
        if resume_path is not None:
            raise ValueError("stage-2 permuted control v1 requires a fresh atomic initialization")
    run_total_steps = int(training.get("total_steps", TOTAL_STEPS))
    scheduler_total_steps = int(
        training.get("scheduler_total_steps", run_total_steps)
    )
    validation_steps = tuple(
        int(step)
        for step in training.get("validation_steps", VALIDATION_STEPS)
    )
    if not validation_steps or validation_steps[0] != 0 or tuple(sorted(set(validation_steps))) != validation_steps:
        raise ValueError("composition LOCO validation schedule must be sorted and start at step zero")
    if validation_steps[-1] > run_total_steps:
        raise ValueError("composition LOCO validation schedule exceeds the train budget")
    retention_resume_format = {
        DIRECT_RETENTION32_PROTOCOL_ID: "s5_composition_loco_direct_retention32_resume_v1",
        DIRECT_RETENTION4_PROTOCOL_ID: "s5_composition_loco_direct_retention4_resume_v1",
        CIRCUIT_RETENTION20_PROTOCOL_ID: "s5_composition_loco_circuit_retention20_resume_v1",
        VECTOR422_PROTOCOL_ID: "s5_composition_loco_vector422_resume_v1",
        QUERY_RETENTION20_PROTOCOL_ID: "s5_composition_loco_query_retention20_resume_v1",
        RAW_RETENTION20_PROTOCOL_ID: "s5_composition_loco_raw_retention20_resume_v1",
    }[config.protocol_id] if is_retention else None
    if world <= 0 or int(training["global_batch_size"]) % world:
        raise ValueError("WORLD_SIZE must divide the logical training batch")
    local_batch_size = int(training["global_batch_size"]) // world
    training_receipt = None
    if config.protocol_id in {
        LR_HALF_PROTOCOL_ID, RECREATED_INPUTS_PROTOCOL_ID, ORIGINAL_LR_RECREATED_PROTOCOL_ID,
    }:
        from .composition_loco_lr_half import AUDIT_PATH, require_training_receipt
        if resume_path is not None or max_steps is not None:
            raise ValueError("this recovery wave requires fresh atomic initialization and full 38000 steps")
        training_receipt = require_training_receipt(
            config, root=Path.cwd(), audit_path=AUDIT_PATH,
            expected_identity=os.environ.get("S5_LOCO_TRAIN_EXPECTED_IDENTITY", ""),
            config_path=os.environ.get("S5_LOCO_TRAIN_CONFIG_PATH"),
        )
        _barrier()
    _seed(int(config.training["seed"]))
    zscore_enabled = (
        config.route == "circuit_a"
        and str(config.data.get("input_treatment", "fixed_log1p_v1"))
        == "train_zscore_v1"
    )
    output = _resolve(config.output_dir)
    if stage2_smoke:
        output = output / "smoke"
    if stage2_permuted is not None and (output / "resolved_config.json").exists():
        raise FileExistsError("stage-2 output already contains an attempt; preserve it before any retry")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    _barrier()
    dataset = CompositionFoldDataset(
        _resolve(str(config.data["cache_path"])),
        _resolve(str(config.data["fold_index"])),
        expected_heldout_task=config.heldout_task,
        expected_train_tasks=config.train_tasks,
        fingerprint_manifest=(
            _resolve(str(config.data["fingerprint_manifest"]))
            if zscore_enabled
            else None
        ),
        fingerprint_normalizer=(
            _resolve(str(config.data["fingerprint_normalizer"]))
            if zscore_enabled
            else None
        ),
        fingerprint_scope=(
            str(config.data["fingerprint_scope"]) if zscore_enabled else None
        ),
        fingerprint_normalizer_artifact_sha256=(
            str(config.data["fingerprint_normalizer_artifact_sha256"])
            if zscore_enabled
            else None
        ),
    )
    if len(dataset) != 1_200_000:
        raise ValueError("composition LOCO fold must contain exactly 1.2M events")
    if config.protocol_id == VECTOR422_PROTOCOL_ID:
        dataset = vector422_dataset_view(config, dataset, "composition")
    atomic_replay_dataset = None
    atomic_replay_identity = None
    retention_sampler = None
    if is_retention:
        atomic_replay_dataset, atomic_replay_identity = load_atomic_replay_dataset(
            config.data["atomic_replay"],
            expected_tokenizer_sha256=str(dataset.cache_metadata["tokenizer_sha256"]),
            expected_base_identity=STAGE_A_BASE_IDENTITY,
            # Preserve the historical Direct retention replay identity.  Only
            # Query/Raw retention protocols are vector-free; Circuit-A still
            # consumes the real Vector397 condition.
            include_vector=config.protocol_id not in {QUERY_RETENTION20_PROTOCOL_ID, RAW_RETENTION20_PROTOCOL_ID},
        )
        if config.protocol_id == VECTOR422_PROTOCOL_ID:
            atomic_replay_dataset = vector422_dataset_view(config, atomic_replay_dataset, "atomic_replay")
            atomic_replay_identity = {
                **atomic_replay_identity,
                "vector422_source": dict(atomic_replay_dataset.vector422_identity),
            }
        composition_pools, atomic_pools = build_task_index_pools(
            dataset,
            atomic_replay_dataset,
            heldout_task=config.heldout_task,
            composition_tasks=config.train_tasks,
        )
        validate_retention_pool_cardinality(
            composition_pools, atomic_pools, composition_tasks=config.train_tasks
        )
        retention_sampler = RetentionBatchSampler(
            composition_pools,
            atomic_pools,
            group_id=str(config.training["raw_retention20_group_id"]) if config.protocol_id == RAW_RETENTION20_PROTOCOL_ID else str(
                config.training.get(
                    "circuit_retention20_group_id",
                    config.training.get(
                        "query_retention20_group_id",
                        config.training.get(
                            "retention32_group_id",
                            config.training.get("retention4_group_id", ""),
                        ),
                    ),
                )
            ),
            seed=int(config.training["seed"]),
            world_size=world,
            composition_tasks=config.train_tasks,
            heldout_task=config.heldout_task,
            protocol_id=config.protocol_id,
        )
    base_ce_full = _load_base_ce(config, dataset) if config.route != "direct_sft" else None
    validation = load_eval_bank(
        _resolve(str(config.data["validation_bank"])),
        fingerprint_normalizer_path=(
            _resolve(str(config.data["fingerprint_normalizer"]))
            if zscore_enabled
            else None
        ),
        expected_fingerprint_normalizer_artifact_sha256=(
            str(config.data["fingerprint_normalizer_artifact_sha256"])
            if zscore_enabled
            else None
        ),
    )
    if config.protocol_id == VECTOR422_PROTOCOL_ID:
        validation = load_vector422_eval_bank(config, "validation")
    validation_counts = Counter(record.task_type for record in validation.records)
    if validation_counts != Counter({task: 500 for task in COMPOSITION_TASKS}):
        raise ValueError(
            "composition LOCO visible validation must contain exactly 5 x 500 rows"
        )
    if (
        validation.metadata.get("corpus_name") != "visible_validation"
        or int(validation.metadata.get("seed_base", -1)) != 2_000_000_000
        or int(validation.metadata.get("samples_per_task", -1)) != 500
        or validation.metadata.get("expected_source_base_identity") != STAGE_A_BASE_IDENTITY
    ):
        raise ValueError("composition LOCO validation bank is not visible_validation")
    atomic_validation = None
    if config.protocol_id in {
        MANUAL160_PROTOCOL_ID, DIRECT_LR32_PROTOCOL_ID, CIRCUIT_SWEEP40_PROTOCOL_ID,
        QUERY_WEIGHT_PROTOCOL_ID, QUERY_CIRCUIT_PROTOCOL_ID, WEIGHT_CIRCUIT_PROTOCOL_ID, NESTED_QUERY_WEIGHT_PROTOCOL_ID,
        DIRECT_RETENTION4_PROTOCOL_ID,
        DIRECT_RETENTION32_PROTOCOL_ID,
        CIRCUIT_RETENTION20_PROTOCOL_ID,
        VECTOR422_PROTOCOL_ID,
        QUERY_RETENTION20_PROTOCOL_ID,
        RAW_RETENTION20_PROTOCOL_ID,
    }:
        atomic_validation = (
            load_vector422_eval_bank(config, "atomic_bank")
            if config.protocol_id == VECTOR422_PROTOCOL_ID
            else load_eval_bank(_resolve(str(config.data["atomic_bank"])))
        )
        atomic_counts = Counter(record.task_type for record in atomic_validation.records)
        expected_atomic_counts = Counter({task: ATOMIC_SAMPLES_PER_TASK for task in ATOMIC_TASKS})
        if atomic_counts != expected_atomic_counts:
            raise ValueError("parameter-sweep atomic validation must contain exactly 4 x 500 rows")
        if (
            atomic_validation.metadata.get("corpus_name") != "atomic_retention"
            or int(atomic_validation.metadata.get("seed_base", -1)) != 4_000_000_000
            or int(atomic_validation.metadata.get("samples_per_task", -1)) != ATOMIC_SAMPLES_PER_TASK
            or int(atomic_validation.metadata.get("event_count", -1)) != len(ATOMIC_TASKS) * ATOMIC_SAMPLES_PER_TASK
            or atomic_validation.metadata.get("tasks") != list(ATOMIC_TASKS)
            or atomic_validation.metadata.get("task_counts") != dict(expected_atomic_counts)
            or atomic_validation.metadata.get("expected_source_base_identity") != STAGE_A_BASE_IDENTITY
        ):
            raise ValueError("parameter-sweep atomic validation bank identity changed")
    # During a retention run the only atomic evaluation performed while
    # training is the step-zero reference.  The second atomic evaluation is
    # run after visible-task selection, on the selected checkpoint (which may
    # be earlier than TOTAL_STEPS), so a resume checkpoint never assumes an
    # uncomputed step-38000 atomic artifact.
    atomic_validation_steps = (0,) if is_retention else validation_steps
    if stage2_smoke:
        atomic_validation_steps = ()
    tokenizer = load_tokenizer(_resolve(str(config.base.get("tokenizer_path", config.data.get("tokenizer_path", ""))))) if config.base.get("tokenizer_path", config.data.get("tokenizer_path")) else None
    if tokenizer is None:
        raise ValueError("composition LOCO config must declare tokenizer_path")
    if sha256_file(_resolve(str(config.data["tokenizer_path"]))) != dataset.cache_metadata["tokenizer_sha256"]:
        raise ValueError("composition LOCO tokenizer identity differs from the cache")
    if atomic_validation is not None and atomic_validation.metadata.get("tokenizer_sha256") != sha256_file(
        _resolve(str(config.data["tokenizer_path"]))
    ):
        raise ValueError("parameter-sweep atomic validation tokenizer identity differs from the cache")
    if not base_content_identity_matches(
        dataset.cache_metadata.get("expected_source_base_identity", {}),
        _base_identity(config),
    ):
        raise ValueError("composition LOCO Vector397 cache is not bound to this Stage-A Base")
    resolved_config = {
        **config.to_dict(),
        "config_sha256": config.computed_sha256,
        "cache_identity": dict(dataset.cache_metadata),
        "validation_bank_identity": dict(validation.metadata),
        "fingerprint_normalizer": (
            {
                "manifest": str(dataset.fingerprint_manifest),
                "scope": dataset.fingerprint_scope,
                "artifact_sha256": dataset.fingerprint_normalizer_metadata.get(
                    "artifact_sha256"
                )
                if dataset.fingerprint_normalizer_metadata is not None
                else None,
            }
            if zscore_enabled
            else None
        ),
        "initialization": dict(config.initialization),
        "gate_schedule": "fixed_init",
        "deployed_commit": os.environ.get("S5_DEPLOY_COMMIT"),
        **({"stage2_smoke": True, "diagnostic_only": True} if stage2_smoke else {}),
    }
    if config.protocol_id == VECTOR422_PROTOCOL_ID:
        resolved_config["vector422_composition_source"] = dict(dataset.vector422_identity)
        resolved_config["vector422_sources"] = {
            "composition": dict(dataset.vector422_identity),
            "atomic_replay": dict(atomic_replay_dataset.vector422_identity),
            "validation": dict(validation.metadata["condition_view"]["vector422_source"]),
            "atomic_bank": dict(atomic_validation.metadata["condition_view"]["vector422_source"]),
        }
    if atomic_validation is not None:
        resolved_config["atomic_validation_bank_identity"] = dict(atomic_validation.metadata)
        resolved_config["atomic_validation_steps"] = (
            list(atomic_validation_steps)
        )
    if is_retention:
        assert atomic_replay_identity is not None and retention_sampler is not None
        resolved_config["atomic_replay_identity"] = dict(atomic_replay_identity)
        resolved_config["retention_sampler"] = retention_sampler.metadata()
        hypernet_retention = config.route != "direct_sft"
        hypernet_identity = (
            "atomic_hypernet_query_only_qo02_step30516"
            if config.protocol_id == QUERY_RETENTION20_PROTOCOL_ID
            else "atomic_hypernet_raw_weight_rq14_p14_step30516"
            if config.protocol_id == RAW_RETENTION20_PROTOCOL_ID
            else "atomic_hypernet_a_p03_step15258"
        )
        resolved_config["retention_protocol"] = {
            "protocol_id": config.protocol_id,
            "student_initialization": (
                hypernet_identity if hypernet_retention else "stage_a_base_step38000"
            ),
            "teacher_initialization": (
                hypernet_identity if hypernet_retention else "stage_a_base_step38000"
            ),
            "teacher_route": config.route,
            "teacher_frozen": True,
            "ce_objective": "full_sequence_nonpadding_next_token_ce",
            "kd_objective": "atomic_response_only_kl_teacher_to_student",
            "kd_temperature": float(config.training["kd_temperature"]),
            "kd_lambda": float(config.training["kd_lambda"]),
            "train_steps": run_total_steps,
            "scheduler_steps": scheduler_total_steps,
        }
        resolved_config["teacher_identity"] = {
            "checkpoint_path": str(config.initialization["checkpoint_path"]),
            "checkpoint_sha256": str(config.initialization["checkpoint_sha256"]),
            "step": int(config.initialization["step"]),
            "route": config.route,
            "condition_source": config.model.get("condition_source"),
            "frozen": True,
            "optimizer": "none",
        }
    if training_receipt is not None:
        resolved_config["lr_half_training_identity"] = training_receipt
    raw_cache_repair = None
    teacher = None
    if config.route == "direct_sft":
        trainable = _load_direct(config, device)
        core = None
        if is_retention:
            # Load an independent copy of the same immutable Stage-A Base.
            # It never enters DDP or the optimizer and is kept in eval mode.
            teacher = _load_retention_teacher(config, device)
    else:
        core = _load_hypernet(config, device)
        trainable = core
        if is_retention:
            # Load an independent frozen copy of this route's atomic system.
            teacher = _load_retention_teacher(config, device)
        if config.protocol_id == VECTOR422_PROTOCOL_ID:
            resolved_config["vector422_migration"] = dict(core._vector422_migration)
            resolved_config["teacher_identity"]["vector422_migration"] = dict(teacher._vector422_migration)
        raw_cache_repair = getattr(core, "_s5_raw_cache_repair", None)
        if raw_cache_repair is not None:
            resolved_config["cache_repair"] = dict(raw_cache_repair)
        if config.protocol_id in {
            LR_HALF_PROTOCOL_ID, RECREATED_INPUTS_PROTOCOL_ID, ORIGINAL_LR_RECREATED_PROTOCOL_ID,
        } and config.route == "raw_weight":
            resolved_config["raw_cache_identity_receipt"] = dict(core._s5_ood_raw_cache_receipt)
        if config.protocol_id == RAW_RETENTION20_PROTOCOL_ID:
            resolved_config["raw_cache_identity_receipt"] = dict(core._s5_ood_raw_cache_receipt)
            if core._s5_ood_raw_cache_receipt != teacher._s5_ood_raw_cache_receipt:
                raise ValueError("Raw student/teacher cache receipts differ")
    if world > 1:
        ddp = nn.parallel.DistributedDataParallel(
            trainable,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        ddp = trainable
    training = dict(config.training)
    # The historical composition protocols intentionally keep their optimizer
    # constants in code.  manual160, Circuit sweep40, and the related
    # sweeps are read from their sealed profile tables instead of silently
    # falling back to a route anchor.
    beta1 = float(training.get("adam_beta1", 0.9))
    beta2 = float(training.get("adam_beta2", 0.95 if config.route == "direct_sft" else 0.98))
    weight_decay = float(training.get("weight_decay", 0.1 if config.route == "direct_sft" else 0.0))
    max_grad_norm = float(training.get("max_grad_norm", 1.0 if config.route == "direct_sft" else 0.5))
    if config.route == "direct_sft":
        optimizer = torch.optim.AdamW(
            [parameter for parameter in trainable.parameters() if parameter.requires_grad],
            lr=float(training["learning_rate"]),
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
    else:
        assert core is not None
        model_cfg = dict(config.model)
        optimizer = torch.optim.AdamW(
            core.parameter_groups(
                pg_lr=float(model_cfg["pg_lr"]),
                query_lr_multiplier=float(model_cfg.get("query_lr_multiplier", 0.5)),
                circuit_lr_multiplier=float(model_cfg.get("circuit_lr_multiplier", 2.0)),
                raw_encoder_lr=(float(model_cfg["raw_encoder_lr"]) if "raw_encoder_lr" in model_cfg else None),
                raw_output_lr=(float(model_cfg["raw_output_lr"]) if "raw_output_lr" in model_cfg else None),
                weight_decay=weight_decay,
            ),
            betas=(beta1, beta2),
            weight_decay=0.0,
        )
    if config.protocol_id in {
        LR_HALF_PROTOCOL_ID, RECREATED_INPUTS_PROTOCOL_ID, ORIGINAL_LR_RECREATED_PROTOCOL_ID,
    }:
        from .composition_loco_lr_half import validate_optimizer_peak_lrs
        validate_optimizer_peak_lrs(config, optimizer.param_groups)
        resolved_config["optimizer_peak_lrs"] = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
    elif config.protocol_id == MANUAL160_PROTOCOL_ID:
        actual_lrs = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        expected_lrs = _manual160_optimizer_peak_lrs(config)
        if actual_lrs != expected_lrs:
            raise ValueError(
                "manual160 optimizer groups do not implement the sealed profile: "
                f"actual={actual_lrs}, expected={expected_lrs}"
            )
        resolved_config["optimizer_peak_lrs"] = actual_lrs
        resolved_config["optimizer_betas"] = [beta1, beta2]
        resolved_config["optimizer_weight_decay"] = weight_decay
        resolved_config["max_grad_norm"] = max_grad_norm
    elif config.protocol_id == DIRECT_LR32_PROTOCOL_ID:
        actual_lrs = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        expected_lrs = _direct_lr32_optimizer_peak_lrs(config)
        if actual_lrs != expected_lrs:
            raise ValueError(
                "direct_lr32 optimizer does not implement the sealed profile: "
                f"actual={actual_lrs}, expected={expected_lrs}"
            )
        resolved_config["optimizer_peak_lrs"] = actual_lrs
        resolved_config["optimizer_betas"] = [beta1, beta2]
        resolved_config["optimizer_weight_decay"] = weight_decay
        resolved_config["max_grad_norm"] = max_grad_norm
    elif config.protocol_id == QUERY_WEIGHT_PROTOCOL_ID:
        actual_lrs = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        expected_lrs = _query_weight_optimizer_peak_lrs(config)
        if actual_lrs != expected_lrs:
            raise ValueError(
                "query_weight optimizer groups do not implement the sealed profile: "
                f"actual={actual_lrs}, expected={expected_lrs}"
            )
        resolved_config["optimizer_peak_lrs"] = actual_lrs
        resolved_config["optimizer_betas"] = [beta1, beta2]
        resolved_config["optimizer_weight_decay"] = weight_decay
        resolved_config["max_grad_norm"] = max_grad_norm
    elif config.protocol_id == QUERY_CIRCUIT_PROTOCOL_ID:
        actual_lrs = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        expected_lrs = _query_circuit_optimizer_peak_lrs(config)
        if actual_lrs != expected_lrs:
            raise ValueError(
                "query_circuit optimizer groups do not implement the sealed profile: "
                f"actual={actual_lrs}, expected={expected_lrs}"
            )
        resolved_config["optimizer_peak_lrs"] = actual_lrs
        resolved_config["optimizer_betas"] = [beta1, beta2]
        resolved_config["optimizer_weight_decay"] = weight_decay
        resolved_config["max_grad_norm"] = max_grad_norm
    elif config.protocol_id == WEIGHT_CIRCUIT_PROTOCOL_ID:
        actual_lrs = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        expected_lrs = _weight_circuit_optimizer_peak_lrs(config)
        if actual_lrs != expected_lrs:
            raise ValueError(
                "weight_circuit optimizer groups do not implement the sealed profile: "
                f"actual={actual_lrs}, expected={expected_lrs}"
            )
        resolved_config["optimizer_peak_lrs"] = actual_lrs
        resolved_config["optimizer_betas"] = [beta1, beta2]
        resolved_config["optimizer_weight_decay"] = weight_decay
        resolved_config["max_grad_norm"] = max_grad_norm
    elif config.protocol_id == NESTED_QUERY_WEIGHT_PROTOCOL_ID:
        actual_lrs = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        expected_lrs = _nested_query_weight_optimizer_peak_lrs(config)
        if actual_lrs != expected_lrs:
            raise ValueError(
                "nested_query_weight optimizer groups do not implement the sealed profile: "
                f"actual={actual_lrs}, expected={expected_lrs}"
            )
        resolved_config["optimizer_peak_lrs"] = actual_lrs
        resolved_config["optimizer_betas"] = [beta1, beta2]
        resolved_config["optimizer_weight_decay"] = weight_decay
        resolved_config["max_grad_norm"] = max_grad_norm
    elif config.protocol_id == CIRCUIT_SWEEP40_PROTOCOL_ID:
        actual_lrs = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        expected_lrs = _circuit_sweep40_optimizer_peak_lrs(config)
        if actual_lrs != expected_lrs:
            raise ValueError(
                "circuit sweep40 optimizer does not implement the sealed profile: "
                f"actual={actual_lrs}, expected={expected_lrs}"
            )
        resolved_config["optimizer_peak_lrs"] = actual_lrs
        resolved_config["optimizer_betas"] = [beta1, beta2]
        resolved_config["optimizer_weight_decay"] = weight_decay
        resolved_config["max_grad_norm"] = max_grad_norm
    elif config.protocol_id in {CIRCUIT_RETENTION20_PROTOCOL_ID, VECTOR422_PROTOCOL_ID}:
        actual_lrs = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        expected_lrs = {
            "pg": float(config.model["pg_lr"]),
            "query": float(config.model["pg_lr"])
            * float(config.model["query_lr_multiplier"]),
            "circuit": float(config.model["pg_lr"])
            * float(config.model["circuit_lr_multiplier"]),
        }
        if actual_lrs != expected_lrs:
            raise ValueError(
                "circuit_retention20 optimizer does not implement the sealed profile: "
                f"actual={actual_lrs}, expected={expected_lrs}"
            )
        resolved_config["optimizer_peak_lrs"] = actual_lrs
        resolved_config["optimizer_betas"] = [beta1, beta2]
        resolved_config["optimizer_weight_decay"] = weight_decay
        resolved_config["max_grad_norm"] = max_grad_norm
    elif config.protocol_id == QUERY_RETENTION20_PROTOCOL_ID:
        actual_lrs = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        expected_lrs = {
            "pg": float(config.model["pg_lr"]),
            "query": float(config.model["pg_lr"])
            * float(config.model["query_lr_multiplier"]),
        }
        if actual_lrs != expected_lrs:
            raise ValueError(
                "query_retention20 optimizer does not implement the sealed profile: "
                f"actual={actual_lrs}, expected={expected_lrs}"
            )
        resolved_config["optimizer_peak_lrs"] = actual_lrs
        resolved_config["optimizer_betas"] = [beta1, beta2]
        resolved_config["optimizer_weight_decay"] = weight_decay
        resolved_config["max_grad_norm"] = max_grad_norm
    elif config.protocol_id == RAW_RETENTION20_PROTOCOL_ID:
        actual_lrs = {
            str(group.get("name", index)): float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        expected_lrs = {
            "raw_encoder": float(config.model["raw_encoder_lr"]),
            "raw_output": float(config.model["raw_output_lr"]),
            "pg": float(config.model["pg_lr"]),
            "query": float(config.model["pg_lr"])
            * float(config.model["query_lr_multiplier"]),
        }
        if actual_lrs != expected_lrs:
            raise ValueError(
                "raw_retention20 optimizer does not implement the sealed profile: "
                f"actual={actual_lrs}, expected={expected_lrs}"
            )
        resolved_config["optimizer_peak_lrs"] = actual_lrs
        resolved_config["optimizer_betas"] = [beta1, beta2]
        resolved_config["optimizer_weight_decay"] = weight_decay
        resolved_config["max_grad_norm"] = max_grad_norm
    scheduler = _cosine_scheduler(
        optimizer,
        scheduler_total_steps,
        float(training["warmup_ratio"]),
        float(training["cosine_min_lr_ratio"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.route == "direct_sft" and device.type == "cuda")
    total_steps = run_total_steps if max_steps is None else int(max_steps)
    if not 0 < total_steps <= run_total_steps:
        raise ValueError("max_steps is outside the approved budget")
    metrics_path = (
        _resolve(str(stage2_permuted["log_dir"]))
        / ("smoke" if stage2_smoke else ".") / "train_metrics.jsonl"
        if stage2_permuted is not None else output / "train_metrics.jsonl"
    )
    if rank == 0:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest_path = output / "checkpoint_manifest.jsonl"
    eval_dir = output / "evaluations"
    ckpt_dir = output / "checkpoints"
    best_step: int | None = None
    best_score = -float("inf")
    best_correct: int | None = None
    eval_records: list[dict[str, Any]] = []
    atomic_eval_records: list[dict[str, Any]] = []
    nonfinite_gradient_steps: int | None = 0
    nonfinite_gradient_steps_known_since = 0
    nonfinite_gradient_steps_since_known = 0
    start_step = 0
    if resume_path is not None:
        resume_file = _resolve(resume_path)
        payload, resume_evidence = load_verified_resume(
            resume_file,
            config=config,
            output_dir=output,
            require_cuda_rng=device.type == "cuda",
            expected_cache_identity=dataset.cache_metadata,
            expected_validation_identity=validation.metadata,
        )
        start_step = int(payload["global_step"])
        if not 0 < start_step <= total_steps:
            raise ValueError("composition LOCO resume step is outside the requested range")
        if is_retention:
            if retention_sampler is None or atomic_replay_identity is None:
                raise RuntimeError("retention provenance was not initialized before resume")
            retention_state = payload.get("retention_state")
            expected_retention_state = {
                "format": retention_resume_format,
                "protocol_id": config.protocol_id,
                "atomic_replay_identity": dict(atomic_replay_identity),
                "sampler": retention_sampler.metadata(),
                "cursor": retention_sampler.cursor_metadata(start_step),
                "teacher_identity": resolved_config["teacher_identity"],
            }
            if config.protocol_id == VECTOR422_PROTOCOL_ID:
                expected_retention_state["vector422_sources"] = resolved_config["vector422_sources"]
            if retention_state != expected_retention_state:
                raise ValueError("Direct retention resume replay/teacher provenance changed")
        target = trainable
        state = payload.get("model_state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("composition LOCO resume has no model_state_dict")
        if config.route == "direct_sft":
            target.load_state_dict(state, strict=True)
        else:
            target.load_trainable_state_dict(state)
        validate_optimizer_layout(payload, optimizer)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        if config.route == "direct_sft":
            scaler.load_state_dict(payload["scaler_state_dict"])
        rng_by_rank = payload.get("rng_by_rank")
        if not isinstance(rng_by_rank, list) or len(rng_by_rank) != world:
            raise ValueError("composition LOCO resume RNG coverage changed")
        _restore_rng(rng_by_rank[rank])
        nonfinite_gradient_steps = resume_evidence["nonfinite_gradient_steps"]
        nonfinite_gradient_steps_known_since = resume_evidence["nonfinite_gradient_steps_known_since"]
        nonfinite_gradient_steps_since_known = resume_evidence["nonfinite_gradient_steps_since_known"]
        if rank == 0:
            restored_atomic_records: list[dict[str, Any]] = []
            if atomic_validation is not None:
                restored_atomic_records = _restore_atomic_evaluation_history(
                    output=output,
                    eval_dir=eval_dir,
                    config=config,
                    resume_payload=payload,
                    start_step=start_step,
                    validation_identity=atomic_validation.metadata,
                    expected_steps=atomic_validation_steps,
                )
            best_step, best_score, eval_records, best_correct = _restore_selection_state(
                output=output,
                eval_dir=eval_dir,
                config=config,
                resume_payload=payload,
                start_step=start_step,
                validation_identity=validation.metadata,
                atomic_records=restored_atomic_records if atomic_validation is not None else None,
                expected_validation_steps=validation_steps,
            )
            atomic_eval_records = restored_atomic_records
        if dist.is_available() and dist.is_initialized():
            box = torch.tensor(
                [
                    -1 if best_step is None else best_step,
                    best_score,
                    -1 if best_correct is None else best_correct,
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.broadcast(box, src=0)
            best_step = None if int(box[0].item()) < 0 else int(box[0].item())
            best_score = float(box[1].item())
            best_correct = None if int(box[2].item()) < 0 else int(box[2].item())

        # A checkpoint can precede old model/evaluation files from the failed
        # attempt. Preserve those files before replay so cached scores cannot
        # be paired with newly written weights at the same step. Every rank
        # has loaded its optimizer and RNG before rank zero changes any file.
        _barrier()
        if rank == 0:
            archived = archive_replay_suffix(
                output,
                resume_step=start_step,
                resume_sha256=resume_evidence["payload_sha256"],
                total_steps=run_total_steps,
            )
            resolved_config["recovery"] = {**resume_evidence, "archive_path": str(archived)}
            print(f"verified composition LOCO recovery at step {start_step}; archive={archived}", flush=True)
        _barrier()
        del payload
    if rank == 0:
        # Do not replace the old provenance until resume identity and history
        # have been verified and a recoverable snapshot has been retained.
        _atomic_json(output / "resolved_config.json", resolved_config)
    _barrier()

    from .hypernet.formal_training import rank_indices_for_step
    from .composition_loco_eval import DEFAULT_EVAL_BATCH_SIZE, evaluate_bank
    eval_batch_size = int(config.evaluation.get("eval_batch_size", DEFAULT_EVAL_BATCH_SIZE))
    if eval_batch_size <= 1:
        raise ValueError("composition validation must use a batched eval_batch_size > 1")

    def evaluate_visible(step: int, checkpoint: Mapping[str, Any] | None) -> None:
        nonlocal best_step, best_score, best_correct
        path = _visible_eval_path(eval_dir, step)
        if path.is_file():
            value = _validate_visible_evaluation(
                json.loads(path.read_text(encoding="utf-8")),
                path=path,
                config=config,
                output=output,
                expected_step=step,
                validation_identity=validation.metadata,
                checkpoint=checkpoint,
            )
        else:
            if config.route == "direct_sft":
                value = evaluate_bank(
                    trainable,
                    tokenizer,
                    validation,
                    device=device,
                    task_types=config.train_tasks,
                    route="direct_sft",
                    condition="matched",
                    gate_multiplier=1.0,
                    batch_size=eval_batch_size,
                )
            else:
                assert core is not None
                core._loco_gate_multiplier = float(config.model.get("gate_max", 1.0))
                value = evaluate_bank(
                    core,
                    tokenizer,
                    validation,
                    device=device,
                    task_types=config.train_tasks,
                    route=config.route,
                    condition="matched",
                    gate_multiplier=float(config.model.get("gate_max", 1.0)),
                    batch_size=eval_batch_size,
                )
            if rank == 0:
                assert value is not None
                value = {
                    "format": "s5_composition_loco_visible_validation_v1",
                    "step": int(step),
                    "checkpoint": None if checkpoint is None else dict(checkpoint),
                    "heldout_task": config.heldout_task,
                    "visible_tasks": list(config.train_tasks),
                    "validation_bank_identity": dict(validation.metadata),
                    **value,
                }
                _atomic_json(path, value)
                value = _validate_visible_evaluation(
                    value,
                    path=path,
                    config=config,
                    output=output,
                    expected_step=step,
                    validation_identity=validation.metadata,
                    checkpoint=checkpoint,
                )
        if rank == 0:
            score = float(value["score"]) if "score" in value else _visible_score(value, config.train_tasks)
            correct = int(value["overall_correct"])
            if (
                not _is_equal_task_selection(config)
                and step > 0
                and (best_step is None or best_correct is None or correct > best_correct)
            ):
                best_step = int(step)
                best_score = score
                best_correct = correct
        if dist.is_available() and dist.is_initialized():
            box = torch.tensor(
                [
                    -1 if best_step is None else best_step,
                    best_score,
                    -1 if best_correct is None else best_correct,
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.broadcast(box, src=0)
            best_step = None if int(box[0].item()) < 0 else int(box[0].item())
            best_score = float(box[1].item())
            best_correct = None if int(box[2].item()) < 0 else int(box[2].item())
        if rank == 0:
            record = dict(value)
            record["score"] = score
            eval_records[:] = [item for item in eval_records if int(item.get("step", -1)) != int(step)]
            eval_records.append(record)
            eval_records.sort(key=lambda item: int(item["step"]))

    def evaluate_atomic(step: int, checkpoint: Mapping[str, Any] | None) -> None:
        """Record atomic retention at every validation checkpoint.

        Atomic accuracy is diagnostic-only for historical protocols.  WEIGHT_CIRCUIT
        WEIGHT_CIRCUIT/NESTED_QUERY_WEIGHT deliberately use it as the second, equal-weight half of their frozen
        eight-task selector; the held-out composition family is still never
        evaluated during training selection.
        """

        nonlocal best_step, best_score, best_correct

        if atomic_validation is None:
            return
        path = _atomic_eval_path(eval_dir, step)
        if path.is_file():
            value = _validate_atomic_evaluation(
                json.loads(path.read_text(encoding="utf-8")),
                path=path,
                config=config,
                output=output,
                expected_step=step,
                validation_identity=atomic_validation.metadata,
                checkpoint=checkpoint,
            )
        else:
            if config.route == "direct_sft":
                value = evaluate_bank(
                    trainable,
                    tokenizer,
                    atomic_validation,
                    device=device,
                    task_types=ATOMIC_TASKS,
                    route="direct_sft",
                    condition="matched",
                    gate_multiplier=1.0,
                    batch_size=eval_batch_size,
                )
            else:
                assert core is not None
                core._loco_gate_multiplier = float(config.model.get("gate_max", 1.0))
                value = evaluate_bank(
                    core,
                    tokenizer,
                    atomic_validation,
                    device=device,
                    task_types=ATOMIC_TASKS,
                    route=config.route,
                    condition="matched",
                    gate_multiplier=float(config.model.get("gate_max", 1.0)),
                    batch_size=eval_batch_size,
                )
            if rank == 0:
                assert value is not None
                value = {
                    "format": "s5_composition_loco_atomic_validation_v1",
                    "step": int(step),
                    "checkpoint": None if checkpoint is None else dict(checkpoint),
                    "route": config.route,
                    "fold_id": config.fold_id,
                    "heldout_task": config.heldout_task,
                    "atomic_tasks": list(ATOMIC_TASKS),
                    "validation_bank_identity": dict(atomic_validation.metadata),
                    **value,
                }
                _atomic_json(path, value)
                value = _validate_atomic_evaluation(
                    value,
                    path=path,
                    config=config,
                    output=output,
                    expected_step=step,
                    validation_identity=atomic_validation.metadata,
                    checkpoint=checkpoint,
                )
        if rank == 0:
            record = dict(value)
            record["score"] = float(value["score"])
            atomic_eval_records[:] = [
                item for item in atomic_eval_records if int(item.get("step", -1)) != int(step)
            ]
            atomic_eval_records.append(record)
            atomic_eval_records.sort(key=lambda item: int(item["step"]))
            if _is_equal_task_selection(config):
                best_step, best_score, best_correct = _weight_circuit_best_selection(
                    eval_records, atomic_eval_records
                )
                details = _weight_circuit_selection_details(
                    eval_records, atomic_eval_records, int(step)
                )
                if details is not None:
                    visible_record = next(
                        item for item in eval_records if int(item["step"]) == int(step)
                    )
                    visible_record.update(
                        {
                            "selection_visible_correct": details["visible_correct"],
                            "selection_visible_total": details["visible_total"],
                            "selection_atomic_correct": details["atomic_correct"],
                            "selection_atomic_total": details["atomic_total"],
                            "selection_correct": details["selection_correct"],
                            "selection_total": details["selection_total"],
                            "selection_score": details["selection_score"],
                        }
                    )
        if _is_equal_task_selection(config) and dist.is_available() and dist.is_initialized():
            box = torch.tensor(
                [
                    -1 if best_step is None else best_step,
                    best_score,
                    -1 if best_correct is None else best_correct,
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.broadcast(box, src=0)
            best_step = None if int(box[0].item()) < 0 else int(box[0].item())
            best_score = float(box[1].item())
            best_correct = None if int(box[2].item()) < 0 else int(box[2].item())

    # Step zero is a reference only and can never become the selected model.
    if start_step == 0 and not stage2_smoke:
        evaluate_visible(0, None)
        evaluate_atomic(0, None)
    epoch_cache: tuple[int, Tensor] | None = None
    trainable.train()
    if core is not None:
        core.train()
    if teacher is not None:
        teacher.eval()
        teacher.requires_grad_(False)
    retention_special_ids = _tokenizer_special_ids(tokenizer) if is_retention else ()
    for step_index in range(start_step, total_steps):
        local_refs = None
        if is_retention:
            if retention_sampler is None or atomic_replay_dataset is None:
                raise RuntimeError("retention sampler/dataset was not initialized")
            local_refs = retention_sampler.local_batch(step_index, rank=rank)
            if local_batch_size != len(local_refs):
                raise ValueError(
                    "retention protocols require one full rank-local batch per microbatch"
                )
            local_batch = collate_retention_batch(
                dataset,
                atomic_replay_dataset,
                local_refs,
                device=device,
                include_vector=config.protocol_id not in {QUERY_RETENTION20_PROTOCOL_ID, RAW_RETENTION20_PROTOCOL_ID},
                expected_vector_dim=422 if config.protocol_id == VECTOR422_PROTOCOL_ID else 397,
            )
            local_lengths = local_batch["input_lengths"].long()
            local_kd_mask = atomic_response_token_mask(
                local_batch["labels"],
                response_start=local_batch["response_start"],
                input_lengths=local_batch["input_lengths"],
                is_atomic=local_batch["is_atomic"],
                special_token_ids=retention_special_ids,
            )
            local_kd_tokens = torch.tensor(
                int(local_kd_mask.sum().item()), dtype=torch.float64, device=device
            )
            global_kd_tokens = _all_reduce(local_kd_tokens)
        else:
            local_indices, epoch_cache = rank_indices_for_step(
                step_index,
                dataset_size=len(dataset),
                seed=int(config.training["seed"]),
                rank=rank,
                world_size=world,
                cached_epoch=epoch_cache,
            )
            local_lengths = dataset.payload["input_lengths"].index_select(0, dataset.indices.index_select(0, local_indices)).long()
            local_batch = None
            global_kd_tokens = torch.zeros((), dtype=torch.float64, device=device)
        global_tokens = _all_reduce((local_lengths - 1).sum().to(device=device))
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), dtype=torch.float64, device=device)
        step_kd = torch.zeros((), dtype=torch.float64, device=device)
        step_kd_tokens = torch.zeros((), dtype=torch.float64, device=device)
        step_hinge = torch.zeros((), dtype=torch.float64, device=device)
        step_delta = torch.zeros((), dtype=torch.float64, device=device)
        gradient_is_finite = True
        started = time.perf_counter()
        micro = local_batch_size
        if is_retention:
            assert local_refs is not None and local_batch is not None
            micro_batches = (local_batch,)
            micro_indices_iter = (None,)
        else:
            micro_indices_iter = local_indices.split(micro)
            micro_batches = (None,) * len(micro_indices_iter)
        for micro_indices, retained_batch in zip(micro_indices_iter, micro_batches, strict=True):
            batch = (
                retained_batch
                if is_retention
                else dataset.collate_indices(micro_indices, device=device)
            )
            # Keep recipient data and the frozen KD teacher's matched Circuit
            # untouched. Only the student's condition is permuted, from the
            # first optimizer update through the fixed stage-2 stop.
            matched_vector397 = batch.get("vector397")
            if stage2_permuted is not None:
                from .composition_stage2_permuted import permuted_student_batch, append_trace
                batch, pairing_trace = permuted_student_batch(
                    batch, seed=int(stage2_permuted["permutation_seed"]), step=step_index,
                )
                if rank == 0:
                    pairing_log = _resolve(str(stage2_permuted["log_dir"]))
                    append_trace(pairing_log / "smoke" if stage2_smoke else pairing_log, pairing_trace)
            if config.route == "direct_sft":
                with _autocast(device, torch.float16):
                    logits = _logits(ddp(batch["input_ids"]))
                    shifted = batch["labels"][:, 1:]
                    local_loss_sum = F.cross_entropy(
                        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                        shifted.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                if is_retention:
                    assert teacher is not None
                    atomic_rows = torch.nonzero(
                        batch["is_atomic"], as_tuple=False
                    ).flatten()
                    if atomic_rows.numel():
                        # Teacher sees exactly the same teacher-forcing inputs,
                        # but only the atomic sub-batch is evaluated.
                        with torch.no_grad(), _autocast(device, torch.float16):
                            teacher_logits = _logits(
                                teacher(batch["input_ids"].index_select(0, atomic_rows))
                            )
                        student_atomic_logits = logits.index_select(0, atomic_rows)
                        atomic_labels = batch["labels"].index_select(0, atomic_rows)
                        atomic_response_start = batch["response_start"].index_select(
                            0, atomic_rows
                        )
                        atomic_lengths = batch["input_lengths"].index_select(0, atomic_rows)
                        atomic_mask = atomic_response_token_mask(
                            atomic_labels,
                            response_start=atomic_response_start,
                            input_lengths=atomic_lengths,
                            is_atomic=torch.ones(
                                atomic_rows.numel(), dtype=torch.bool, device=device
                            ),
                            special_token_ids=retention_special_ids,
                        )
                        local_kd_sum, local_kd_count = atomic_kd_loss_sum(
                            student_atomic_logits,
                            teacher_logits,
                            atomic_mask,
                            temperature=float(config.training["kd_temperature"]),
                        )
                    else:
                        local_kd_sum = logits.sum() * 0.0
                        local_kd_count = 0
                    if float(global_kd_tokens.item()) > 0.0:
                        kd_term = (
                            local_kd_sum
                            * float(world)
                            / global_kd_tokens.to(dtype=local_kd_sum.dtype)
                        )
                    else:
                        kd_term = logits.sum() * 0.0
                    loss = (
                        local_loss_sum * float(world) / global_tokens.to(dtype=local_loss_sum.dtype)
                        + float(config.training["kd_lambda"]) * kd_term
                    )
                else:
                    local_kd_sum = logits.sum() * 0.0
                    local_kd_count = 0
                    loss = local_loss_sum * float(world) / global_tokens.to(dtype=local_loss_sum.dtype)
                hinge_sum = torch.zeros((), device=device)
                delta_sum = torch.zeros((), device=device)
            else:
                assert core is not None
                if base_ce_full is None:
                    raise RuntimeError("HyperNet route has no frozen Base-CE artifact")
                if is_retention:
                    base_ce = _mixed_frozen_base_ce(
                        core,
                        batch,
                        base_ce_full,
                        device=device,
                    )
                else:
                    base_ce = base_ce_full.index_select(
                        0, batch["global_indices"].detach().cpu()
                    ).to(device=device, non_blocking=True)
                batch["frozen_base_ce"] = base_ce.detach()
                batch_vector397 = batch.get("vector397")
                # Continuation starts from a trained route checkpoint.  The
                # audited gate is fixed at that route's multiplier; restarting
                # the old 20% gate warmup would silently weaken the init and
                # make validation incomparable with the selected checkpoint.
                gate = float(config.model["gate_max"])
                with _autocast(device, torch.bfloat16):
                    result = ddp(
                        query_ids=batch["query_ids"],
                        query_mask=batch["query_mask"],
                        input_ids=batch["input_ids"],
                        labels=batch["labels"],
                        vector397=batch_vector397,
                        frozen_base_ce=batch["frozen_base_ce"],
                        gate_multiplier=gate,
                    )
                    hinge_sum = result["base_preservation_hinge"].sum()
                    delta_sum = result["effective_delta_l2"].sum()
                    loss = scale_microbatch_objective_for_ddp(
                        loss_sum=result["loss_sum"],
                        hinge_sum=hinge_sum,
                        effective_delta_sum=delta_sum,
                        world_size=world,
                        global_token_count=global_tokens,
                        global_event_count=GLOBAL_BATCH_SIZE,
                        base_preservation_coefficient=float(model_cfg.get("base_preservation_coefficient", 0.5)),
                        effective_delta_coefficient=float(model_cfg.get("effective_delta_l2", 1.0e-4)),
                    )
                    if is_retention:
                        assert teacher is not None
                        atomic_rows = torch.nonzero(
                            batch["is_atomic"], as_tuple=False
                        ).flatten()
                        if atomic_rows.numel():
                            with torch.no_grad(), _autocast(device, torch.bfloat16):
                                teacher_result = teacher(
                                    query_ids=batch["query_ids"].index_select(0, atomic_rows),
                                    query_mask=batch["query_mask"].index_select(0, atomic_rows),
                                    input_ids=batch["input_ids"].index_select(0, atomic_rows),
                                    labels=batch["labels"].index_select(0, atomic_rows),
                                    vector397=(
                                        None
                                        if matched_vector397 is None
                                        else matched_vector397.index_select(0, atomic_rows)
                                    ),
                                    frozen_base_ce=base_ce.index_select(0, atomic_rows),
                                    gate_multiplier=gate,
                                )
                            student_atomic_logits = result["logits"].index_select(
                                0, atomic_rows
                            )
                            atomic_labels = batch["labels"].index_select(0, atomic_rows)
                            atomic_response_start = batch["response_start"].index_select(
                                0, atomic_rows
                            )
                            atomic_lengths = batch["input_lengths"].index_select(0, atomic_rows)
                            atomic_mask = atomic_response_token_mask(
                                atomic_labels,
                                response_start=atomic_response_start,
                                input_lengths=atomic_lengths,
                                is_atomic=torch.ones(
                                    atomic_rows.numel(), dtype=torch.bool, device=device
                                ),
                                special_token_ids=retention_special_ids,
                            )
                            local_kd_sum, local_kd_count = atomic_kd_loss_sum(
                                student_atomic_logits,
                                teacher_result["logits"],
                                atomic_mask,
                                temperature=float(config.training["kd_temperature"]),
                            )
                        else:
                            local_kd_sum = result["loss_sum"] * 0.0
                            local_kd_count = 0
                        if float(global_kd_tokens.item()) > 0.0:
                            kd_term = (
                                local_kd_sum
                                * float(world)
                                / global_kd_tokens.to(dtype=local_kd_sum.dtype)
                            )
                        else:
                            kd_term = result["loss_sum"] * 0.0
                        loss = loss + float(config.training["kd_lambda"]) * kd_term
                    else:
                        local_kd_sum = result["loss_sum"] * 0.0
                        local_kd_count = 0
                local_loss_sum = result["loss_sum"].detach()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite composition LOCO loss at step {step_index + 1}")
            if config.route == "direct_sft":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            step_loss += local_loss_sum.detach().double()
            if is_retention:
                step_kd += local_kd_sum.detach().double()
                step_kd_tokens += torch.tensor(
                    float(local_kd_count), dtype=torch.float64, device=device
                )
            step_hinge += hinge_sum.detach().double()
            step_delta += delta_sum.detach().double()
        raw_update_probe = None
        if (
            rank == 0
            and config.route == "raw_weight"
            and (
                (_is_equal_task_selection(config) and step_index == start_step)
                or (config.protocol_id == RAW_RETENTION20_PROTOCOL_ID
                    and step_index == max(1, start_step)
                    and all(float(group["lr"]) > 0 for group in optimizer.param_groups))
            )
        ):
            raw_update_probe = _capture_raw_update_probe(core)
        if config.route == "direct_sft":
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(trainable.parameters(), max_grad_norm)
            gradient_is_finite = bool(torch.isfinite(torch.as_tensor(grad_norm)))
            if dist.is_available() and dist.is_initialized():
                finite_flag = torch.tensor(
                    1 if gradient_is_finite else 0,
                    dtype=torch.int32,
                    device=device,
                )
                # DDP must make the same optimizer decision on every rank.
                # A single overflowing rank invalidates the whole update.
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
                gradient_is_finite = bool(finite_flag.item())
            if gradient_is_finite:
                scaler.step(optimizer)
            else:
                # GradScaler recorded the overflow during unscale_.  Let it
                # lower the scale, discard this batch, and keep the global
                # training cursor moving instead of aborting the run.
                nonfinite_gradient_steps_since_known += 1
                if nonfinite_gradient_steps is not None:
                    nonfinite_gradient_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if rank == 0:
                    _append_jsonl(
                        output / "optimizer_skip_events.jsonl",
                        {
                            "step": step_index + 1,
                            "reason": "nonfinite_gradient",
                            "nonfinite_gradient_steps": nonfinite_gradient_steps,
                            "nonfinite_gradient_steps_known_since": nonfinite_gradient_steps_known_since,
                            "nonfinite_gradient_steps_since_known": nonfinite_gradient_steps_since_known,
                        },
                    )
                    print(
                        "skipping non-finite composition LOCO gradient "
                        f"at step {step_index + 1}; GradScaler will update",
                        flush=True,
                    )
            scaler.update()
        else:
            grad_norm = nn.utils.clip_grad_norm_(
                [parameter for parameter in trainable.parameters() if parameter.requires_grad], max_grad_norm
            )
            if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
                raise FloatingPointError(f"non-finite composition LOCO gradient at step {step_index + 1}")
            optimizer.step()
            if raw_update_probe is not None:
                _finish_raw_update_probe(
                    raw_update_probe,
                    core=core,
                    config=config,
                    output=output,
                    step=step_index + 1,
                )
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step = step_index + 1
        elapsed = time.perf_counter() - started
        if step == 1 or step % 50 == 0 or step == total_steps:
            global_loss = _all_reduce(step_loss)
            global_kd = _all_reduce(step_kd)
            global_kd_token_count = _all_reduce(step_kd_tokens)
            global_hinge = _all_reduce(step_hinge)
            global_delta = _all_reduce(step_delta)
        if rank == 0 and (step == 1 or step % 50 == 0 or step == total_steps):
            tokens = global_tokens
            metric = {
                "step": step,
                "full_sequence_ce": float(global_loss.item() / max(1, tokens.item())),
                "atomic_kd": (
                    float(global_kd.item() / max(1.0, global_kd_token_count.item()))
                    if is_retention
                    else 0.0
                ),
                "atomic_response_tokens": int(global_kd_token_count.item()) if is_retention else 0,
                "kd_lambda": float(config.training["kd_lambda"]) if is_retention else 0.0,
                "kd_temperature": float(config.training["kd_temperature"]) if is_retention else 0.0,
                "base_preservation_hinge": float(global_hinge.item() / GLOBAL_BATCH_SIZE),
                "effective_delta_l2": float(global_delta.item() / GLOBAL_BATCH_SIZE),
                "grad_norm": float(torch.as_tensor(grad_norm).float().item()),
                "optimizer_step_skipped": not gradient_is_finite
                if config.route == "direct_sft"
                else False,
                "nonfinite_gradient_steps": nonfinite_gradient_steps,
                "nonfinite_gradient_steps_known_since": nonfinite_gradient_steps_known_since,
                "nonfinite_gradient_steps_since_known": nonfinite_gradient_steps_since_known,
                "step_seconds": elapsed,
                "global_tokens": int(tokens.item()),
                "lr": {str(group.get("name", i)): float(group["lr"]) for i, group in enumerate(optimizer.param_groups)},
                "max_memory_allocated": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric, sort_keys=True) + "\n")
        model_due = step % 1_000 == 0 or step == total_steps
        if model_due:
            _barrier()
            checkpoint = None
            if rank == 0:
                checkpoint = _save_model_checkpoint(
                    ckpt_dir / f"model_step{step:06d}.pt",
                    config=config,
                    model=trainable,
                    core=core,
                    step=step,
                )
                _append_jsonl(
                    checkpoint_manifest_path,
                    {"kind": "model_only", **checkpoint},
                )
            _barrier()
            if step in validation_steps:  # legacy equivalent: if step in VALIDATION_STEPS
                # Validation is a distributed collective; every rank must
                # enter.  Model checkpoints are denser than evaluations.
                evaluate_visible(step, checkpoint)
                if not is_retention:
                    evaluate_atomic(step, checkpoint)
                _barrier()
        if step % 5_000 == 0 or step == total_steps:
            _barrier()
            rng_by_rank = _gather_rng_states(world)
            if rank == 0:
                resume_output = ckpt_dir / f"resume_step{step:06d}.pt"
                _atomic_torch(
                    resume_output,
                    {
                        "format": "s5_composition_loco_resume_v1",
                        "config_sha256": config.computed_sha256,
                        "route": config.route,
                        "fold_id": config.fold_id,
                        "global_step": step,
                        "model_state_dict": (
                            {key: value.detach().cpu() for key, value in trainable.state_dict().items()}
                            if config.route == "direct_sft"
                            else trainable.trainable_state_dict()
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict()
                        if config.route == "direct_sft"
                        else None,
                        "rng_by_rank": rng_by_rank,
                        "nonfinite_gradient_steps": nonfinite_gradient_steps,
                        "nonfinite_gradient_steps_known_since": nonfinite_gradient_steps_known_since,
                        "nonfinite_gradient_steps_since_known": nonfinite_gradient_steps_since_known,
                        "retention_state": (
                            {
                                "format": retention_resume_format,
                                "protocol_id": config.protocol_id,
                                "atomic_replay_identity": dict(atomic_replay_identity),
                                "sampler": retention_sampler.metadata(),
                                "cursor": retention_sampler.cursor_metadata(step),
                                "teacher_identity": resolved_config["teacher_identity"],
                                **({"vector422_sources": resolved_config["vector422_sources"]}
                                   if config.protocol_id == VECTOR422_PROTOCOL_ID else {}),
                            }
                            if is_retention
                            else None
                        ),
                        "selection_state": {
                            "format": "s5_composition_loco_selection_state_v1",
                            "config_sha256": config.computed_sha256,
                            "route": config.route,
                            "fold_id": config.fold_id,
                            "heldout_task": config.heldout_task,
                            "selection_metric": (
                                "eight_task_equal_weight_correct_count"
                                if _is_equal_task_selection(config)
                                else "visible_composition_macro_accuracy"
                            ),
                            "selection_denominator": (
                                4_000 if _is_equal_task_selection(config) else 2_000
                            ),
                            "selection_group_weights": (
                                {"atomic": 0.5, "visible_composition": 0.5}
                                if _is_equal_task_selection(config)
                                else None
                            ),
                            "visible_tasks": list(config.train_tasks),
                            "validation_bank_identity": dict(validation.metadata),
                            "validation_steps": list(validation_steps),
                            "best_step": best_step,
                            "best_score": best_score,
                            "best_correct": best_correct,
                            "evaluations": list(eval_records),
                            "atomic_validation_bank_identity": (
                                None if atomic_validation is None else dict(atomic_validation.metadata)
                            ),
                            "atomic_validation_steps": (
                                [] if atomic_validation is None else list(atomic_validation_steps)
                            ),
                            "atomic_evaluations": list(atomic_eval_records),
                        },
                    },
                )
                _append_jsonl(
                    checkpoint_manifest_path,
                    {
                        "kind": "full_resume",
                        "path": resume_output.name,
                        "global_step": step,
                        "payload_sha256": sha256_file(resume_output),
                        "payload_bytes": resume_output.stat().st_size,
                        "config_sha256": config.computed_sha256,
                    },
                )
            _barrier()
    retention_atomic_selected_step: int | None = None
    if is_retention and not stage2_smoke:
        # Visible selection is now frozen.  Evaluate atomic retention exactly
        # once on that selected model, rather than spending evaluation budget
        # on every visible checkpoint or silently reporting the last step.
        if best_step is None or int(best_step) <= 0:
            raise RuntimeError("retention completed without a selected visible checkpoint")
        selected_checkpoint = _expected_model_checkpoint(output, int(best_step))
        selected_payload = torch.load(
            output / "checkpoints" / f"model_step{int(best_step):06d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        selected_state = selected_payload.get("model_state_dict")
        if not isinstance(selected_state, Mapping):
            raise ValueError(
                "Direct retention selected checkpoint has no model_state_dict"
            )
        target = trainable.module if hasattr(trainable, "module") else trainable
        if config.route == "direct_sft":
            target.load_state_dict(selected_state, strict=True)
        else:
            # Model-only HyperNet checkpoints intentionally contain only
            # trainable parameters; the immutable Base remains in target.
            target.load_trainable_state_dict(selected_state)
        trainable.eval()
        evaluate_atomic(int(best_step), selected_checkpoint)
        _barrier()
        retention_atomic_selected_step = int(best_step)
    if rank == 0:
        result_atomic_steps = (
            [0, retention_atomic_selected_step]
            if is_retention and retention_atomic_selected_step is not None
            else (
                list(atomic_validation_steps)
                if atomic_validation is not None
                else []
            )
        )
        if training_receipt is not None:
            for source, expected_sha in training_receipt["source_files_sha256"].items():
                if sha256_file(_resolve(source)) != expected_sha:
                    raise ValueError("LR-half source changed during training; refusing complete/selection publication")
        weight_circuit_details = (
            _weight_circuit_selection_details(eval_records, atomic_eval_records, best_step)
            if _is_equal_task_selection(config) and best_step is not None
            else None
        )
        selected = {
            "format": "s5_composition_loco_selection_v1",
            **({"diagnostic_only": True, "stage2_smoke": True} if stage2_smoke else {}),
            "status": "complete" if total_steps == run_total_steps else "partial",
            "config_sha256": config.computed_sha256,
            "route": config.route,
            "fold_id": config.fold_id,
            "heldout_task": config.heldout_task,
            "visible_tasks": list(config.train_tasks),
            "selection_metric": (
                "eight_task_equal_weight_correct_count"
                if _is_equal_task_selection(config)
                else "visible_composition_macro_accuracy"
            ),
            "selection_denominator": 4_000 if _is_equal_task_selection(config) else 2_000,
            "selection_group_weights": (
                {"atomic": 0.5, "visible_composition": 0.5}
                if _is_equal_task_selection(config)
                else None
            ),
            "selected_step": best_step,
            "selected_score": best_score,
            "selected_correct": best_correct,
            "step_zero_reference_only": True,
            "validation_steps": list(validation_steps),
            "evaluations": eval_records,
            "nonfinite_gradient_steps": nonfinite_gradient_steps,
            "nonfinite_gradient_steps_known_since": nonfinite_gradient_steps_known_since,
            "nonfinite_gradient_steps_since_known": nonfinite_gradient_steps_since_known,
            "total_steps": total_steps,
            "total_padded_tokens": total_steps * GLOBAL_BATCH_SIZE * 128,
            "atomic_validation": {
                "enabled": atomic_validation is not None,
                "selection_uses_atomic": _is_equal_task_selection(config),
                # Retention deliberately evaluates the atomic bank only at
                # step zero during training and once more on the selected
                # visible-task checkpoint.  Keep the published schedule
                # aligned with the records rather than claiming per-step
                # atomic evaluations that were never run.
                "steps": result_atomic_steps,
                "bank_identity": (
                    None if atomic_validation is None else dict(atomic_validation.metadata)
                ),
                "evaluations": list(atomic_eval_records),
            },
        }
        if weight_circuit_details is not None:
            selected.update(
                {
                    "selected_visible_correct": weight_circuit_details["visible_correct"],
                    "selected_visible_total": weight_circuit_details["visible_total"],
                    "selected_visible_score": weight_circuit_details["visible_score"],
                    "selected_atomic_correct": weight_circuit_details["atomic_correct"],
                    "selected_atomic_total": weight_circuit_details["atomic_total"],
                    "selected_atomic_score": weight_circuit_details["atomic_score"],
                    "selected_correct": weight_circuit_details["selection_correct"],
                    "selected_total": weight_circuit_details["selection_total"],
                    "selected_score": weight_circuit_details["selection_score"],
                }
            )
        if raw_cache_repair is not None:
            selected["cache_repair"] = dict(raw_cache_repair)
        if total_steps == run_total_steps and best_step is None:
            raise RuntimeError("composition LOCO completed without a selectable checkpoint")
        _atomic_json(output / "selection.json", selected)
        complete = {
            "format": "s5_composition_loco_complete_v1",
            **({"diagnostic_only": True, "stage2_smoke": True} if stage2_smoke else {}),
            "status": "complete" if total_steps == run_total_steps else "partial",
            "config_sha256": config.computed_sha256,
            "global_step": total_steps,
            "selected_step": best_step,
            "nonfinite_gradient_steps": nonfinite_gradient_steps,
            "nonfinite_gradient_steps_known_since": nonfinite_gradient_steps_known_since,
            "nonfinite_gradient_steps_since_known": nonfinite_gradient_steps_since_known,
            "protocol_id": config.protocol_id,
            "heldout_task": config.heldout_task,
            "route": config.route,
            "atomic_validation_steps": result_atomic_steps,
        }
        if raw_cache_repair is not None:
            complete["cache_repair"] = dict(raw_cache_repair)
        _atomic_json(
            output / "complete.json",
            complete,
        )
    _barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return {"status": "complete" if total_steps == run_total_steps else "partial", "selected_step": best_step}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--stage2-smoke", action="store_true")
    args = parser.parse_args(argv)
    train(
        load_config(_resolve(args.config)),
        max_steps=args.max_steps,
        resume_path=None if args.resume is None else _resolve(args.resume),
        stage2_smoke=args.stage2_smoke,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["train", "main"]
