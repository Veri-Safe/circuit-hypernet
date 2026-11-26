"""Explicit frozen Lean HyperNet adapter for PG transcoder diagnostics.

Supports native formal atomic and composition-LOCO model checkpoints with
Circuit append conditioning.  Config/model byte identities are supplied by
the target; historical checkpoint names or hashes are never defaults.  Circuit
events already contain the exact treated model inputs: this module never
normalizes, log-transforms, or substitutes Circuit values.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .hypernet.runtime import S5HyperNet
from .modeling import build_model


TARGET_FORMAT = "hypercombine_pg_transcoder_target_v1"
LOADERS = frozenset({"lean_atomic", "lean_loco"})


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"explicit {label} path is required")
    path = Path(value)
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return result


def _model_state_digest(state: Mapping[str, Tensor]) -> str:
    """Native state-identity algorithm, with BF16-safe raw-byte conversion."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _generation(value: Mapping[str, Any], native_context: int = 128) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("generation must be an explicit object")
    if set(value) - {"n_ctx", "max_new_tokens"}:
        raise ValueError("Lean diagnostic supports only n_ctx and max_new_tokens generation settings")
    result = {"n_ctx": value.get("n_ctx", native_context), "max_new_tokens": value.get("max_new_tokens", 50)}
    if (any(type(item) is not int for item in result.values())
            or not 2 <= result["n_ctx"] <= min(128, native_context)
            or not 1 <= result["max_new_tokens"] <= 128):
        raise ValueError("invalid Lean diagnostic generation budget")
    return result


class LeanPGTranscoderAdapter:
    def __init__(self, core: S5HyperNet, tokenizer: Any, *, device: torch.device,
                 condition: str, gate_multiplier: float, generation: Mapping[str, Any],
                 identity: Mapping[str, Any]) -> None:
        if condition not in {"matched", "absent"}:
            raise ValueError("condition must be matched or absent")
        if not math.isfinite(float(gate_multiplier)) or float(gate_multiplier) < 0:
            raise ValueError("native gate must be finite and nonnegative")
        if core.condition_source != "circuit" or core.fusion_mode != "append":
            raise ValueError("this adapter supports Circuit append HyperNets only")
        if (core.controller.scale_multiplier != 1.0
                or core.controller.target_scope != "all"):
            raise ValueError("source controller must keep native multiplier 1 and all target modules")
        self.hypernet = core
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.condition = condition
        self.gate_multiplier = float(gate_multiplier)
        native_context = int(getattr(getattr(core.base_model, "config", None), "block_size", 128))
        self.generation = _generation(generation, native_context)
        self.identity = deepcopy(dict(identity))
        self.identity.update({"domain": "lean", "condition": condition,
                              "gate_multiplier": self.gate_multiplier,
                              "generation": dict(self.generation),
                              "circuit_values": "exact_model_input_no_transform",
                              "replay_scope": "unconstrained_greedy_diagnostic_not_benchmark"})
        self._closed = False
        self.hypernet.eval()
        for parameter in self.hypernet.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_core(cls, core: S5HyperNet, tokenizer: Any, *, device: torch.device,
                  condition: str = "matched", gate_multiplier: float = 0.5,
                  generation: Mapping[str, Any] | None = None,
                  identity: Mapping[str, Any] | None = None) -> "LeanPGTranscoderAdapter":
        """Inject already-loaded real components; callers own source provenance."""
        return cls(core, tokenizer, device=device, condition=condition,
                   gate_multiplier=gate_multiplier, generation=generation or {},
                   identity=identity or {"source": "from_core_no_file_provenance"})

    def _ready(self) -> None:
        if self._closed:
            raise RuntimeError("Lean PG adapter is closed")
        if any(module.training for module in (self.hypernet, self.hypernet.base_model,
                                              self.hypernet.parameter_generator, self.hypernet.query_encoder)):
            raise RuntimeError("source HyperNet must remain in eval mode")
        if self.hypernet.controller._state is not None:
            raise RuntimeError("LoRA state is already active")

    def mlp(self, layer: int) -> nn.Module:
        self._ready()
        blocks = self.hypernet.parameter_generator.model.blocks
        if type(layer) is not int or not 0 <= layer < len(blocks):
            raise ValueError("PG layer is outside this checkpoint's block depth")
        return blocks[layer].feed_forward

    def _event(self, event: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        self._ready()
        ids = event.get("input_ids")
        if (not isinstance(ids, (list, tuple)) or not ids
                or any(type(value) is not int or value < 0 for value in ids)
                or len(ids) >= self.generation["n_ctx"]):
            raise ValueError("event requires nonempty unpadded prompt-only input_ids within context")
        vocab = int(self.hypernet.base_model.config.vocab_size)
        if any(value >= vocab for value in ids):
            raise ValueError("event token outside source vocabulary")
        present = event.get("circuit_present")
        if type(present) is not bool or present != (self.condition == "matched"):
            raise ValueError("event circuit_present disagrees with explicit target condition")
        values = event.get("circuit_vector")
        if not isinstance(values, (list, tuple)) or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError("event requires an explicit numeric exact-model-input Circuit vector")
        vector = torch.tensor([values], dtype=torch.float32, device=self.device)
        expected = int(self.hypernet.circuit_input_dim)
        if vector.shape != (1, expected) or not bool(torch.isfinite(vector).all()):
            raise ValueError(f"event Circuit vector must be finite with width {expected}")
        return (torch.tensor([ids], dtype=torch.long, device=self.device), vector,
                torch.tensor([present], dtype=torch.bool, device=self.device))

    @torch.inference_mode()
    def generate(self, event: Mapping[str, Any]) -> dict[str, Tensor]:
        ids, vector, present = self._event(event)
        with _autocast(self.device):
            state = self.hypernet.generate_lora(ids, torch.ones_like(ids, dtype=torch.bool), vector,
                circuit_present=present, gate_multiplier=self.gate_multiplier)
        if not isinstance(state, Mapping) or not state or any(
            not isinstance(value, Tensor) or not bool(torch.isfinite(value).all()) for value in state.values()
        ):
            raise ValueError("source generated an invalid LoRA state")
        return dict(state)

    @torch.inference_mode()
    def replay(self, event: Mapping[str, Any], state: Mapping[str, Tensor]) -> dict[str, Any]:
        values, _, _ = self._event(event)
        start = int(values.shape[1])
        first_logits = None
        stop_reason = "max_new_tokens"
        with self.hypernet.controller.activate(state):
            for _ in range(self.generation["max_new_tokens"]):
                if values.shape[1] >= self.generation["n_ctx"]:
                    stop_reason = "context_limit"
                    break
                with _autocast(self.device):
                    logits = self.hypernet.base_logits(values)[:, -1, :]
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("nonfinite downstream replay logits")
                if first_logits is None:
                    first_logits = logits[0].detach().float().clone()
                values = torch.cat((values, logits.argmax(-1, keepdim=True)), dim=1)
                # Native Lean greedy evaluation checks complete decoded text.
                full = self.tokenizer.decode(values[0].tolist())
                marker = next((value for value in ("proof complete", "no goals") if value in full), None)
                if marker:
                    stop_reason = marker.replace(" ", "_")
                    break
        if first_logits is None:
            raise RuntimeError("replay did not execute any source forward")
        return {"logits": first_logits, "token_ids": values[0, start:].tolist(),
                "stop_reason": stop_reason}

    @torch.inference_mode()
    def score_continuations(self, event: Mapping[str, Any], state: Mapping[str, Tensor],
                            continuations: Any) -> list[dict[str, Any]]:
        """Teacher-force explicit candidate tokens under one supplied frozen LoRA.

        Each token is scored from its complete causal prefix.  This also works
        with native Base implementations that expose only last-position logits;
        no generated continuation, context cropping, or PG regeneration occurs.
        """
        prompt, _, _ = self._event(event)
        if (any(module.training for module in self.hypernet.modules())
                or any(parameter.requires_grad for parameter in self.hypernet.parameters())):
            raise RuntimeError("source HyperNet must remain frozen and in eval mode")
        if (self.hypernet.controller.scale_multiplier != 1.0
                or self.hypernet.controller.target_scope != "all"):
            raise RuntimeError("source controller must retain native multiplier 1 and all targets")
        if not isinstance(continuations, (list, tuple)):
            raise ValueError("continuations must be an explicit list or tuple of token sequences")
        vocab = int(self.hypernet.base_model.config.vocab_size)
        candidates = []
        # Validate the entire request before activating LoRA or running Base.
        for candidate in continuations:
            if (not isinstance(candidate, (list, tuple)) or not candidate
                    or any(type(token) is not int or not 0 <= token < vocab for token in candidate)):
                raise ValueError("each continuation requires nonempty integer token_ids within vocabulary")
            if len(candidate) > self.generation["max_new_tokens"]:
                raise ValueError("continuation exceeds max_new_tokens")
            if int(prompt.shape[1]) + len(candidate) > self.generation["n_ctx"]:
                raise ValueError("prompt and full continuation exceed n_ctx")
            candidates.append(list(candidate))
        if not isinstance(state, Mapping) or not state or any(
            not isinstance(value, Tensor) or not bool(torch.isfinite(value).all()) for value in state.values()
        ):
            raise ValueError("scoring requires a finite explicit LoRA state")
        results = []
        with self.hypernet.controller.activate(state):
            for candidate in candidates:
                values = prompt
                token_logprobs = []
                for token in candidate:
                    with _autocast(self.device):
                        logits = self.hypernet.base_logits(values)[:, -1, :]
                    if not bool(torch.isfinite(logits).all()):
                        raise ValueError("nonfinite downstream candidate logits")
                    logprobs = logits.float().log_softmax(dim=-1)
                    if not bool(torch.isfinite(logprobs).all()):
                        raise ValueError("nonfinite downstream candidate log probabilities")
                    token_logprobs.append(float(logprobs[0, token].item()))
                    values = torch.cat((values, values.new_tensor([[token]])), dim=1)
                total = math.fsum(token_logprobs)
                results.append({"token_ids": candidate, "token_logprobs": token_logprobs,
                                "sum_logprob": total, "mean_logprob": total / len(candidate),
                                "token_count": len(candidate), "finite": True})
        return results

    def close(self) -> None:
        if not self._closed:
            self.hypernet.close()
            self._closed = True


def _load_base(config: Mapping[str, Any], *, root: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Config-bound Base loading, including newly retrained source identities."""
    info = config.get("base")
    if not isinstance(info, Mapping):
        raise ValueError("native config requires explicit Base identity")
    architecture_path = _path(root, info.get("architecture_config_path"), "Base architecture")
    checkpoint_path = _path(root, info.get("checkpoint_path"), "Base checkpoint")
    checkpoint_sha = _hash(info.get("checkpoint_sha256"), "Base checkpoint identity")
    state_sha = _hash(info.get("model_state_sha256"), "Base state identity")
    config_sha = _hash(info.get("config_sha256"), "Base config identity")
    if type(info.get("step")) is not int or info["step"] < 0:
        raise ValueError("Base step must be explicitly configured")
    if _sha(checkpoint_path) != checkpoint_sha:
        raise ValueError("Base checkpoint bytes disagree with native config")
    architecture = _json(architecture_path)
    if architecture.get("config_sha256") != config_sha:
        raise ValueError("Base architecture config identity mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if (not isinstance(payload, Mapping) or payload.get("step") != info["step"]
            or payload.get("config_sha256") != config_sha
            or not isinstance(payload.get("model_state_dict"), Mapping)):
        raise ValueError("Base checkpoint envelope mismatch")
    if _model_state_digest(payload["model_state_dict"]) != state_sha:
        raise ValueError("Base model-state identity mismatch")
    model = build_model(architecture["model"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.config.grad_checkpointing = False
    model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {**dict(info), "architecture_file_sha256": _sha(architecture_path)}


def load_adapter(target: Mapping[str, Any], *, root: Path, device: torch.device) -> LeanPGTranscoderAdapter:
    if (target.get("format") != TARGET_FORMAT or target.get("domain") != "lean"
            or target.get("loader") not in LOADERS):
        raise ValueError("unsupported Lean PG transcoder target")
    root, device = Path(root).resolve(), torch.device(device)
    config_path = _path(root, target.get("config_path"), "native HyperNet config")
    checkpoint_path = _path(root, target.get("checkpoint_path"), "HyperNet checkpoint")
    expected_sha = _hash(target.get("checkpoint_sha256"), "HyperNet checkpoint identity")
    if _sha(checkpoint_path) != expected_sha:
        raise ValueError("HyperNet checkpoint SHA256 mismatch")
    config = _json(config_path)
    config_sha = _hash(config.get("config_sha256"), "native config identity")
    if _digest({key: value for key, value in config.items() if key != "config_sha256"}) != config_sha:
        raise ValueError("native HyperNet config semantic checksum mismatch")
    # Native LOCO configs serialize an unused ``extra`` field as JSON null.
    # Treat that exactly like an omitted optional section; model/data remain
    # mandatory mappings and all source/fusion defaults are checked below.
    model, extra, data = (
        config.get("model", {}),
        config.get("extra") or {},
        config.get("data", {}),
    )
    if not all(isinstance(value, Mapping) for value in (model, extra, data)):
        raise ValueError("native model/extra/data config must be objects")
    source = str(extra.get("condition_source", model.get("condition_source", "circuit")))
    fusion = str(extra.get("fusion_mode", "append"))
    if source != "circuit" or fusion != "append":
        raise ValueError("initial Lean adapter supports Circuit append checkpoints only")
    if data.get("input_treatment") == "train_zscore_compact381_v1":
        raise ValueError("compact381 requires an explicit retained-index adapter; not supported by this loader")
    circuit_source = target.get("circuit_source")
    if not isinstance(circuit_source, Mapping) or circuit_source.get("input_treatment") != data.get("input_treatment"):
        raise ValueError("Circuit source must declare the native config input_treatment")
    treatment_identity = config.get("input_treatment_identity")
    if treatment_identity is None and data.get("input_treatment") == "fixed_log1p_v1":
        from .hypernet.circuit_normalizer import FixedLog1pVector397
        treatment_identity = FixedLog1pVector397().artifact_sha256
    if "input_treatment_identity" in circuit_source and circuit_source["input_treatment_identity"] != treatment_identity:
        raise ValueError("Circuit treatment artifact identity mismatch")
    condition = target.get("condition")
    if condition not in {"matched", "absent"}:
        raise ValueError("target requires explicit matched/absent condition")
    generation = _generation(target.get("generation"), int(model.get("n_ctx", 128)))
    gate = float(model["gate_max"])
    if "gate_multiplier" in target and float(target["gate_multiplier"]) != gate:
        raise ValueError("target gate differs from native configured evaluation gate")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, Mapping) or payload.get("config_sha256") != config_sha:
        raise ValueError("HyperNet checkpoint config identity mismatch")
    candidate = str(model.get("candidate", "A")).upper()
    if target["loader"] == "lean_atomic":
        expected = {"format": "s5_hypernet_formal_checkpoint_v1", "protocol_id": config.get("protocol_id"),
                    "candidate": candidate, "group": model.get("group")}
        # Original formal checkpoints bind source/fusion through the config SHA
        # and omit these two redundant envelope fields.  If a newer checkpoint
        # does include them, they must still agree with the native config.
        for key, value in (("condition_source", source), ("fusion_mode", fusion)):
            if key in payload and payload[key] != value:
                raise ValueError(f"native HyperNet checkpoint {key} differs")
    else:
        if config.get("format") != "s5_composition_loco_run_config_v1" or config.get("route") != "circuit_a":
            raise ValueError("Lean LOCO target must be a native Circuit route config")
        expected = {"format": "s5_composition_loco_checkpoint_v1", "protocol_id": config.get("protocol_id"),
                    "route": "circuit_a", "fold_id": config.get("fold_id"), "heldout_task": config.get("heldout_task")}
    if (any(payload.get(key) != value for key, value in expected.items())
            or payload.get("kind") not in {"model_only", "full_resume"}
            or type(payload.get("global_step")) is not int or payload["global_step"] < 0
            or not isinstance(payload.get("model_state_dict"), Mapping)):
        raise ValueError("native HyperNet checkpoint envelope mismatch")
    tokenizer_path = _path(root, target.get("tokenizer_path", data.get("tokenizer_path")), "tokenizer")
    recorded_tokenizer = extra.get("matched_validation", {}).get("tokenizer_sha256")
    tokenizer_sha = _hash(target.get("tokenizer_sha256"), "tokenizer identity")
    if recorded_tokenizer is not None and recorded_tokenizer != tokenizer_sha:
        raise ValueError("target tokenizer differs from native validation tokenizer")
    if _sha(tokenizer_path) != tokenizer_sha:
        raise ValueError("tokenizer SHA256 mismatch")
    from .circuit.model_io import load_tokenizer
    tokenizer = load_tokenizer(tokenizer_path, expected_sha256=tokenizer_sha)
    base, base_identity = _load_base(config, root=root, device=device)
    core = S5HyperNet(base, candidate=candidate, seed=int(model.get("seed", 42)),
                     condition_source=source, fusion_mode=fusion)
    try:
        core.load_trainable_state_dict(payload["model_state_dict"])
        core.to(device).eval()
        adapter = LeanPGTranscoderAdapter(core, tokenizer, device=device, condition=condition,
            gate_multiplier=gate, generation=generation, identity={
                "format": "lean_pg_transcoder_adapter_v1", "loader": target["loader"],
                "config_path": str(config_path), "config_file_sha256": _sha(config_path),
                "config_sha256": config_sha, "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": expected_sha, "global_step": payload["global_step"],
                "candidate": candidate, "condition_source": source, "fusion_mode": fusion,
                "circuit_source": dict(circuit_source), "base": base_identity,
                "tokenizer_path": str(tokenizer_path), "tokenizer_sha256": tokenizer_sha,
                "circuit_dim": 397, "circuit_input_mode": "exact_model_input",
                "layer": target["layer"],
                "mlp_module": f"parameter_generator.model.blocks.{target['layer']}.feed_forward",
                "pg_width": int(core.parameter_generator.model.d_model),
                "pg_layers": len(core.parameter_generator.model.blocks),
                "target_keys": list(core.layout.target_keys),
                "native_gate_multiplier": gate, "external_lora_multiplier": 1.0,
            })
        adapter.mlp(target.get("layer"))
        return adapter
    except BaseException:
        core.close()
        raise


__all__ = ["LeanPGTranscoderAdapter", "load_adapter"]
