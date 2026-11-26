"""Frozen, pre-fusion token representations for the four-way probe.

The historical loaders bind the three atomic-stage checkpoints.  This adapter
changes only the observation: query tokens are returned before mean pooling.
It never forwards labels, answers, task IDs or query tokens to the circuit path.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Mapping

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[3]
ROUTE_WIDTHS = {
    "base_query2048": 2048,
    "hn_query1024": 1024,
    "query_only_hn1024": 1024,
    "circuit1024": 1024,
}


def _import_loader(name: str):
    # Imports are delayed so CPU tests do not require the server checkpoint
    # files, matplotlib/resource, or the native Base runtime dependencies.
    for directory in (ROOT / "s5_lean_circuit/scripts", ROOT / "temp_s"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Cannot import frozen probe loader {name!r}. Synchronize the full "
            "server source revision and install its existing dependencies."
        ) from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _verified_file(path: Path, expected: str) -> dict:
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(f"Frozen checkpoint hash mismatch: {_relative(path)}")
    return {"path": _relative(path), "sha256": actual, "bytes": path.stat().st_size}


def _load_route(route: str, device: torch.device) -> tuple[nn.Module, dict]:
    if route == "base_query2048":
        loader = _import_loader("base_query_probe_representation")
        model = loader.load_base(device)
        identity = deepcopy(model._base_query_probe_identity)
        identity["source_loader_representation"] = identity.pop("representation", None)
        identity.update(step=38000, condition_source="base_query", lora=False)
        return model, identity

    if route == "query_only_hn1024":
        loader = _import_loader("query_only_hn_probe_id_matrix_20261121")
        # This loader verifies canonical qo02, atomic-only metadata and the
        # original query/type state; its receipt hashes the checkpoint bytes.
        return loader.load_query(device)

    loader = _import_loader("analyze_atomic_encoder_task_readout")
    checkpoint = _verified_file(loader.CHECKPOINT, loader.EXPECTED_SHA[loader.CHECKPOINT])
    config = json.loads(loader.CONFIG.read_text(encoding="utf-8"))
    if config["model"]["candidate"] != "A" or config["data"]["prefix"] != 1_200_000:
        raise ValueError("Circuit origin must be atomic Candidate A on 1.2M rows")
    identity = {
        "checkpoint": checkpoint,
        "config": {"path": _relative(loader.CONFIG), "sha256": _file_sha256(loader.CONFIG)},
        "step": 15258,
        "condition_source": "circuit",
        "base_identity": deepcopy(config["base"]),
        "training_cache": config["data"]["cache_path"],
        "atomic_only_origin": "a_p03@15258",
    }
    if route == "hn_query1024":
        model = _import_loader("query_atomic_only_probe").load_query(device)
        return model, identity

    if route != "circuit1024":
        raise ValueError(f"Unknown frozen encoder route: {route}")
    payload = torch.load(loader.CHECKPOINT, weights_only=True, mmap=True, map_location="cpu")
    if payload["global_step"] != 15258 or payload["candidate"] != "A":
        raise ValueError("Circuit origin is not a_p03@15258")
    state = payload["model_state_dict"]
    keys = ("network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias")
    model = loader.make_encoder(
        {key: state["circuit_encoder." + key] for key in keys},
        state["type_embeddings.circuit"],
    )
    # Restore even unused type parameters so the frozen-state audit observes
    # the actual checkpoint, not randomly initialized unused module entries.
    model.type_embeddings.load_state_dict(
        {key: state["type_embeddings." + key] for key in model.type_embeddings.state_dict()},
        strict=True,
    )
    if model.input_dim != 397 or not model.add_type_embedding:
        raise ValueError("Expected the 397D Candidate A encoder with its original type embedding")
    del state, payload
    return model, identity


class FrozenEncoder:
    """Return detached FP32 tokens and a bool validity mask without pooling.

    ``encode`` accepts a minibatch mapping.  Query routes read exactly
    ``query_ids`` and ``query_lengths``; the circuit route reads ``vectors``.
    ``state_digest`` includes parameters and buffers for a before/after check.
    """

    def __init__(self, route: str, device: str | torch.device):
        if route not in ROUTE_WIDTHS:
            raise ValueError(f"Unknown frozen encoder route: {route}")
        self.route = route
        self.device = torch.device(device)
        self.width = ROUTE_WIDTHS[route]
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        self.model, source_identity = _load_route(route, self.device)
        self.model = self.model.float().eval().requires_grad_(False).to(self.device)
        self.identity = deepcopy(source_identity)
        self.identity.update({
            "route": route,
            "representation": self._attachment(),
            "width": self.width,
            "pooling": "none",
            "precision": "FP32; no AMP/TF32",
            "frozen": True,
            "encoder_updates": 0,
            "input_fields": ["vectors"] if route == "circuit1024" else ["query_ids", "query_lengths"],
        })
        self._check_frozen()

    def _attachment(self) -> str:
        if self.route == "base_query2048":
            return "Native transformer.ln_f output before lm_head; all valid query tokens; no LoRA"
        if self.route == "circuit1024":
            return "CandidateA 397->2048->1024 + original circuit type embedding; one token before fusion"
        return "QueryEncoder output_norm + validity mask; original pre-transformer query type; before fusion/PG"

    def _check_frozen(self) -> None:
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("Encoder must remain frozen and in eval mode")
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError("Frozen encoder parameters must remain FP32")

    def _query_inputs(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        ids, lengths = batch["query_ids"], batch["query_lengths"]
        integral = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
        if ids.ndim != 2 or lengths.ndim != 1 or ids.shape[0] != lengths.shape[0] or not ids.shape[0]:
            raise ValueError("Expected nonempty query IDs [B,L] and lengths [B]")
        if ids.dtype not in integral or lengths.dtype not in integral:
            raise ValueError("Query IDs and lengths must be integer tensors")
        width = int(lengths.max())
        if int(lengths.min()) < 1 or width > ids.shape[1] or width > 128:
            raise ValueError("Query length must be positive and fit the original 128-token context")
        ids = ids[:, :width].to(device=self.device, dtype=torch.long)
        mask = torch.arange(width, device=self.device)[None, :] < lengths.to(self.device)[:, None]
        return ids, mask

    def encode(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        self._check_frozen()
        # Trainable probe layers need ordinary detached tensors.  Producing an
        # inference tensor would prevent their backward pass from saving it.
        with torch.inference_mode(False), torch.no_grad(), torch.autocast(
            device_type=self.device.type, enabled=False
        ):
            if self.route == "circuit1024":
                vectors = batch["vectors"]
                if vectors.ndim != 2 or vectors.shape[1] != 397 or not vectors.shape[0]:
                    raise ValueError("Circuit vectors must have shape [B,397]")
                vectors = vectors.to(device=self.device, dtype=torch.float32)
                if not torch.isfinite(vectors).all():
                    raise ValueError("Circuit vectors contain nonfinite entries")
                tokens = self.model(vectors)
                mask = torch.ones((vectors.shape[0], 1), device=self.device, dtype=torch.bool)
            else:
                ids, mask = self._query_inputs(batch)
                if self.route == "base_query2048":
                    captured: list[Tensor] = []
                    handle = self.model.transformer.ln_f.register_forward_hook(
                        lambda _module, _inputs, output: captured.append(output)
                    )
                    try:
                        # Keep the native model forward, causal mask and final
                        # normalization unchanged.  Observe, do not replace it.
                        self.model(ids)
                    finally:
                        handle.remove()
                    if len(captured) != 1:
                        raise ValueError("Expected exactly one native final-normalization output")
                    tokens = captured[0]
                else:
                    tokens = self.model(ids, mask, add_type_embedding=True)
            expected = (*mask.shape, self.width)
            if tokens.shape != expected or tokens.dtype != torch.float32:
                raise ValueError(f"Unexpected encoder token shape/dtype; expected {expected} FP32")
            tokens = tokens.masked_fill(~mask.unsqueeze(-1), 0)
            if not torch.isfinite(tokens).all():
                raise ValueError("Encoder produced nonfinite valid token representations")
            # clone also detaches storage from any loader-owned activation cache.
            return tokens.detach().clone(), mask.detach().clone()

    def state_digest(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()
