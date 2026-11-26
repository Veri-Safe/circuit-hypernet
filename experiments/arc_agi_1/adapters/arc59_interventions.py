"""Evaluation-only ARC59 value interventions and explicit F3 branch removal.

Raw value replacement deliberately leaves circuit tokens present. Group removal
acts after encoding, so learned encoder biases cannot leak through a removed
group, and masks attention keys as well as zeroing their values.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

from .contracts import semantic_sha256, sha256_file
from .fusion import FusionInputs
from .representations import validate_raw_vector59


GROUPS = ("global", "layer", "topology")
GROUP_DIMENSIONS = {
    "global": tuple(range(0, 11)),
    "layer": tuple(range(11, 47)),
    "topology": tuple(range(47, 59)),
}
_SPEC_KEYS = {
    "version", "name", "groups", "vector_treatment", "dimensions",
    "statistics_path", "donor_seed",
}


@dataclass(frozen=True)
class Arc59Intervention:
    name: str
    groups: tuple[str, ...]
    vector_treatment: str
    dimensions: tuple[int, ...]
    donor_seed: int
    statistics_path: str | None
    statistics_sha256: str | None
    specification_path: str
    specification: dict[str, Any]
    specification_file_sha256: str
    mean: Tensor | None = field(default=None, repr=False, compare=False)
    donor_vectors: Tensor | None = field(default=None, repr=False, compare=False)

    @property
    def condition_sha256(self) -> str:
        # Content identity is independent of where a job stages the specification
        # and statistics. The statistics digest binds the donor row ordering.
        return semantic_sha256({
            "version": 1,
            "name": self.name,
            "groups": self.groups,
            "vector_treatment": self.vector_treatment,
            "dimensions": self.dimensions,
            "donor_seed": self.donor_seed,
            "statistics_sha256": self.statistics_sha256,
        })

    def manifest_record(self) -> dict[str, Any]:
        return {
            "version": 1,
            "name": self.name,
            "groups": list(self.groups),
            "vector_treatment": self.vector_treatment,
            "dimensions": list(self.dimensions),
            "donor_seed": self.donor_seed,
            "donor_assignment": "sha256_seed_query_index_mod_training_rows_v1",
            "statistics_path": self.statistics_path,
            "statistics_sha256": self.statistics_sha256,
            "condition_sha256": self.condition_sha256,
            "specification": self.specification,
            "specification_sha256": semantic_sha256(self.specification),
            "specification_path": self.specification_path,
            "specification_file_sha256": self.specification_file_sha256,
        }

    def prediction_metadata(self) -> dict[str, Any]:
        return {
            "circuit_intervention_id": self.condition_sha256,
            "circuit_group_presence": {group: group in self.groups for group in GROUPS},
        }

    def donor_indices(self, query_indices: Sequence[int]) -> tuple[int, ...]:
        if self.donor_vectors is None:
            raise ValueError("Donor assignment requires training donor_vectors")
        indices = []
        for query_index in query_indices:
            if type(query_index) is not int or query_index < 0:
                raise ValueError("query_indices must be nonnegative integers")
            identity = f"arc59_donor_v1:{self.donor_seed}:{query_index}".encode("ascii")
            digest = hashlib.sha256(identity).digest()
            indices.append(int.from_bytes(digest[:8], "big") % len(self.donor_vectors))
        return tuple(indices)

    def apply_vectors(self, vectors: Tensor, *, query_indices: Sequence[int]) -> Tensor:
        validate_raw_vector59(vectors)
        if len(query_indices) != len(vectors):
            raise ValueError("query_indices must align with the vector batch")
        if any(type(index) is not int or index < 0 for index in query_indices):
            raise ValueError("query_indices must be nonnegative integers")
        if self.vector_treatment == "matched":
            return vectors
        output = vectors.clone()
        dimensions = list(self.dimensions)
        if self.vector_treatment == "zero":
            output[:, dimensions] = 0.0
        elif self.vector_treatment == "mean":
            assert self.mean is not None
            output[:, dimensions] = self.mean.to(device=vectors.device)[dimensions]
        elif self.vector_treatment == "donor":
            assert self.donor_vectors is not None
            replacements = self.donor_vectors[list(self.donor_indices(query_indices))]
            output[:, dimensions] = replacements[:, dimensions].to(device=vectors.device)
        else:
            raise ValueError(f"Unsupported vector treatment: {self.vector_treatment}")
        return output


def load_intervention(path: str | Path) -> Arc59Intervention:
    source = Path(path).resolve()
    payload_bytes = source.read_bytes()
    specification = json.loads(payload_bytes)
    if not isinstance(specification, dict):
        raise ValueError("ARC59 intervention must be a JSON object")
    unknown = set(specification) - _SPEC_KEYS
    if unknown:
        raise ValueError(f"Unknown ARC59 intervention fields: {sorted(unknown)}")
    if type(specification.get("version")) is not int or specification["version"] != 1:
        raise ValueError("ARC59 intervention version must be integer 1")
    name = specification.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("ARC59 intervention name must be a nonempty string")
    groups = specification.get("groups", list(GROUPS))
    if (
        not isinstance(groups, list)
        or any(not isinstance(group, str) or group not in GROUPS for group in groups)
        or len(set(groups)) != len(groups)
    ):
        raise ValueError("groups must be a unique subset of global, layer, topology")
    treatment = specification.get("vector_treatment", "matched")
    if not isinstance(treatment, str) or treatment not in {"matched", "zero", "mean", "donor"}:
        raise ValueError("vector_treatment must be matched, zero, mean, or donor")
    dimensions = specification.get("dimensions", [] if treatment == "matched" else list(range(59)))
    if (
        not isinstance(dimensions, list)
        or any(type(dimension) is not int or not 0 <= dimension < 59 for dimension in dimensions)
        or len(set(dimensions)) != len(dimensions)
    ):
        raise ValueError("dimensions must be unique 0-based integer indices in [0, 59)")
    if treatment == "matched" and dimensions:
        raise ValueError("matched vector treatment cannot replace dimensions")
    if treatment != "matched" and not dimensions:
        raise ValueError("Non-matched vector treatment requires at least one dimension")
    donor_seed = specification.get("donor_seed", 42)
    if type(donor_seed) is not int or donor_seed < 0:
        raise ValueError("donor_seed must be a nonnegative integer")
    statistics = specification.get("statistics_path")
    if statistics is not None and (not isinstance(statistics, str) or not statistics.strip()):
        raise ValueError("statistics_path must be a nonempty string")
    if treatment in {"mean", "donor"} and statistics is None:
        raise ValueError("mean/donor treatment requires training statistics_path")
    mean = donors = None
    statistics_path = statistics_sha = None
    if statistics is not None:
        stats_path = Path(statistics)
        if not stats_path.is_absolute():
            stats_path = source.parent / stats_path
        stats_path = stats_path.resolve()
        statistics_path = str(stats_path)
        statistics_sha = sha256_file(stats_path)
        with np.load(stats_path, allow_pickle=False) as bank:
            if not {"mean", "donor_vectors"}.issubset(bank.files):
                raise ValueError("Training statistics NPZ requires mean and donor_vectors")
            mean_array = bank["mean"]
            donor_array = bank["donor_vectors"]
            if mean_array.shape != (59,):
                raise ValueError("Training mean must have shape (59,)")
            if donor_array.ndim != 2 or donor_array.shape[1] != 59 or not len(donor_array):
                raise ValueError("Training donor_vectors must have nonempty shape (N, 59)")
            for name_, array in (("mean", mean_array), ("donor_vectors", donor_array)):
                if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                    raise ValueError(f"Training {name_} must contain finite real values")
            with np.errstate(over="ignore"):
                mean = torch.from_numpy(np.array(mean_array, dtype=np.float32, copy=True))
                donors = torch.from_numpy(np.array(donor_array, dtype=np.float32, copy=True))
            if not torch.isfinite(mean).all() or not torch.isfinite(donors).all():
                raise ValueError("Training statistics must be finite in raw FP32")
    return Arc59Intervention(
        name=name,
        groups=tuple(group for group in GROUPS if group in groups),
        vector_treatment=treatment,
        dimensions=tuple(sorted(dimensions)),
        donor_seed=donor_seed,
        statistics_path=statistics_path,
        statistics_sha256=statistics_sha,
        specification_path=str(source),
        specification=specification,
        specification_file_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        mean=mean,
        donor_vectors=donors,
    )


def validate_intervention_evaluation(
    *, variant_id: str, mode: str, validation_scope: str, lora_scale_multiplier: float,
) -> None:
    if variant_id != "f3_independent_attention":
        raise ValueError("ARC59 interventions require frozen F3 independent attention")
    if mode != "matched":
        raise ValueError("ARC59 interventions require mode=matched; absent/shuffled remain native anchors")
    if validation_scope != "full" or lora_scale_multiplier != 1.0:
        raise ValueError("ARC59 interventions require full evaluation and LoRA multiplier 1.0")


def _gate_fusion(fusion: FusionInputs, groups: tuple[str, ...]) -> FusionInputs:
    if groups == GROUPS:
        return fusion
    if fusion.memory_tokens is None or fusion.memory_mask is None or fusion.encoded is None:
        raise ValueError("ARC59 group gating requires structured F3 circuit memory")
    token_groups = torch.tensor(
        ["global" in groups, "topology" in groups],
        dtype=torch.bool,
        device=fusion.memory_mask.device,
    )
    memory_mask = fusion.memory_mask & token_groups[None, :]
    memory_tokens = fusion.memory_tokens.masked_fill(~memory_mask[:, :, None], 0.0)
    direct = fusion.direct_residual if "layer" in groups else torch.zeros_like(fusion.direct_residual)
    encoded = fusion.encoded
    encoded = replace(
        encoded,
        global_context=encoded.global_context if "global" in groups else torch.zeros_like(encoded.global_context),
        topology_context=encoded.topology_context if "topology" in groups else torch.zeros_like(encoded.topology_context),
        layer_context=encoded.layer_context if "layer" in groups else torch.zeros_like(encoded.layer_context),
        layer_mask=encoded.layer_mask if "layer" in groups else torch.zeros_like(encoded.layer_mask),
        circuit_present=encoded.circuit_present if groups else torch.zeros_like(encoded.circuit_present),
    )
    return FusionInputs(direct, memory_tokens, memory_mask, encoded)


@contextmanager
def scoped_group_gating(model: Any, intervention: Arc59Intervention) -> Iterator[None]:
    """Temporarily gate fusion.encode; never change checkpoint state or training.

    The PG calls ``fusion.encode`` directly, bypassing nn.Module.forward hooks.
    Restoring the instance dictionary also preserves the original method binding.
    """
    if model.variant_id != "f3_independent_attention":
        raise ValueError("ARC59 group gating only supports F3 independent attention")
    if model.training:
        raise ValueError("ARC59 group gating is evaluation-only; call model.eval()")
    fusion = model.fusion
    original = fusion.encode
    had_override = "encode" in fusion.__dict__
    previous_override = fusion.__dict__.get("encode")

    def encode(*args: Any, **kwargs: Any) -> FusionInputs:
        return _gate_fusion(original(*args, **kwargs), intervention.groups)

    fusion.encode = encode
    try:
        yield
    finally:
        if had_override:
            fusion.encode = previous_override
        else:
            delattr(fusion, "encode")
