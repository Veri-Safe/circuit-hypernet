"""Condition-safe graph inputs and inductive S5 circuit encoders.

This module deliberately lives beside, rather than inside, the historical
Vector397 path.  ``GraphRecord`` is the in-memory contract for the new
512-budget *unpruned selected-node* artifact.  It keeps all non-zero signed
edges, including edges whose non-negative influence is zero; the legacy
``S5GraphPayloadV2`` validator is intentionally not relaxed here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import Tensor, nn


FEATURE_INPUT_DIM = 2080 + 1 + 2048
FEATURE_DEF_HIDDEN = 512
NODE_WIDTH = 256
OUTPUT_WIDTH = 1024
FEATURE_BUDGET = 512
NODE_KINDS = {"token": 0, "feature": 1, "error": 2, "root": 3}
NODE_KIND_COUNT = 4
REGIONS = {"prompt": 0, "generated": 1, "none": 2}
REGION_COUNT = 3
READOUT_GROUPS = 19


@dataclass(frozen=True)
class FeatureDefinitionBank:
    """Frozen transcoder rows used by the shared inductive feature encoder."""

    w_enc: Tensor  # [layers, features, 2080]
    b_enc: Tensor  # [layers, features]
    w_dec: Tensor  # [layers, features, 2048]

    def validate(self) -> None:
        if self.w_enc.ndim != 3 or self.w_dec.ndim != 3 or self.b_enc.ndim != 2:
            raise ValueError("feature bank tensors have invalid rank")
        if self.w_enc.shape[:2] != self.w_dec.shape[:2] or self.w_enc.shape[:2] != self.b_enc.shape:
            raise ValueError("feature bank layer/feature shapes do not align")
        if self.w_enc.shape[-1] != 2080 or self.w_dec.shape[-1] != 2048:
            raise ValueError("S5 feature definitions must be W_enc[2080], W_dec[2048]")
        if not all(torch.isfinite(tensor).all() for tensor in (self.w_enc, self.b_enc, self.w_dec)):
            raise ValueError("feature bank contains non-finite values")

    @property
    def layers(self) -> int:
        return int(self.w_enc.shape[0])

    @property
    def features_per_layer(self) -> int:
        return int(self.w_enc.shape[1])

    def lookup(self, layers: Tensor, feature_ids: Tensor, *, device: torch.device) -> Tensor:
        """Return raw ``[N,4129]`` rows; layer and ID remain lookup metadata."""

        if self.w_enc.ndim != 3 or self.w_dec.ndim != 3 or self.b_enc.ndim != 2:
            raise ValueError("feature bank tensors have invalid rank")
        if self.w_enc.shape[:2] != self.w_dec.shape[:2] or self.w_enc.shape[:2] != self.b_enc.shape:
            raise ValueError("feature bank layer/feature shapes do not align")
        if self.w_enc.shape[-1] != 2080 or self.w_dec.shape[-1] != 2048:
            raise ValueError("S5 feature definitions must be W_enc[2080], W_dec[2048]")
        layers = torch.as_tensor(layers, dtype=torch.long, device="cpu")
        feature_ids = torch.as_tensor(feature_ids, dtype=torch.long, device="cpu")
        if layers.shape != feature_ids.shape:
            raise ValueError("feature layer and feature ID shapes do not align")
        if layers.numel():
            if int(layers.min()) < 0 or int(layers.max()) >= self.layers:
                raise ValueError("feature layer is outside the frozen bank")
            if int(feature_ids.min()) < 0 or int(feature_ids.max()) >= self.features_per_layer:
                raise ValueError("feature ID is outside the frozen bank")
        index_device = self.w_enc.device
        index_layers = layers.to(device=index_device)
        index_ids = feature_ids.to(device=index_device)
        enc = self.w_enc[index_layers, index_ids]
        bias = self.b_enc[index_layers, index_ids].unsqueeze(-1)
        dec = self.w_dec[index_layers, index_ids]
        return torch.cat((enc, bias, dec), dim=-1).detach().to(device=device)

    def to(self, device: torch.device | str) -> "FeatureDefinitionBank":
        """Move the frozen bank once; it is not part of trainable state."""

        return FeatureDefinitionBank(
            w_enc=self.w_enc.detach().to(device=device),
            b_enc=self.b_enc.detach().to(device=device),
            w_dec=self.w_dec.detach().to(device=device),
        )


@dataclass(frozen=True)
class GraphRecord:
    """One condition-safe selected-node graph before graph-encoder batching.

    Node IDs are local to this record.  Edge direction is ``source -> target``
    and the edge tensors may contain ``signed != 0, influence == 0`` rows.
    """

    node_kind: Tensor  # int8, [N], values 0 token / 1 feature / 2 error / 3 root
    node_layer: Tensor  # int16, [N], -1 for token/root
    node_position: Tensor  # int16, [N]
    node_feature_id: Tensor  # int32, -1 for non-feature
    node_activation: Tensor  # fp32, [N]
    node_influence: Tensor  # fp32, [N]
    node_region: Tensor  # int8, prompt/generated/none
    edge_source: Tensor  # int32, [E]
    edge_target: Tensor  # int32, [E]
    edge_signed: Tensor  # fp32, [E]
    edge_influence: Tensor  # fp32, [E]
    condition_length: int
    event_key: str = ""

    @classmethod
    def from_symbols(
        cls,
        *,
        node_kind: Sequence[str],
        node_layer: Tensor,
        node_position: Tensor,
        node_feature_id: Tensor,
        node_activation: Tensor,
        node_influence: Tensor,
        node_region: Sequence[str],
        edge_source: Tensor,
        edge_target: Tensor,
        edge_signed: Tensor,
        edge_influence: Tensor,
        condition_length: int,
        event_key: str = "",
    ) -> "GraphRecord":
        try:
            kinds = torch.tensor([NODE_KINDS[value] for value in node_kind], dtype=torch.int8)
            regions = torch.tensor([REGIONS[value] for value in node_region], dtype=torch.int8)
        except KeyError as error:
            raise ValueError(f"unknown graph symbol: {error.args[0]}") from error
        return cls(
            node_kind=kinds,
            node_layer=node_layer,
            node_position=node_position,
            node_feature_id=node_feature_id,
            node_activation=node_activation,
            node_influence=node_influence,
            node_region=regions,
            edge_source=edge_source,
            edge_target=edge_target,
            edge_signed=edge_signed,
            edge_influence=edge_influence,
            condition_length=condition_length,
            event_key=event_key,
        )

    @property
    def node_count(self) -> int:
        return int(self.node_kind.numel())

    @property
    def edge_count(self) -> int:
        return int(self.edge_source.numel())

    @property
    def feature_count(self) -> int:
        return int((self.node_kind == NODE_KINDS["feature"]).sum().item())

    def validate(self, *, feature_budget: int = FEATURE_BUDGET) -> None:
        node_fields = (
            self.node_kind,
            self.node_layer,
            self.node_position,
            self.node_feature_id,
            self.node_activation,
            self.node_influence,
            self.node_region,
        )
        if any(field.ndim != 1 or field.shape[0] != self.node_count for field in node_fields):
            raise ValueError("graph node fields are not aligned")
        edge_fields = (self.edge_source, self.edge_target, self.edge_signed, self.edge_influence)
        if any(field.ndim != 1 or field.shape[0] != self.edge_count for field in edge_fields):
            raise ValueError("graph edge fields are not aligned")
        if self.node_count == 0:
            raise ValueError("graph must contain at least one node")
        if int(self.node_kind.min()) < 0 or int(self.node_kind.max()) >= NODE_KIND_COUNT:
            raise ValueError("node kind is invalid")
        if int(self.node_region.min()) < 0 or int(self.node_region.max()) >= REGION_COUNT:
            raise ValueError("node region is invalid")
        if self.feature_count > int(feature_budget):
            raise ValueError(f"graph has {self.feature_count} feature nodes; budget is {feature_budget}")
        if self.edge_count:
            if int(self.edge_source.min()) < 0 or int(self.edge_target.min()) < 0:
                raise ValueError("edge endpoint is negative")
            if int(self.edge_source.max()) >= self.node_count or int(self.edge_target.max()) >= self.node_count:
                raise ValueError("edge endpoint is outside the graph")
            if bool(torch.any(self.edge_source == self.edge_target)):
                raise ValueError("self edges are not allowed")
            if not torch.isfinite(self.edge_signed).all() or not torch.isfinite(self.edge_influence).all():
                raise ValueError("edge attributes are non-finite")
            if bool(torch.any(self.edge_influence < 0)):
                raise ValueError("nonnegative edge influence cannot be negative")
            # Do not reject signed nonzero / zero-influence edges: they are an
            # intentional part of the unpruned graph contract.
        if int(self.condition_length) < 0:
            raise ValueError("condition_length cannot be negative")


@dataclass(frozen=True)
class GraphBatch:
    """Concatenated variable-size graphs with explicit graph boundaries."""

    node_kind: Tensor
    node_layer: Tensor
    node_position: Tensor
    node_feature_id: Tensor
    node_activation: Tensor
    node_influence: Tensor
    node_region: Tensor
    edge_source: Tensor
    edge_target: Tensor
    edge_signed: Tensor
    edge_influence: Tensor
    node_ptr: Tensor
    edge_ptr: Tensor
    condition_lengths: Tensor
    event_keys: tuple[str, ...]

    @classmethod
    def from_records(cls, records: Sequence[GraphRecord]) -> "GraphBatch":
        if not records:
            raise ValueError("GraphBatch requires at least one graph")
        for record in records:
            record.validate()
        node_parts: dict[str, list[Tensor]] = {name: [] for name in (
            "node_kind", "node_layer", "node_position", "node_feature_id",
            "node_activation", "node_influence", "node_region",
        )}
        edge_parts: dict[str, list[Tensor]] = {name: [] for name in (
            "edge_source", "edge_target", "edge_signed", "edge_influence",
        )}
        node_ptr = [0]
        edge_ptr = [0]
        for record in records:
            for name in node_parts:
                node_parts[name].append(getattr(record, name).detach().cpu().contiguous())
            offset = node_ptr[-1]
            edge_parts["edge_source"].append((record.edge_source + offset).detach().cpu().long())
            edge_parts["edge_target"].append((record.edge_target + offset).detach().cpu().long())
            edge_parts["edge_signed"].append(record.edge_signed.detach().cpu().float())
            edge_parts["edge_influence"].append(record.edge_influence.detach().cpu().float())
            node_ptr.append(offset + record.node_count)
            edge_ptr.append(edge_ptr[-1] + record.edge_count)
        return cls(
            **{name: torch.cat(parts) for name, parts in node_parts.items()},
            **{
                name: torch.cat(parts) if any(part.numel() for part in parts) else parts[0]
                for name, parts in edge_parts.items()
            },
            node_ptr=torch.tensor(node_ptr, dtype=torch.long),
            edge_ptr=torch.tensor(edge_ptr, dtype=torch.long),
            condition_lengths=torch.tensor([record.condition_length for record in records], dtype=torch.long),
            event_keys=tuple(record.event_key for record in records),
        )

    @property
    def graph_count(self) -> int:
        return int(self.node_ptr.numel() - 1)

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(
            node_kind=self.node_kind.to(device=device),
            node_layer=self.node_layer.to(device=device),
            node_position=self.node_position.to(device=device),
            node_feature_id=self.node_feature_id.to(device=device),
            node_activation=self.node_activation.to(device=device),
            node_influence=self.node_influence.to(device=device),
            node_region=self.node_region.to(device=device),
            edge_source=self.edge_source.to(device=device),
            edge_target=self.edge_target.to(device=device),
            edge_signed=self.edge_signed.to(device=device),
            edge_influence=self.edge_influence.to(device=device),
            node_ptr=self.node_ptr.to(device=device),
            edge_ptr=self.edge_ptr.to(device=device),
            condition_lengths=self.condition_lengths.to(device=device),
            event_keys=self.event_keys,
        )

    def validate(self) -> None:
        if self.node_ptr.ndim != 1 or self.edge_ptr.ndim != 1:
            raise ValueError("graph pointers must be one-dimensional")
        if self.node_ptr.numel() != self.edge_ptr.numel() or self.condition_lengths.numel() != self.graph_count:
            raise ValueError("graph pointer lengths do not align")
        if int(self.node_ptr[0]) != 0 or int(self.edge_ptr[0]) != 0:
            raise ValueError("graph pointers must start at zero")
        if bool(torch.any(self.node_ptr[1:] < self.node_ptr[:-1])) or bool(torch.any(self.edge_ptr[1:] < self.edge_ptr[:-1])):
            raise ValueError("graph pointers must be monotonic")
        if int(self.node_ptr[-1]) != self.node_kind.numel() or int(self.edge_ptr[-1]) != self.edge_source.numel():
            raise ValueError("graph pointers do not cover concatenated tensors")
        for index in range(self.graph_count):
            start, end = int(self.node_ptr[index]), int(self.node_ptr[index + 1])
            edge_start, edge_end = int(self.edge_ptr[index]), int(self.edge_ptr[index + 1])
            if edge_end > edge_start:
                source = self.edge_source[edge_start:edge_end]
                target = self.edge_target[edge_start:edge_end]
                if int(source.min()) < start or int(target.min()) < start or int(source.max()) >= end or int(target.max()) >= end:
                    raise ValueError("edge crosses graph boundary")


class SharedFeatureDefinitionEncoder(nn.Module):
    """Inductive encoder for frozen ``W_enc + b_enc + W_dec`` definitions."""

    def __init__(self, *, output_width: int = NODE_WIDTH) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_INPUT_DIM, FEATURE_DEF_HIDDEN),
            nn.SiLU(),
            nn.Linear(FEATURE_DEF_HIDDEN, output_width),
        )

    def forward(self, definitions: Tensor) -> Tensor:
        if definitions.ndim != 2 or definitions.shape[1] != FEATURE_INPUT_DIM:
            raise ValueError(f"feature definitions must have shape [N,{FEATURE_INPUT_DIM}]")
        if not torch.isfinite(definitions).all():
            raise ValueError("feature definitions contain non-finite values")
        return self.network(definitions)


class _DirectionalPass(nn.Module):
    """Three relation channels; no learned or saved [E,H] message tensors."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.source = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(3))
        self.update = nn.GRUCell(3 * width + 2, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, initial: Tensor, batch: GraphBatch, *, reverse: bool) -> Tensor:
        source, target = batch.edge_source.long(), batch.edge_target.long()
        if reverse:
            source, target = target, source
            levels = torch.where(batch.node_kind == 3, 0,
                                 torch.where(batch.node_kind == 1, 8 - batch.node_layer, 9))
        else:
            levels = torch.where(batch.node_kind == 3, 9,
                                 torch.where(batch.node_kind == 1, batch.node_layer + 1, 0))
        n = initial.shape[0]
        magnitude = batch.edge_signed.float().abs()
        total_mass = magnitude.new_zeros(n).index_add(0, target, magnitude)
        influence_mass = magnitude.new_zeros(n).index_add(0, target, batch.edge_influence.float())
        denom = total_mass.clamp_min(1e-10)
        influence_denom = influence_mass.clamp_min(1e-10)
        values = (batch.edge_signed.clamp_min(0) / denom[target],
                  (-batch.edge_signed).clamp_min(0) / denom[target],
                  batch.edge_influence / influence_denom[target])
        states = initial
        projected = [layer(initial) for layer in self.source]
        # Level batches contain nodes from every graph, not Python per-graph loops.
        for level in range(10):
            targets = torch.where(levels == level)[0]
            if targets.numel() == 0:
                continue
            edges = torch.where(levels[target] == level)[0]
            local = torch.full((n,), -1, device=initial.device, dtype=torch.long)
            local[targets] = torch.arange(targets.numel(), device=initial.device)
            indices = torch.stack((local[target[edges]], source[edges]))
            aggregates = []
            for value, state in zip(values, projected):
                adjacency = torch.sparse_coo_tensor(
                    indices, value[edges].float(), (targets.numel(), n),
                    device=initial.device, check_invariants=False).coalesce()
                aggregates.append(torch.sparse.mm(adjacency, state.float()).to(initial.dtype))
            previous = states[targets]
            totals = torch.stack((total_mass[targets].log1p(), influence_mass[targets].log1p()), -1)
            message = torch.cat((*aggregates, totals.to(initial.dtype)), -1)
            updated = self.norm(self.update(message, previous) + previous)
            active = torch.bincount(local[target[edges]], minlength=targets.numel()) > 0
            updated = torch.where(active[:, None], updated, previous)
            states = states.index_copy(0, targets, updated)
            projected = [state.index_copy(0, targets, layer(updated))
                         for state, layer in zip(projected, self.source)]
        return states


def graph_group_indices(batch: GraphBatch) -> Tensor:
    graph = torch.repeat_interleave(
        torch.arange(batch.graph_count, device=batch.node_kind.device), batch.node_ptr.diff())
    group = torch.where(batch.node_kind == 1, batch.node_layer,
                        torch.where(batch.node_kind == 2, batch.node_layer + 8,
                                    torch.where(batch.node_kind == 0, batch.node_region + 16, 18)))
    return (graph * READOUT_GROUPS + group).long()


def graph_group_mask(batch: GraphBatch) -> Tensor:
    return torch.bincount(graph_group_indices(batch),
                          minlength=batch.graph_count * READOUT_GROUPS).reshape(-1, READOUT_GROUPS) > 0


class _Readout(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(READOUT_GROUPS, width) * .02)
        self.score = nn.Linear(width, width, bias=False)
        self.output = nn.Sequential(nn.Linear(width + 2, OUTPUT_WIDTH), nn.LayerNorm(OUTPUT_WIDTH))
        self.address = nn.Embedding(READOUT_GROUPS, OUTPUT_WIDTH)

    def forward(self, states: Tensor, batch: GraphBatch) -> Tensor:
        group = graph_group_indices(batch)
        size = batch.graph_count * READOUT_GROUPS
        scores = (self.score(states).float() * self.queries[group % READOUT_GROUPS].float()).sum(-1)
        maxima = scores.new_full((size,), -torch.inf).scatter_reduce(
            0, group, scores.detach(), reduce="amax")
        numer = (scores - maxima[group]).exp()
        denominator = scores.new_zeros(size).index_add(0, group, numer)
        weights = numer / denominator[group].clamp_min(1e-20)
        pooled = states.new_zeros(size, states.shape[-1]).float().index_add(
            0, group, states.float() * weights[:, None])
        count = torch.bincount(group, minlength=size)
        mass = scores.new_zeros(size).index_add(0, group, batch.node_influence.float())
        values = torch.cat((pooled, count.float().log1p()[:, None], mass.log1p()[:, None]), -1)
        result = self.output(values.to(states.dtype)) + self.address.weight.repeat(batch.graph_count, 1)
        return (result * (count > 0)[:, None]).reshape(batch.graph_count, READOUT_GROUPS, OUTPUT_WIDTH)


class S5GraphEncoder(nn.Module):
    """Shared raw-feature definitions, typed positions, bidirectional DAG, 19 tokens."""

    def __init__(self, *, layers: int = 8, feature_count: int = 16384,
                 node_width: int = NODE_WIDTH, message_passing: bool = True) -> None:
        super().__init__()
        if layers != 8 or node_width <= 0:
            raise ValueError("S5 graph encoding requires eight layers")
        self.node_width = node_width
        self.feature_definitions = SharedFeatureDefinitionEncoder(output_width=node_width)
        self.kind_embedding = nn.Embedding(4, node_width)
        self.layer_embedding = nn.Embedding(9, node_width)
        self.region_embedding = nn.Embedding(3, node_width)
        self.scalar = nn.Sequential(nn.Linear(6, node_width), nn.SiLU(), nn.Linear(node_width, node_width))
        # Common components consume the same seed before route-specific modules.
        self.readout = _Readout(node_width * 2)
        self.forward_pass = _DirectionalPass(node_width) if message_passing else None
        self.backward_pass = _DirectionalPass(node_width) if message_passing else None
        self._source_bank = None
        self._bank_key = None
        self._device_bank = None

    def _initial(self, batch: GraphBatch, bank: FeatureDefinitionBank) -> Tensor:
        device = batch.node_kind.device
        key = (device, id(bank), *(getattr(bank, k)._version for k in ("w_enc", "b_enc", "w_dec")))
        if key != self._bank_key:
            bank.validate()
            self._source_bank, self._device_bank, self._bank_key = bank, bank.to(device), key
        feature = torch.where(batch.node_kind == 1)[0]
        if feature.numel() and bool((batch.node_feature_id[feature] >= bank.features_per_layer).any()):
            raise ValueError("feature ID outside dictionary")
        identities = batch.node_layer[feature].long() * bank.features_per_layer + batch.node_feature_id[feature]
        unique, inverse = torch.unique(identities, return_inverse=True)
        definitions = self._device_bank.lookup(
            unique // bank.features_per_layer, unique % bank.features_per_layer, device=device)
        encoded = self.feature_definitions(definitions)
        initial = encoded.new_zeros(batch.node_kind.numel(), self.node_width).index_copy(0, feature, encoded[inverse])
        layer = torch.where(batch.node_layer >= 0, batch.node_layer, 8).long()
        graph = torch.repeat_interleave(torch.arange(batch.graph_count, device=device), batch.node_ptr.diff())
        boundary = batch.condition_lengths[graph]
        lengths = torch.bincount(graph[batch.node_kind == 0], minlength=batch.graph_count)[graph]
        pos = batch.node_position.float().clamp_min(0)
        relative = torch.where(pos < boundary, pos / (boundary - 1).clamp_min(1),
                               (pos - boundary) / (lengths - boundary - 1).clamp_min(1))
        relative = torch.where(batch.node_kind == 3, 0., relative)
        scalar = torch.stack((batch.node_activation, batch.node_influence,
                              batch.node_activation.log1p(), batch.node_influence.log1p(),
                              pos / 127., relative), -1)
        return (initial + self.kind_embedding(batch.node_kind.long()) + self.layer_embedding(layer)
                + self.region_embedding(batch.node_region.long()) + self.scalar(scalar))

    def forward(self, batch: GraphBatch, bank: FeatureDefinitionBank) -> Tensor:
        if batch.node_kind.device.type == "cpu":
            batch.validate()
        batch = batch.to(next(self.parameters()).device)
        initial = self._initial(batch, bank)
        forward = self.forward_pass(initial, batch, reverse=False) if self.forward_pass else initial
        backward = self.backward_pass(initial, batch, reverse=True) if self.backward_pass else initial
        return self.readout(torch.cat((forward, backward), -1), batch)


class S5NodeSetEncoder(S5GraphEncoder):
    """Shares input/readout modules, without unused graph-message parameters."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, message_passing=False)


__all__ = [
    "FEATURE_INPUT_DIM", "FEATURE_BUDGET", "READOUT_GROUPS", "FeatureDefinitionBank",
    "GraphRecord", "GraphBatch", "SharedFeatureDefinitionEncoder", "S5GraphEncoder",
    "S5NodeSetEncoder",
]
