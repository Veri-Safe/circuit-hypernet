from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch

from s3circuit.graph_extract import ensure_codecircuit_on_path


PAYLOAD_VERSION = "s3_compact_exact_v1"
PAYLOAD_VECTOR_TYPE = "codecircuit_advanced_graph_features_exact_payload"
DENSE_REFERENCE_BACKEND = "dense_reference"
SPARSE_EXACT_BACKEND = "sparse_exact"
TOPOLOGY_BACKENDS = (DENSE_REFERENCE_BACKEND, SPARSE_EXACT_BACKEND)


def _tensor_to_cpu(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    return torch.as_tensor(value).detach().to("cpu")


def _pruned_indices_from_mask(node_mask: torch.Tensor) -> list[int]:
    pruned_node_indices = node_mask.nonzero().squeeze().tolist()
    if isinstance(pruned_node_indices, int):
        return [pruned_node_indices]
    if pruned_node_indices is None:
        return []
    return [int(item) for item in pruned_node_indices]


def build_compact_payload(graph: Any, node_threshold: float = 0.8) -> dict[str, Any]:
    """Build a sufficient exact payload for CodeCircuit's 49D graph vector.

    The payload stores the exact pruned subgraph and the graph metadata used by
    CodeCircuit's ``advanced_graph_features`` path. It intentionally does not
    store the full dense adjacency matrix.
    """

    ensure_codecircuit_on_path()
    from circuit_tracer.graph import prune_graph

    graph.to("cpu")
    node_mask, _, cumulative_scores = prune_graph(graph, node_threshold=node_threshold)
    pruned_node_indices = _pruned_indices_from_mask(node_mask)
    index = torch.tensor(pruned_node_indices, dtype=torch.long)
    if index.numel() > 0:
        pruned_adjacency = graph.adjacency_matrix.to("cpu")[index][:, index].to_sparse_coo().coalesce()
        pruned_cumulative_scores = cumulative_scores.to("cpu")[index]
    else:
        pruned_adjacency = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=graph.adjacency_matrix.dtype),
            size=(0, 0),
        ).coalesce()
        pruned_cumulative_scores = torch.empty((0,), dtype=cumulative_scores.dtype)

    n_layers = int(graph.cfg.n_layers)
    n_pos = int(graph.n_pos)
    selected_features = _tensor_to_cpu(graph.selected_features).to(torch.long)
    active_features = _tensor_to_cpu(graph.active_features).to(torch.long)

    return {
        "payload_version": PAYLOAD_VERSION,
        "node_threshold": float(node_threshold),
        "n_layers": n_layers,
        "n_pos": n_pos,
        "n_features_total": int(len(selected_features)),
        "n_error_nodes_total": int(n_layers * n_pos),
        "pruned_node_indices": index,
        "pruned_cumulative_scores": pruned_cumulative_scores,
        "pruned_adjacency_sparse": pruned_adjacency,
        "selected_features": selected_features,
        "active_features": active_features,
        "activation_values": _tensor_to_cpu(graph.activation_values),
        "logit_tokens": _tensor_to_cpu(graph.logit_tokens).to(torch.long),
        "logit_probabilities": _tensor_to_cpu(graph.logit_probabilities),
        "input_tokens": _tensor_to_cpu(graph.input_tokens).to(torch.long),
        "scan": getattr(graph, "scan", None),
    }


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def load_compact_payload(
    path: str | Path,
    check_sparse_invariants: bool = False,
) -> dict[str, Any]:
    checker = getattr(torch.sparse, "check_sparse_tensor_invariants", None)
    check_context = (
        checker(enable=True)
        if check_sparse_invariants and checker is not None
        else nullcontext()
    )
    with check_context:
        try:
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Compact payload must be a dict: {path}")
    version = payload.get("payload_version")
    if version != PAYLOAD_VERSION:
        raise ValueError(f"Unsupported compact payload version {version!r}: {path}")
    return payload


def save_compact_payload(graph: Any, path: str | Path, node_threshold: float = 0.8) -> dict[str, Any]:
    payload = build_compact_payload(graph, node_threshold=node_threshold)
    atomic_torch_save(payload, path)
    done_path = Path(str(path) + ".done")
    done_path.write_text("ok\n", encoding="utf-8")
    return payload


def extract_advanced_graph_features_from_payload(
    payload: dict[str, Any],
    topology_backend: str = DENSE_REFERENCE_BACKEND,
) -> np.ndarray:
    pruned_node_indices = [int(item) for item in payload["pruned_node_indices"].tolist()]
    n_layers = int(payload["n_layers"])
    n_features_total = int(payload["n_features_total"])
    n_error_nodes_total = int(payload["n_error_nodes_total"])

    pruned_feature_nodes = [i for i in pruned_node_indices if i < n_features_total]
    pruned_error_nodes = [
        i for i in pruned_node_indices if n_features_total <= i < n_features_total + n_error_nodes_total
    ]
    pruned_global_to_payload = {global_idx: local_idx for local_idx, global_idx in enumerate(pruned_node_indices)}

    features: list[float] = []
    activation_values = payload["activation_values"].to("cpu")
    selected_features = payload["selected_features"].to("cpu")
    active_features = payload["active_features"].to("cpu")
    logit_probabilities = payload["logit_probabilities"].to("cpu")
    pruned_cumulative_scores = payload["pruned_cumulative_scores"].to("cpu")

    features.append(float(activation_values.shape[0]))
    features.append(float(len(pruned_feature_nodes)))
    features.append(float(len(pruned_error_nodes)))

    if logit_probabilities.numel() > 0:
        features.append(float(logit_probabilities[0].item()))
        features.append(float(-torch.sum(logit_probabilities * torch.log(logit_probabilities + 1e-8)).item()))
    else:
        features.extend([0.0, 0.0])

    features.append(float(pruned_cumulative_scores.mean().item()) if pruned_cumulative_scores.numel() else 0.0)

    if pruned_error_nodes:
        error_local = [pruned_global_to_payload[idx] for idx in pruned_error_nodes]
        error_influences = pruned_cumulative_scores[torch.tensor(error_local, dtype=torch.long)]
        features.append(float(error_influences.sum().item()))
        features.append(float(error_influences.mean().item()))
    else:
        features.extend([0.0, 0.0])

    if pruned_feature_nodes:
        selected_indices = [int(selected_features[i]) for i in pruned_feature_nodes]
        pruned_activations = activation_values[torch.tensor(selected_indices, dtype=torch.long)]
        if pruned_activations.numel() > 0:
            features.append(float(pruned_activations.mean().item()))
            features.append(float(pruned_activations.max().item()))
            features.append(float(pruned_activations.std().item()))
        else:
            features.extend([0.0, 0.0, 0.0])
    else:
        features.extend([0.0, 0.0, 0.0])

    layer_counts = [0.0] * n_layers
    for node_idx in pruned_feature_nodes:
        feature_idx = int(selected_features[node_idx])
        if feature_idx < len(active_features):
            layer, _, _ = active_features[feature_idx].tolist()
            if int(layer) < n_layers:
                layer_counts[int(layer)] += 1.0
    features.extend(layer_counts)

    if topology_backend == DENSE_REFERENCE_BACKEND:
        topo_features = extract_topological_and_edge_features_from_payload(
            payload,
            pruned_node_indices,
        )
    elif topology_backend == SPARSE_EXACT_BACKEND:
        topo_features = extract_topological_and_edge_features_from_payload_sparse_exact(
            payload,
            pruned_node_indices,
        )
    else:
        raise ValueError(
            f"Unsupported topology backend {topology_backend!r}; "
            f"expected one of {TOPOLOGY_BACKENDS}."
        )
    ordered_keys = [
        "sum_edge_weights",
        "mean_edge_weights",
        "std_edge_weights",
        "n_edges_pruned",
        "graph_density",
        "n_connected_components",
        "mean_degree_centrality",
        "max_degree_centrality",
        "mean_betweenness_centrality",
        "max_betweenness_centrality",
        "avg_shortest_path_length",
        "input_to_logit_shortest_path",
    ]
    features.extend(float(topo_features[key]) for key in ordered_keys)
    return np.array(features, dtype=np.float32)


def extract_advanced_graph_features_from_payload_sparse_exact(
    payload: dict[str, Any],
) -> np.ndarray:
    return extract_advanced_graph_features_from_payload(
        payload,
        topology_backend=SPARSE_EXACT_BACKEND,
    )


def validate_compact_payload_sparse_adjacency(payload: dict[str, Any]) -> dict[str, Any]:
    sparse_adj = payload.get("pruned_adjacency_sparse")
    if not isinstance(sparse_adj, torch.Tensor):
        raise TypeError("pruned_adjacency_sparse must be a torch.Tensor")
    if sparse_adj.layout != torch.sparse_coo:
        raise TypeError(
            "pruned_adjacency_sparse must use sparse COO layout, "
            f"got {sparse_adj.layout}"
        )
    if sparse_adj.ndim != 2:
        raise ValueError(f"Sparse adjacency must be 2D, got shape={tuple(sparse_adj.shape)}")

    shape = [int(size) for size in sparse_adj.shape]
    pruned_indices = payload.get("pruned_node_indices")
    if not isinstance(pruned_indices, torch.Tensor):
        raise TypeError("pruned_node_indices must be a torch.Tensor")
    expected_nodes = int(pruned_indices.numel())
    if shape != [expected_nodes, expected_nodes]:
        raise ValueError(
            "Sparse adjacency shape does not match pruned nodes: "
            f"shape={shape} pruned_nodes={expected_nodes}"
        )

    indices = sparse_adj._indices().to("cpu")
    values = sparse_adj._values().to("cpu")
    if indices.ndim != 2 or indices.shape[0] != 2:
        raise ValueError(f"Sparse adjacency indices must have shape [2, nnz], got {indices.shape}")
    if values.ndim != 1:
        raise ValueError(f"Sparse adjacency values must be scalar, got shape={values.shape}")

    indices_in_bounds = True
    if indices.numel() > 0:
        indices_in_bounds = bool(
            (indices[0] >= 0).all()
            and (indices[0] < shape[0]).all()
            and (indices[1] >= 0).all()
            and (indices[1] < shape[1]).all()
        )
    if not indices_in_bounds:
        raise ValueError(
            "Sparse adjacency contains out-of-bounds indices: "
            f"shape={shape} min={int(indices.min().item())} max={int(indices.max().item())}"
        )

    values_finite = bool(torch.isfinite(values).all().item()) if values.numel() else True
    dense_elements = shape[0] * shape[1]
    return {
        "shape": shape,
        "dtype": str(values.dtype),
        "is_coalesced": bool(sparse_adj.is_coalesced()),
        "nnz_stored": int(values.numel()),
        "nnz_nonzero": int(torch.count_nonzero(values).item()),
        "indices_in_bounds": indices_in_bounds,
        "values_finite": values_finite,
        "dense_estimated_bytes": int(dense_elements * values.element_size()),
    }


def _topology_defaults(num_nodes_pruned: int) -> dict[str, float]:
    return {
        "sum_edge_weights": 0.0,
        "mean_edge_weights": 0.0,
        "std_edge_weights": 0.0,
        "n_edges_pruned": 0.0,
        "graph_density": 0.0,
        "n_connected_components": float(num_nodes_pruned),
        "mean_degree_centrality": 0.0,
        "max_degree_centrality": 0.0,
        "mean_betweenness_centrality": 0.0,
        "max_betweenness_centrality": 0.0,
        "avg_shortest_path_length": -1.0,
        "input_to_logit_shortest_path": -1.0,
    }


def extract_topological_and_edge_features_from_payload(
    payload: dict[str, Any],
    pruned_node_indices: list[int],
) -> dict[str, float]:
    num_nodes_pruned = len(pruned_node_indices)
    defaults = _topology_defaults(num_nodes_pruned)
    if num_nodes_pruned < 2:
        return defaults

    sparse_adj = payload["pruned_adjacency_sparse"].to("cpu").coalesce()
    pruned_adj_matrix = sparse_adj.to_dense().numpy()
    G = nx.from_numpy_array(pruned_adj_matrix, create_using=nx.DiGraph)

    return _extract_topological_features_from_networkx(
        payload=payload,
        pruned_node_indices=pruned_node_indices,
        graph=G,
        defaults=defaults,
    )


def extract_topological_and_edge_features_from_payload_sparse_exact(
    payload: dict[str, Any],
    pruned_node_indices: list[int],
) -> dict[str, float]:
    num_nodes_pruned = len(pruned_node_indices)
    defaults = _topology_defaults(num_nodes_pruned)
    if num_nodes_pruned < 2:
        return defaults

    validate_compact_payload_sparse_adjacency(payload)
    sparse_adj = payload["pruned_adjacency_sparse"].to("cpu").coalesce()
    indices = sparse_adj.indices().t().tolist()
    values = sparse_adj.values().tolist()

    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes_pruned))
    graph.add_weighted_edges_from(
        (int(source), int(target), value)
        for (source, target), value in zip(indices, values)
        if value != 0
    )

    return _extract_topological_features_from_networkx(
        payload=payload,
        pruned_node_indices=pruned_node_indices,
        graph=graph,
        defaults=defaults,
    )


def _extract_topological_features_from_networkx(
    payload: dict[str, Any],
    pruned_node_indices: list[int],
    graph: nx.DiGraph,
    defaults: dict[str, float],
) -> dict[str, float]:
    G = graph

    features = dict(defaults)
    edge_weights = np.array(list(nx.get_edge_attributes(G, "weight").values()), dtype=np.float64)
    if edge_weights.size > 0:
        features["sum_edge_weights"] = float(np.sum(edge_weights))
        features["mean_edge_weights"] = float(np.mean(edge_weights))
        features["std_edge_weights"] = float(np.std(edge_weights))

    features["n_edges_pruned"] = float(G.number_of_edges())
    features["graph_density"] = float(nx.density(G))
    features["n_connected_components"] = float(nx.number_weakly_connected_components(G))

    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight="weight")
    if degree_centrality:
        features["mean_degree_centrality"] = float(np.mean(list(degree_centrality.values())))
        features["max_degree_centrality"] = float(np.max(list(degree_centrality.values())))
    if betweenness_centrality:
        features["mean_betweenness_centrality"] = float(np.mean(list(betweenness_centrality.values())))
        features["max_betweenness_centrality"] = float(np.max(list(betweenness_centrality.values())))

    largest_cc = max(nx.weakly_connected_components(G), key=len)
    if len(largest_cc) > 1:
        subgraph = G.subgraph(largest_cc)
        try:
            features["avg_shortest_path_length"] = float(
                nx.average_shortest_path_length(subgraph, weight="weight")
            )
        except (nx.NetworkXError, ValueError):
            features["avg_shortest_path_length"] = -1.0

    n_features_total = int(payload["n_features_total"])
    n_error_nodes_total = int(payload["n_error_nodes_total"])
    start_token_nodes = n_features_total + n_error_nodes_total
    end_token_nodes = start_token_nodes + int(payload["n_pos"])
    start_logit_nodes = end_token_nodes
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(pruned_node_indices)}
    input_nodes = [
        global_to_local[i] for i in pruned_node_indices if start_token_nodes <= i < end_token_nodes
    ]
    logit_nodes = [global_to_local[i] for i in pruned_node_indices if i >= start_logit_nodes]

    min_path_len = float("inf")
    for source in input_nodes:
        for target in logit_nodes:
            if nx.has_path(G, source=source, target=target):
                try:
                    path_len = nx.shortest_path_length(G, source=source, target=target, weight="weight")
                except ValueError:
                    continue
                min_path_len = min(min_path_len, float(path_len))
    if min_path_len != float("inf"):
        features["input_to_logit_shortest_path"] = min_path_len
    return features
