"""Immutable cached-graph edits and task SAE readouts; never generate Base here."""
from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import sys
from collections import defaultdict

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "s5_lean_circuit/src"))
from s5_lean_circuit.circuit.descriptor_v2_397 import fixed_log1p_397, vectorize_payload_v2_397, SCHEMA_397
from s5_lean_circuit.circuit.payload_v2 import read_payload_v2_shard, verify_payload_v2_shard_record

TASKS = ("S_and", "S_or_l", "S_or_r", "S_id")
TARGET = "S_or_r"
SOURCE_SHA = "8e3119fe61fabc85039fa790668a9e562ae0c8cac7288d04655c2df6bb3dae03"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary.write_bytes(gzip.compress(encoded, mtime=0) if path.suffix == ".gz" else encoded)
    temporary.replace(path)


def relocate(path):
    path = Path(path)
    if path.anchor and not path.is_relative_to(ROOT) and "HyperCombine" in path.parts:
        return ROOT.joinpath(*path.parts[path.parts.index("HyperCombine") + 1:])
    return path if path.is_absolute() else ROOT / path


def load_graph(record):
    if "payload" in record:
        return record["payload"]
    if "graph_path" in record:
        payloads = read_payload_v2_shard(relocate(record["graph_path"]))
    else:
        shard = dict(record["graph"]["shard"])
        for key in ("path", "tensor_path", "sidecar_path", "sidecar"):
            if key in shard:
                shard[key] = str(relocate(shard[key]))
        payloads = verify_payload_v2_shard_record(shard, expected_archive_identity_sha256=SOURCE_SHA)
    if len(payloads) != 1:
        raise ValueError("expected exactly one graph payload per question")
    return payloads[0]


def tensor_sha(tensor):
    tensor = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
    h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def graph_sha(payload):
    h = hashlib.sha256()
    for field in payload.__dataclass_fields__:
        value = getattr(payload, field)
        h.update(field.encode())
        h.update((tensor_sha(value) if isinstance(value, torch.Tensor) else str(value)).encode())
    return h.hexdigest()


def original_vector(payload, record=None):
    raw = vectorize_payload_v2_397(payload, n_layers=8, d_feature=16384)
    treated = fixed_log1p_397(raw)
    if record:
        for key, value in (("raw_vector", raw), ("model_input_vector", treated)):
            if key in record and not torch.equal(value, torch.tensor(record[key], dtype=torch.float32)):
                raise ValueError("cached graph does not exactly reproduce " + key)
    return raw, treated


def edit_vector(payload, identities):
    """Delete all (layer, feature) occurrences and incident edges, on a new payload."""
    identities = sorted(set(tuple(x) for x in identities))
    if not identities or any(not 0 <= l < 8 or not 0 <= f < 16384 for l, f in identities):
        raise ValueError("invalid layer-qualified graph intervention")
    payload.validate(n_layers=8, d_feature=16384)
    selected = torch.zeros(payload.feature_count, dtype=torch.bool)
    detail = []
    for layer, fid in identities:
        hits = (payload.feature_layer == layer) & (payload.feature_id == fid)
        selected |= hits
        detail.append({"layer": layer, "feature_id": fid, "removed_nodes": int(hits.sum()),
                       "positions": payload.feature_position[hits].tolist()})
    raw0, input0 = original_vector(payload)
    if bool(selected.any()):
        keep = ~selected
        removed = int(selected.sum())
        mapping = torch.full((payload.node_count,), -1, dtype=torch.int32)
        mapping[:payload.feature_count][keep] = torch.arange(payload.feature_count - removed, dtype=torch.int32)
        mapping[payload.feature_count:] = torch.arange(payload.feature_count - removed,
                                                       payload.node_count - removed, dtype=torch.int32)
        edge_keep = (mapping[payload.edge_source.long()] >= 0) & (mapping[payload.edge_target.long()] >= 0)
        edited = replace(payload, feature_layer=payload.feature_layer[keep],
                         feature_position=payload.feature_position[keep], feature_id=payload.feature_id[keep],
                         feature_activation=payload.feature_activation[keep],
                         feature_influence=payload.feature_influence[keep],
                         edge_source=mapping[payload.edge_source.long()[edge_keep]],
                         edge_target=mapping[payload.edge_target.long()[edge_keep]],
                         edge_signed_direct_effect=payload.edge_signed_direct_effect[edge_keep],
                         edge_nonnegative_influence=payload.edge_nonnegative_influence[edge_keep])
        edited.validate(n_layers=8, d_feature=16384)
        raw = vectorize_payload_v2_397(edited, n_layers=8, d_feature=16384)
        treated = fixed_log1p_397(raw)  # Exactly once; caller passes this to Runtime.capture.
    else:
        removed, edited, raw, treated = 0, payload, raw0, input0
    changed = torch.where((raw != raw0) | (treated != input0))[0].tolist()
    receipt = {"kind": "cached_graph_edit_from_original", "identities": detail,
               "removed_nodes": removed, "removed_edges": payload.edge_count - edited.edge_count,
               "original_graph_sha256": graph_sha(payload), "edited_graph_sha256": graph_sha(edited),
               "original_model_input_sha256": tensor_sha(input0), "edited_model_input_sha256": tensor_sha(treated),
               "preprocess": "fixed_log1p_v1_once", "circuit_present": True,
               "delta_raw_l2": float(torch.linalg.vector_norm(raw - raw0)),
               "delta_model_input_l2": float(torch.linalg.vector_norm(treated - input0)),
               "changed_coordinates": [{"index": i, "name": SCHEMA_397.coordinate_names[i],
                                         "raw_before": float(raw0[i]), "raw_after": float(raw[i]),
                                         "input_before": float(input0[i]), "input_after": float(treated[i])}
                                        for i in changed]}
    return treated, receipt


def candidate_pool(questions, records, dictionary_hashes, size=256):
    """Only graph influence/coverage is used; no HyperNet score can change this pool."""
    if size not in (256, 512):
        raise ValueError("first pool must have 256 or 512 nodes")
    table = {}
    for q in questions:
        p = load_graph(records[q["question_id"]])
        seen = set()
        for layer, fid, influence in zip(p.feature_layer.tolist(), p.feature_id.tolist(), p.feature_influence.tolist()):
            key = (layer, fid)
            if key not in table:
                table[key] = {"layer": layer, "feature_id": fid,
                              "dictionary_sha256": dictionary_hashes[str(layer)],
                              "task_coverage": dict.fromkeys(TASKS, 0),
                              "task_abs_influence": dict.fromkeys(TASKS, 0.0)}
            row = table[key]
            row["task_abs_influence"][q["task_type"]] += abs(float(influence))
            if key not in seen:
                row["task_coverage"][q["task_type"]] += 1
                seen.add(key)
    quotas = size // 8
    picked = []
    for layer in range(8):
        entries = [x for x in table.values() if x["layer"] == layer]
        entries.sort(key=lambda x: (-x["task_abs_influence"][TARGET],
                                    -sum(x["task_coverage"].values()), x["feature_id"]))
        # Half strongest target influence; half best cross-question coverage.
        chosen = entries[:quotas // 2]
        selected = {x["feature_id"] for x in chosen}
        remaining = sorted((x for x in entries if x["feature_id"] not in selected),
                           key=lambda x: (-sum(x["task_coverage"].values()),
                                          -x["task_abs_influence"][TARGET], x["feature_id"]))
        chosen += remaining[:quotas - len(chosen)]
        if not chosen:
            raise ValueError(f"graph identity union has no layer {layer}")
        picked += chosen
    if len(picked) < size:
        chosen_ids = {(x["layer"], x["feature_id"]) for x in picked}
        remaining = sorted((x for k, x in table.items() if k not in chosen_ids),
                           key=lambda x: (-x["task_abs_influence"][TARGET],
                                          -sum(x["task_coverage"].values()), x["layer"], x["feature_id"]))
        picked += remaining[:size - len(picked)]
    return {"format": "S_or_r_graph_only_candidate_pool_v1", "requested_size": size,
            "actual_size": len(picked), "union_size": len(table), "question_count": len(questions),
            "selection_rule": "8 layer quotas: half target_abs_influence, half all-task question coverage; stable identity ties",
            "source_graph_identity_sha256": SOURCE_SHA, "candidates": sorted(picked, key=lambda x: (x["layer"], x["feature_id"])),
            "union_identity_sha256": digest(sorted(table))}


class GraphReadout:
    def __init__(self, runtime, group, questions, records, output, sigma_min=1e-6):
        self.runtime, self.group, self.questions, self.records = runtime, group, questions, records
        self.payloads = {q["question_id"]: load_graph(records[q["question_id"]]) for q in questions}
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.coordinates = runtime.coordinates(group)
        sites = defaultdict(lambda: defaultdict(list))
        for i, coordinate in enumerate(self.coordinates):
            sites[coordinate["site"]][coordinate["slot"]].append(i)
        self.weights = torch.zeros(len(self.coordinates), dtype=torch.float64)
        for slots in sites.values():
            for indices in slots.values():
                self.weights[indices] = 1.0 / (len(sites) * len(slots) * len(indices))
        manifest = {"group": group, "question_ids": [q["question_id"] for q in questions],
                    "sigma_min": sigma_min, "coordinates": self.coordinates,
                    "hierarchy": "equal sites, equal slots within site, equal features within slot"}
        self.identity = digest(manifest)
        capture_path = self.output / "baseline_activations.pt"
        if capture_path.exists():
            saved = torch.load(capture_path, map_location="cpu", weights_only=False)
            if saved["identity"] != self.identity:
                raise ValueError("readout baseline cache identity mismatch")
            z = saved["z"]
        else:
            zs = []
            for q in questions:
                p = self.payloads[q["question_id"]]
                _, vector = original_vector(p, records[q["question_id"]])
                zs.append(runtime.capture(q["input_ids"], vector.tolist(), group).reshape(-1).cpu())
            z = torch.stack(zs)
            torch.save({"identity": self.identity, "z": z}, capture_path)
        if z.shape != (len(questions), len(self.coordinates)) or not torch.isfinite(z).all():
            raise ValueError("qualified H group capture shape/finiteness mismatch")
        self.z = z.double()
        self.sigma = self.z.std(0, unbiased=False)
        self.active = self.sigma >= sigma_min
        self.weights[~self.active] = 0
        # Re-normalize remaining coordinates within the original hierarchy, preserving site/slot balance.
        live_sites = {s: {slot: [i for i in inds if self.active[i]] for slot, inds in slots.items()}
                      for s, slots in sites.items()}
        live_sites = {s: {slot: inds for slot, inds in slots.items() if inds} for s, slots in live_sites.items()}
        live_sites = {s: slots for s, slots in live_sites.items() if slots}
        for slots in live_sites.values():
            for indices in slots.values():
                self.weights[indices] = 1.0 / (len(live_sites) * len(slots) * len(indices))
        self.sigma[~self.active] = 1.0  # excluded weights remain zero, never divide by a small sigma
        if not self.active.any():
            raise ValueError("all H coordinates have near-zero development variance")
        means = {task: self.z[[q["task_type"] == task for q in questions]].mean(0) for task in TASKS}
        v = (means[TARGET] - torch.stack([means[t] for t in TASKS if t != TARGET]).mean(0)) / self.sigma
        v[~self.active] = 0
        norm = torch.sqrt((self.weights * v.square()).sum())
        self.projection = v / norm if norm > 0 else None
        normalization = {**manifest, "identity": self.identity, "active_count": int(self.active.sum()),
                         "sigma": self.sigma.tolist(), "weights": self.weights.tolist(),
                         "excluded_coordinate_indices": (~self.active).nonzero().reshape(-1).tolist(),
                         "projection": self.projection.tolist() if self.projection is not None else None,
                         "baseline_activation_sha256": tensor_sha(z)}
        write_json(self.output / "readout_normalization.json.gz", normalization)

    def measure(self, identity):
        layer, fid = identity
        path = self.output / "features" / f"L{layer}_F{fid:05}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["readout_identity"] != self.identity:
                raise ValueError("candidate H response cache identity mismatch")
            return result
        rows, best_examples = [], []
        forward_count = 0
        for index, q in enumerate(self.questions):
            payload = self.payloads[q["question_id"]]
            edited, receipt = edit_vector(payload, [identity])
            if receipt["removed_nodes"]:
                zedit = self.runtime.capture(q["input_ids"], edited.tolist(), self.group).reshape(-1).double().cpu()
                forward_count += 1
            else:
                zedit = self.z[index]
            delta = (self.z[index] - zedit) / self.sigma
            unsigned = float((self.weights * delta.abs()).sum())
            projected = float((self.weights * self.projection * delta).sum()) if self.projection is not None else None
            row = {"question_id": q["question_id"], "task_type": q["task_type"],
                   "unsigned": unsigned, "task_projection": projected,
                   "removed_nodes": receipt["removed_nodes"], "removed_edges": receipt["removed_edges"],
                   "delta_raw_l2": receipt["delta_raw_l2"], "delta_model_input_l2": receipt["delta_model_input_l2"]}
            rows.append(row)
            if receipt["removed_nodes"]:
                top = (self.weights * delta.abs()).topk(min(12, len(delta))).indices.tolist()
                example = {**row, **receipt,
                           "hn_coordinates": [{**self.coordinates[i], "z0": float(self.z[index, i]),
                                               "zedit": float(zedit[i]), "sigma": float(self.sigma[i]),
                                               "standardized_z0_minus_zedit": float(delta[i]),
                                               "coordinate_weight": float(self.weights[i])} for i in top]}
                best_examples.append(example)
                best_examples.sort(key=lambda x: (x["task_type"] == TARGET, x["unsigned"]), reverse=True)
                best_examples = best_examples[:3]
        variants = {}
        for family in ("unsigned", "task_projection"):
            if family == "task_projection" and self.projection is None:
                continue
            target_mean = sum(x[family] for x in rows if x["task_type"] == TARGET) / sum(x["task_type"] == TARGET for x in rows)
            controls = {t: sum(abs(x[family]) for x in rows if x["task_type"] == t) / sum(x["task_type"] == t for x in rows)
                        for t in TASKS if t != TARGET}
            for penalty in (0, 0.5, 1, 2):
                variants[f"{family}_lambda{penalty:g}"] = {"score": target_mean - penalty * max(controls.values()),
                                                        "target_mean": target_mean, "controls_mean_absolute": controls}
        result = {"layer": layer, "feature_id": fid, "readout_identity": self.identity,
                  "actual_graph_edit": True, "hn_forward_count": forward_count,
                  "question_count": len(rows), "variants": variants, "per_question": rows, "examples": best_examples}
        write_json(path, result)
        return result
