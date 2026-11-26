from contextlib import nullcontext
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from s3circuit.nvarc_circuit import attribution, vector59
from s3circuit.nvarc_circuit.payload import build_payload_from_streaming_attribution


class LinearCircuit:
    """A CPU model and attribution context with eight features in a directed chain."""

    def __init__(self):
        self.cfg = SimpleNamespace(device="cpu")
        self.unembed = SimpleNamespace(W_U=torch.tensor([[0.0, 1.0, 2.0]]))
        self.ln_final = torch.nn.Identity()
        self.forward_widths = []
        self.row_widths = []
        self._batch_buffer = None

    def setup_attribution(self, token_ids):
        count = 8
        positions = token_ids.numel()
        activations = torch.zeros(36, positions, 1)
        activations[:count, 1, 0] = 1
        self.activation_matrix = activations.to_sparse()
        self.encoder_vecs = torch.ones(count, 1)
        self.logits = torch.zeros(1, positions, 3)
        self._resid_activations = [None]
        self._row_size = count + 36 * positions + positions
        return self

    def install_hooks(self, model):
        return nullcontext()

    def forward(self, tokens, stop_at_layer):
        width, positions = tokens.shape
        self.forward_widths.append(width)
        self.sources = torch.ones(width, self._row_size, requires_grad=True)
        self.sources.register_hook(self._collect_root_gradient)

        # Feature i has a token parent and (except for i=0) a preceding feature.
        feature_count = self.encoder_vecs.shape[0]
        token_source = self.sources[:, -1:]
        self.targets = torch.cat((token_source, token_source + self.sources[:, :feature_count - 1]), dim=1)
        coefficients = torch.arange(1, feature_count + 1, dtype=torch.float32)
        root = (self.sources[:, :feature_count] * coefficients).sum(dim=1)
        return root[:, None, None].expand(width, positions, 1)

    def _collect_root_gradient(self, gradient):
        if self._batch_buffer is not None:
            self._batch_buffer.copy_(gradient.T)

    def compute_batch(self, *, layers, positions, inject_values, retain_graph):
        width = len(layers)
        self.row_widths.append(width)
        targets = self.targets[torch.arange(width), layers]
        gradients, = torch.autograd.grad(
            targets, self.sources, grad_outputs=inject_values[:, 0], retain_graph=retain_graph
        )
        return gradients[:width]


@pytest.fixture(scope="module")
def tiny_cpu_graphs():
    # Thread-pool overhead dominates the tiny reductions in this fixture.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("extract", [
    attribution.attribute_joint_span_streaming_exact,
    attribution.attribute_joint_span_cached_exact_v2,
])
def test_physical_microbatch_preserves_exact_graph_and_descriptor(monkeypatch, extract, tiny_cpu_graphs):
    # The fixture supplies CodeCircuit's model interface; no external checkout is needed.
    monkeypatch.setattr(attribution, "ensure_codecircuit_on_path", lambda: None)
    # Use the built-in NetworkX backend for this tiny graph to skip JIT compilation.
    monkeypatch.setattr(vector59, "numba_available", lambda: False)
    outputs = []
    for expected_width, overrides in ((4, {"compute_microbatch_cap": 4}), (1, {})):
        model = LinearCircuit()
        result = extract(
            model,
            condition_token_ids=[1, 2],
            target_token_ids=[0],
            row_batch_size=4,
            max_feature_nodes=7,
            update_interval=1,
            node_threshold=0.99,
            edge_threshold=0.99,
            **overrides,
        )
        assert model.forward_widths == [expected_width]
        assert max(model.row_widths) == expected_width
        assert result.batching.logical_queue_size == 4
        assert len(result.graph.selected_features) == 7
        payload = build_payload_from_streaming_attribution(
            event_key="1" * 64, artifact_id="2" * 64, condition_length=2, attribution=result
        )
        assert payload.feature_count > 1 and payload.edge_source.numel() > 1
        outputs.append((result.graph.selected_features, payload, vector59.vectorize_payload(payload).values))

    (wide_selection, wide_graph, wide_vector), (narrow_selection, narrow_graph, narrow_vector) = outputs
    assert torch.equal(wide_selection, narrow_selection)
    for field in fields(wide_graph):
        wide, narrow = getattr(wide_graph, field.name), getattr(narrow_graph, field.name)
        if isinstance(wide, torch.Tensor):
            torch.testing.assert_close(wide, narrow, rtol=0, atol=0)
        else:
            assert wide == narrow
    np.testing.assert_array_equal(wide_vector, narrow_vector)
