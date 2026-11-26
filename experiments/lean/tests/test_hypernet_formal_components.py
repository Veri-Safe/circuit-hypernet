from __future__ import annotations

import torch

from s2_hypernet.lora_tokenizer import Tokenizer2DBatchedLoRA
from s2_hypernet.nvarc_pg.generated_lora import batched_lora_delta
from s5_lean_circuit.hypernet.circuit_encoders import (
    CandidateAEncoder,
    CandidateBEncoder,
    append_after_valid_query,
)
from s5_lean_circuit.hypernet.query_encoder import (
    ConditionTypeEmbeddings,
    QueryEncoder,
)
from s5_lean_circuit.hypernet.target_layout import build_s5_target_layout


def test_real_query_encoder_masks_padding_without_changing_valid_tokens() -> None:
    encoder = QueryEncoder().eval()
    ids = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 7, 8, 9]])
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    changed = ids.clone()
    changed[0, 3:] = torch.tensor([123, 456])
    with torch.no_grad():
        left = encoder(ids, mask)
        right = encoder(changed, mask)
    assert torch.equal(left[0, :3], right[0, :3])
    assert torch.equal(left[0, 3:], torch.zeros_like(left[0, 3:]))


def test_real_candidate_encoders_share_types_and_preserve_block_addresses() -> None:
    types = ConditionTypeEmbeddings()
    encoder_a = CandidateAEncoder(type_embeddings=types).eval()
    encoder_b = CandidateBEncoder(type_embeddings=types).eval()
    assert encoder_a.type_embeddings is encoder_b.type_embeddings is types
    vector = torch.arange(2 * 397, dtype=torch.float32).reshape(2, 397) / 397.0
    permutation = torch.tensor([1, 2, 3, 4, 0])
    with torch.no_grad():
        token_a = encoder_a(vector)
        matched = encoder_b(vector)
        permuted = encoder_b(vector, block_permutation=permutation)
    assert token_a.shape == (2, 1, 1024)
    assert matched.shape == permuted.shape == (2, 5, 1024)
    addresses = types.block.unsqueeze(0) + types.circuit.view(1, 1, -1)
    matched_content = matched - addresses
    permuted_content = permuted - addresses
    assert torch.allclose(
        permuted_content,
        matched_content.index_select(1, permutation),
    )


def test_append_after_valid_query_removes_absent_memory_instead_of_zeroing_it() -> None:
    query = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    query_mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]]
    )
    circuit = torch.full((2, 2, 3), 99.0)
    circuit_mask = torch.tensor([[False, False], [True, True]])
    memory, mask = append_after_valid_query(
        query, query_mask, circuit, circuit_mask
    )
    assert mask.tolist() == [
        [True, True, False, False, False],
        [True, True, True, True, True],
    ]
    assert torch.equal(memory[0, :2], query[0, :2])
    assert not bool((memory[0] == 99.0).all(dim=-1).any())
    assert torch.equal(memory[1, 3:], circuit[1])


def test_s5_lora_tokenize_detokenize_roundtrip_is_exact() -> None:
    layout = build_s5_target_layout()
    tokenizer = Tokenizer2DBatchedLoRA(
        token_dim=layout.token_dim,
        rank=layout.rank,
        alpha=layout.alpha,
        pg_mapping=layout.pg_mapping,
    )
    state = {}
    generator = torch.Generator().manual_seed(42)
    for key, shape in tokenizer.shape_state_dict.items():
        state[key] = torch.randn((2, *shape), generator=generator)
    tokens = tokenizer.tokenize(state)
    restored = tokenizer.detokenize(tokenizer.shape_state_dict, tokens)
    assert restored.keys() == state.keys()
    assert all(torch.equal(restored[key], state[key]) for key in state)


def test_batched_generated_lora_has_no_cross_example_crosstalk() -> None:
    hidden = torch.randn(2, 3, 5)
    a = torch.randn(2, 2, 5)
    b = torch.randn(2, 7, 2)
    baseline = batched_lora_delta(hidden, a, b)
    changed_a = a.clone()
    changed_b = b.clone()
    changed_a[1].mul_(100.0)
    changed_b[1].mul_(-100.0)
    changed = batched_lora_delta(hidden, changed_a, changed_b)
    assert torch.equal(baseline[0], changed[0])
    assert not torch.equal(baseline[1], changed[1])
