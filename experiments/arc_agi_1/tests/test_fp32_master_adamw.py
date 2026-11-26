from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from s4_circuit.fp32_master_adamw import FP32MasterAdamW


def _optimizer(parameter, cls=torch.optim.AdamW):
    return cls([parameter], lr=5e-6, betas=(0.9, 0.98), weight_decay=0, fused=True)


def _step(parameter, optimizer):
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_small_updates_accumulate_in_master_instead_of_rounding_away():
    old = nn.Parameter(torch.tensor([0.02, 0.0, 1.0], dtype=torch.bfloat16))
    model = nn.Parameter(old.detach().clone())
    reference = nn.Parameter(old.detach().float().clone())
    initial = old.detach().clone()
    old_opt = _optimizer(old)
    master_opt = _optimizer(model, FP32MasterAdamW)
    ref_opt = _optimizer(reference)
    for _ in range(130):
        _step(old, old_opt)
        _step(model, master_opt)
        _step(reference, ref_opt)
    assert torch.equal(old[[0, 2]], initial[[0, 2]])
    assert not torch.equal(model[0], initial[0])
    assert torch.equal(model, reference.to(torch.bfloat16))
    master = master_opt.param_groups[0]["params"][0]
    assert torch.equal(master, reference)
    assert model.dtype == torch.bfloat16
    assert master_opt.state[master]["exp_avg"].dtype == torch.float32
    assert master_opt.state[master]["exp_avg_sq"].dtype == torch.float32


def test_resume_preserves_updates_smaller_than_bf16_resolution():
    model = nn.Parameter(torch.tensor([0.02], dtype=torch.bfloat16))
    opt = _optimizer(model, FP32MasterAdamW)
    _step(model, opt)
    saved = copy.deepcopy(opt.state_dict())
    restored = nn.Parameter(model.detach().clone())
    resumed_opt = _optimizer(restored, FP32MasterAdamW)
    resumed_opt.load_state_dict(saved)
    for _ in range(129):
        _step(model, opt)
        _step(restored, resumed_opt)
    assert torch.equal(model, restored)
    assert torch.equal(opt.param_groups[0]["params"][0], resumed_opt.param_groups[0]["params"][0])


def test_groups_scheduler_clipping_and_missing_grad_match_fp32_reference():
    torch.manual_seed(42)
    model = [nn.Parameter(torch.randn(4, dtype=dtype))
             for dtype in (torch.bfloat16, torch.float32)]
    ref = [nn.Parameter(p.detach().float().clone()) for p in model]
    def groups(params):
        return [{"params": [params[0]], "lr": 5e-6, "name": "trunk"},
                {"params": [params[1]], "lr": 2e-5, "name": "circuit"}]
    opt = FP32MasterAdamW(groups(model), weight_decay=0.01, fused=True)
    expected = torch.optim.AdamW(groups(ref), weight_decay=0.01, fused=True)
    schedules = [torch.optim.lr_scheduler.LambdaLR(o, lambda s: 1 / (s + 1))
                 for o in (opt, expected)]
    for step in range(5):
        for index, p in enumerate(model):
            p.grad = None if step == 2 and index == 0 else torch.randn_like(p)
        torch.nn.utils.clip_grad_norm_(model, max_norm=0.5)
        for p, r in zip(model, ref):
            r.grad = None if p.grad is None else p.grad.float().clone()
        opt.step()
        expected.step()
        for scheduler in schedules:
            scheduler.step()
        for index, (p, r) in enumerate(zip(model, ref)):
            assert torch.equal(opt.param_groups[index]["params"][0], r)
            assert torch.equal(p, r.to(p.dtype))
        opt.zero_grad(set_to_none=False)
        assert all(p.grad is None or not torch.count_nonzero(p.grad) for p in model)
    assert opt.param_groups[1]["params"][0] is model[1]


def test_rejects_old_optimizer_checkpoints_without_master_state():
    model = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    old = _optimizer(model)
    opt = _optimizer(model, FP32MasterAdamW)
    with pytest.raises(ValueError, match="master"):
        opt.load_state_dict(old.state_dict())


def test_rejects_incompatible_or_nonfinite_master_state_before_mutation():
    model = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    opt = _optimizer(model, FP32MasterAdamW)
    saved = copy.deepcopy(opt.state_dict())
    saved["fp32_master"]["weights"][0][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        opt.load_state_dict(saved)
    assert torch.equal(model, torch.ones_like(model))


def test_rejects_duplicate_parameters_and_dynamic_groups():
    model = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="duplicate"):
        FP32MasterAdamW([model, model])
    opt = _optimizer(model, FP32MasterAdamW)
    with pytest.raises(ValueError, match="group"):
        opt.add_param_group({"params": [nn.Parameter(torch.ones(1))]})


def test_closure_and_zero_grad_apply_to_forward_parameters():
    model = nn.Parameter(torch.tensor([0.02], dtype=torch.bfloat16))
    opt = _optimizer(model, FP32MasterAdamW)
    def closure():
        opt.zero_grad(set_to_none=True)
        loss = model.float().square().sum()
        loss.backward()
        return loss
    loss = opt.step(closure)
    assert loss is not None and loss.item() > 0
    opt.zero_grad(set_to_none=True)
    assert model.grad is None
    assert opt.param_groups[0]["params"][0].grad is None
