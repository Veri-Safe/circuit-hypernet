from __future__ import annotations

import os
from typing import Any

import torch


DETERMINISTIC_RUNTIME_FORMAT = "nvarc_cuda_deterministic_runtime_v1"
DETERMINISTIC_RUNTIME_MODE = "strict_cuda_deterministic_v1"
CURRENT_RUNTIME_MODE = "current_runtime_v1"
REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_runtime(mode: str) -> dict[str, Any]:
    if mode not in {CURRENT_RUNTIME_MODE, DETERMINISTIC_RUNTIME_MODE}:
        raise ValueError(f"Unsupported NVARC runtime mode: {mode}")
    if mode == DETERMINISTIC_RUNTIME_MODE:
        if (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            != REQUIRED_CUBLAS_WORKSPACE_CONFIG
        ):
            raise ValueError(
                "Deterministic runtime requires "
                f"CUBLAS_WORKSPACE_CONFIG={REQUIRED_CUBLAS_WORKSPACE_CONFIG}"
            )
        torch.use_deterministic_algorithms(True)
        torch.set_deterministic_debug_mode("error")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return runtime_settings(mode)


def runtime_settings(mode: str) -> dict[str, Any]:
    return {
        "format": DETERMINISTIC_RUNTIME_FORMAT,
        "mode": mode,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def deterministic_runtime_exact(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    expected = {
        "format": DETERMINISTIC_RUNTIME_FORMAT,
        "mode": DETERMINISTIC_RUNTIME_MODE,
        "deterministic_algorithms": True,
        "deterministic_debug_mode": 2,
        "cublas_workspace_config": REQUIRED_CUBLAS_WORKSPACE_CONFIG,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    return all(value.get(key) == expected_value for key, expected_value in expected.items())
