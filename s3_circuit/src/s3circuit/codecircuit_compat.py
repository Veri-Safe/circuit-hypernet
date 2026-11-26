from __future__ import annotations

import os


def patch_huggingface_hub_for_codecircuit() -> None:
    """Provide old Hugging Face Hub constants expected by vendored CodeCircuit."""

    import huggingface_hub.constants as hf_constants

    if hasattr(hf_constants, "HF_HUB_ENABLE_HF_TRANSFER"):
        return

    true_values = {"1", "ON", "YES", "TRUE"}
    raw_value = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
    hf_constants.HF_HUB_ENABLE_HF_TRANSFER = (
        raw_value is not None and raw_value.upper() in true_values
    )
