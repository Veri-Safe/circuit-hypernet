from __future__ import annotations

import os
import sys
from pathlib import Path


HC_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HC_ROOT", str(HC_ROOT))
for source_root in (
    HC_ROOT / "s4_circuit" / "src",
    HC_ROOT / "s3_circuit" / "src",
    HC_ROOT / "s2_hypernet" / "src",
    HC_ROOT / "s1_lora_sft" / "src",
):
    sys.path.insert(0, str(source_root))
