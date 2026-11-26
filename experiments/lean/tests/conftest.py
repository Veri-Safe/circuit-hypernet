from __future__ import annotations

import sys
from pathlib import Path


S5_ROOT = Path(__file__).resolve().parents[1]
SRC = S5_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

S3_SRC = S5_ROOT.parent / "s3_circuit" / "src"
if str(S3_SRC) not in sys.path:
    sys.path.insert(0, str(S3_SRC))

S2_SRC = S5_ROOT.parent / "s2_hypernet" / "src"
if str(S2_SRC) not in sys.path:
    sys.path.insert(0, str(S2_SRC))
