"""Load the repo-local S3 graph algorithms without duplicating their code."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
S3_SRC = REPO_ROOT / "s3_circuit" / "src"
if str(S3_SRC) not in sys.path:
    sys.path.insert(0, str(S3_SRC))

from s3circuit.nvarc_circuit.streaming import (  # noqa: E402
    ROOT_TARGET,
    CachedFeatureSelection,
    StreamingPrunedGraph,
    materialize_cached_exact_graph,
    materialize_streaming_exact_graph,
    selected_feature_influence_cached,
    select_features_cached_exact,
    select_features_streaming_exact,
)
from s3circuit.nvarc_circuit.influence import propagate_influence  # noqa: E402

__all__ = [
    "ROOT_TARGET",
    "CachedFeatureSelection",
    "StreamingPrunedGraph",
    "materialize_cached_exact_graph",
    "materialize_streaming_exact_graph",
    "selected_feature_influence_cached",
    "select_features_cached_exact",
    "select_features_streaming_exact",
    "propagate_influence",
]
