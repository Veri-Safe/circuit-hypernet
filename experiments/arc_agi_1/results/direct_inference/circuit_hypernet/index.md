# ARC S0 step300 archive

Existing matched evaluation; no retraining or generation. All metrics independently recompute from 419 grouped inputs and the archived gold data: selected exact@2 = 69/400 = 17.25%. The numerator is a sum of task fractions, not a count of correct inputs.

Concatenate `predictions.part*.jsonl` and `query_predictions.part*.jsonl` in numeric order to restore the original byte streams. Chunk hashes, original hashes and paths are in `archive_manifest.json`; checkpoint bytes remain on shared storage.
