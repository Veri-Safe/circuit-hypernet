"""Vendored model runtime from circuit-extraction-stability.

The source files in this package are adapted from commit
4624427df151befca87eea252314ea1f6d4c4ba7 of
https://github.com/Stepuuu/circuit-extraction-stability and retain the
upstream Apache-2.0 provenance notices.
"""

from .gpt import GPT, GPTConfig

__all__ = ["GPT", "GPTConfig"]
