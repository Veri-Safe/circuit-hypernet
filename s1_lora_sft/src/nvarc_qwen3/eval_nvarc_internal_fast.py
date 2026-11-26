#!/usr/bin/env python3
"""High-throughput, protocol-equivalent NVARC ARC evaluator.

This implementation keeps the frozen prompt/view/scoring protocol unchanged.
It vectorizes the 16-token grid grammar and optionally batches views from
different test cases under a conservative live-token budget.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import eval_nvarc_internal as reference
from arc_eval_protocol import (
    ViewPrediction,
    build_author_view_specs,
    canonical_view_spec,
    invert_grid,
    rank_view_predictions,
    sha256_file,
    summarize_view_counts,
    transform_test_view,
    truncate_prompt_to_tokens,
)
from nvarc_grid_grammar import decode_grid_tokens


@dataclass(frozen=True)
class PromptUnit:
    global_test_index: int
    position: int
    token_ids: tuple[int, ...]
    prompt_tokens: int
    removed_train_examples: int
    view_descriptor: str


@dataclass(frozen=True)
class TestContext:
    global_test_index: int
    task_id: str
    test_index: int
    specs: tuple[Any, ...]
    units: tuple[PromptUnit, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--challenges", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--batch-mode", choices=("per_test", "token_budget"), required=True
    )
    parser.add_argument("--max-batch-tokens", type=int, default=0)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--expected-world-size", type=int)
    return parser.parse_args()


class TensorGridGrammar:
    """GPU-resident equivalent of a batch of GridGrammarState objects."""

    def __init__(self, batch_size: int, token_contract: dict[str, Any], device: Any, torch_module: Any) -> None:
        self.torch = torch_module
        self.device = device
        self.batch_size = batch_size
        self.digit_ids = torch_module.tensor(
            token_contract["digit_token_ids"], device=device, dtype=torch_module.long
        )
        self.newline_id = int(token_contract["newline_token_id"])
        self.eos_id = int(token_contract["eos_token_id"])
        self.pad_id = int(token_contract["pad_token_id"])
        # Candidate order deliberately matches scalar GridGrammarState:
        # digits, then eos, then newline. It preserves tie-breaking semantics.
        self.candidate_ids = torch_module.cat(
            (
                self.digit_ids,
                torch_module.tensor(
                    [self.eos_id, self.newline_id],
                    device=device,
                    dtype=torch_module.long,
                ),
            )
        )
        self.completed_rows = torch_module.zeros(batch_size, device=device, dtype=torch_module.long)
        self.current_width = torch_module.zeros(batch_size, device=device, dtype=torch_module.long)
        self.target_width = torch_module.full((batch_size,), -1, device=device, dtype=torch_module.long)
        self.finished = torch_module.zeros(batch_size, device=device, dtype=torch_module.bool)
        lookup = torch_module.zeros(int(token_contract["vocab_size"]), device=device, dtype=torch_module.bool)
        lookup[self.digit_ids] = True
        self.is_digit_lookup = lookup

    def select_and_consume(self, log_probs: Any) -> tuple[Any, Any, Any]:
        torch_module = self.torch
        active = ~self.finished
        row_limit = torch_module.where(self.target_width >= 0, self.target_width, 30)
        digit_allowed = active & (self.current_width < row_limit)
        row_complete = active & (self.current_width > 0) & (
            (self.target_width < 0) | (self.current_width == self.target_width)
        )
        newline_allowed = row_complete & (self.completed_rows + 1 < 30)
        allowed = torch_module.cat(
            (
                digit_allowed[:, None].expand(-1, len(self.digit_ids)),
                row_complete[:, None],
                newline_allowed[:, None],
            ),
            dim=1,
        )
        candidate_scores = log_probs.index_select(1, self.candidate_ids)
        candidate_scores = candidate_scores.masked_fill(~allowed, float("-inf"))
        offsets = candidate_scores.argmax(dim=1)
        selected = self.candidate_ids.index_select(0, offsets)
        selected_log_probs = log_probs.gather(1, selected[:, None]).squeeze(1)

        selected = torch_module.where(active, selected, self.pad_id)
        digit = active & self.is_digit_lookup.index_select(0, selected)
        newline = active & (selected == self.newline_id)
        eos = active & (selected == self.eos_id)
        first_row_end = (newline | eos) & (self.target_width < 0)
        self.target_width = torch_module.where(first_row_end, self.current_width, self.target_width)
        self.current_width = torch_module.where(
            newline | eos,
            torch_module.zeros_like(self.current_width),
            self.current_width + digit.to(self.current_width.dtype),
        )
        self.completed_rows = self.completed_rows + (newline | eos).to(self.completed_rows.dtype)
        self.finished = self.finished | eos
        return selected, selected_log_probs, active


def plan_token_batches(
    units: Sequence[PromptUnit],
    *,
    max_batch_tokens: int,
    max_batch_size: int,
    max_new_tokens: int,
) -> list[list[PromptUnit]]:
    if max_batch_tokens <= 0 or max_batch_size <= 0:
        raise ValueError("Token-budget batching requires positive limits")
    ordered = sorted(
        units,
        key=lambda unit: (
            len(unit.token_ids),
            unit.global_test_index,
            unit.position,
        ),
    )
    batches: list[list[PromptUnit]] = []
    current: list[PromptUnit] = []
    for unit in ordered:
        candidate_size = len(current) + 1
        candidate_tokens = candidate_size * (len(unit.token_ids) + max_new_tokens)
        if current and (
            candidate_size > max_batch_size or candidate_tokens > max_batch_tokens
        ):
            batches.append(current)
            current = []
        current.append(unit)
    if current:
        batches.append(current)
    flattened = [unit for batch in batches for unit in batch]
    if len(flattened) != len(units) or set(flattened) != set(units):
        raise AssertionError("Batch planner lost or duplicated prompt units")
    return batches


def constrained_greedy_vectorized(
    model: Any,
    encoded_prompts: Sequence[Sequence[int]],
    token_contract: dict[str, Any],
    max_new_tokens: int,
    device: Any,
    torch_module: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not encoded_prompts:
        raise ValueError("Cannot decode an empty prompt batch")
    batch_size = len(encoded_prompts)
    pad_id = int(token_contract["pad_token_id"])
    max_prompt = max(map(len, encoded_prompts))
    input_ids = torch_module.full(
        (batch_size, max_prompt), pad_id, device=device, dtype=torch_module.long
    )
    attention_mask = torch_module.zeros(
        (batch_size, max_prompt + max_new_tokens),
        device=device,
        dtype=torch_module.long,
    )
    for index, values in enumerate(encoded_prompts):
        width = len(values)
        input_ids[index, max_prompt - width :] = torch_module.tensor(
            values, device=device, dtype=torch_module.long
        )
        attention_mask[index, max_prompt - width : max_prompt] = 1
    grammar = TensorGridGrammar(batch_size, token_contract, device, torch_module)
    generated = torch_module.full(
        (batch_size, max_new_tokens), pad_id, device=device, dtype=torch_module.long
    )
    selected_logprobs = torch_module.zeros(
        (batch_size, max_new_tokens),
        device=device,
        dtype=torch_module.float32,
    )
    token_count = torch_module.zeros(batch_size, device=device, dtype=torch_module.long)
    sync_count = 0
    started = time.perf_counter()
    with torch_module.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask[:, :max_prompt],
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        steps = 0
        for step in range(max_new_tokens):
            log_probs = torch_module.nn.functional.log_softmax(logits.float(), dim=-1)
            selected, selected_log_probs, active_before = grammar.select_and_consume(log_probs)
            generated[:, step] = selected
            selected_logprobs[:, step] = torch_module.where(
                active_before, selected_log_probs, torch_module.zeros_like(selected_log_probs)
            )
            token_count += active_before.to(token_count.dtype)
            steps = step + 1
            sync_count += 1
            if bool(grammar.finished.all().item()):
                break
            attention_mask[:, max_prompt + step] = active_before.to(attention_mask.dtype)
            outputs = model(
                input_ids=selected[:, None],
                attention_mask=attention_mask[:, : max_prompt + step + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
    if getattr(device, "type", None) == "cuda":
        torch_module.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    generated_cpu = generated[:, :steps].cpu().tolist()
    count_cpu = token_count.cpu().tolist()
    logprob_cpu = selected_logprobs[:, :steps].cpu().tolist()
    finished_cpu = grammar.finished.cpu().tolist()
    digit_by_id = {
        token_id: value
        for value, token_id in enumerate(token_contract["digit_token_ids"])
    }
    results = []
    for index in range(batch_size):
        count = int(count_cpu[index])
        tokens = generated_cpu[index][:count]
        results.append(
            {
                "token_ids": tokens,
                "grid": decode_grid_tokens(
                    tokens,
                    digit_by_id,
                    int(token_contract["newline_token_id"]),
                    int(token_contract["eos_token_id"]),
                ),
                "normalized_logprob": (
                    sum(float(value) for value in logprob_cpu[index][:count]) / count
                    if count
                    else float("-inf")
                ),
                "new_tokens": count,
                "finished": bool(finished_cpu[index]),
            }
        )
    return results, {
        "batch_size": batch_size,
        "max_prompt_tokens": max_prompt,
        "padded_prompt_tokens": batch_size * max_prompt,
        "max_live_tokens": batch_size * (max_prompt + max_new_tokens),
        "generated_tokens": sum(map(int, count_cpu)),
        "decode_steps": steps,
        "cpu_gpu_completion_syncs": sync_count,
        "elapsed_seconds": elapsed,
    }


def prepare_local_work(
    challenges: dict[str, Any],
    config: dict[str, Any],
    tokenizer: Any,
    *,
    rank: int,
    world_size: int,
) -> tuple[list[TestContext], list[PromptUnit]]:
    test_units = [
        (task_id, test_index)
        for task_id in sorted(challenges)
        for test_index in range(len(challenges[task_id]["test"]))
    ]
    max_input_tokens = int(config["prompt"]["max_sequence_length"]) - int(
        config["prompt"]["max_new_tokens"]
    )
    contexts: list[TestContext] = []
    flat: list[PromptUnit] = []
    for global_test_index, (task_id, test_index) in enumerate(test_units):
        if global_test_index % world_size != rank:
            continue
        task = challenges[task_id]
        selection_specs = build_author_view_specs(
            len(task["train"]),
            augment_n=int(config["selection_views"]["author_augment_n"]),
            seed=int(config["selection_views"]["seed"]),
        )
        if len(selection_specs) != int(config["selection_views"]["count"]):
            raise ValueError("Frozen selection view count does not match generated views")
        specs = (canonical_view_spec(len(task["train"])), *selection_specs)
        prompt_units = []
        for position, spec in enumerate(specs):
            transformed = transform_test_view(task, test_index, spec)
            prompt, prompt_tokens, removed = truncate_prompt_to_tokens(
                transformed,
                lambda text: tokenizer.encode(text, add_special_tokens=False),
                max_input_tokens,
            )
            token_ids = tuple(tokenizer.encode(prompt, add_special_tokens=False))
            if len(token_ids) != prompt_tokens:
                raise AssertionError("Prompt token count changed between deterministic encodes")
            unit = PromptUnit(
                global_test_index=global_test_index,
                position=position,
                token_ids=token_ids,
                prompt_tokens=prompt_tokens,
                removed_train_examples=removed,
                view_descriptor=spec.descriptor,
            )
            prompt_units.append(unit)
            flat.append(unit)
        contexts.append(
            TestContext(
                global_test_index=global_test_index,
                task_id=task_id,
                test_index=test_index,
                specs=tuple(specs),
                units=tuple(prompt_units),
            )
        )
    return contexts, flat


def make_batches(
    contexts: Sequence[TestContext],
    flat: Sequence[PromptUnit],
    *,
    batch_mode: str,
    max_batch_tokens: int,
    max_batch_size: int,
    max_new_tokens: int,
) -> list[list[PromptUnit]]:
    if batch_mode == "per_test":
        return [list(context.units) for context in contexts]
    return plan_token_batches(
        flat,
        max_batch_tokens=max_batch_tokens,
        max_batch_size=max_batch_size,
        max_new_tokens=max_new_tokens,
    )


def assemble_records(
    contexts: Sequence[TestContext], decoded: dict[PromptUnit, dict[str, Any]], rank: int, total_tests: int
) -> list[dict[str, Any]]:
    local_records = []
    for context in contexts:
        predictions = [decoded[unit] for unit in context.units]
        canonical = predictions[0]
        selection_specs = context.specs[1:]
        view_predictions = []
        selection_view_records = []
        for spec, prediction, unit in zip(selection_specs, predictions[1:], context.units[1:]):
            inverted = invert_grid(prediction["grid"], spec) if prediction["grid"] is not None else None
            view_predictions.append(
                ViewPrediction(
                    view_index=spec.view_index,
                    grid=inverted,
                    normalized_logprob=prediction["normalized_logprob"],
                )
            )
            selection_view_records.append(
                {
                    "view_index": spec.view_index,
                    "view_descriptor": unit.view_descriptor,
                    "grid": inverted,
                    "normalized_logprob": prediction["normalized_logprob"],
                    "new_tokens": prediction["new_tokens"],
                    "prompt_tokens": unit.prompt_tokens,
                    "removed_train_examples": unit.removed_train_examples,
                }
            )
        ranked = rank_view_predictions(view_predictions)
        record = {
            "global_test_index": context.global_test_index,
            "task_id": context.task_id,
            "test_index": context.test_index,
            "canonical_prediction": canonical["grid"],
            "canonical_normalized_logprob": canonical["normalized_logprob"],
            "canonical_prompt_tokens": context.units[0].prompt_tokens,
            "canonical_removed_train_examples": context.units[0].removed_train_examples,
            "selection_views": selection_view_records,
            "selected_candidates": [
                {
                    "grid": candidate.grid,
                    "votes": candidate.votes,
                    "mean_normalized_logprob": candidate.mean_normalized_logprob,
                    "serialized": candidate.serialized,
                }
                for candidate in ranked[:2]
            ],
            "view_summary": summarize_view_counts(view_predictions),
        }
        local_records.append(record)
        print(
            f"rank={rank} test={context.global_test_index + 1}/{total_tests} "
            f"task={context.task_id} test_index={context.test_index} "
            f"valid={record['view_summary']['valid_view_count']}/16",
            flush=True,
        )
    return local_records


def aggregate_performance(output_dir: Path, world_size: int, started_at: float) -> dict[str, Any]:
    ranks = [
        json.loads((output_dir / f"performance.rank{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    batches = [batch for rank in ranks for batch in rank["batches"]]
    return {
        "format_version": "nvarc_fast_eval_performance_v1",
        "batch_mode": ranks[0]["batch_mode"],
        "max_batch_tokens": ranks[0]["max_batch_tokens"],
        "max_batch_size": ranks[0]["max_batch_size"],
        "world_size": world_size,
        "rank_count": len(ranks),
        "prompt_count": sum(rank["prompt_count"] for rank in ranks),
        "test_case_count": sum(rank["test_case_count"] for rank in ranks),
        "batch_count": sum(rank["batch_count"] for rank in ranks),
        "mean_batch_size": statistics.mean(batch["batch_size"] for batch in batches),
        "max_observed_batch_size": max(batch["batch_size"] for batch in batches),
        "max_observed_live_tokens": max(batch["max_live_tokens"] for batch in batches),
        "max_rank_decode_seconds": max(rank["decode_seconds"] for rank in ranks),
        "max_gpu_memory_allocated_bytes": max(rank["max_gpu_memory_allocated_bytes"] for rank in ranks),
        "total_wall_seconds": time.time() - started_at,
        "ranks": ranks,
    }


def main() -> None:
    args = parse_args()
    if args.batch_mode == "per_test" and args.max_batch_tokens != 0:
        raise ValueError("per_test mode must use max_batch_tokens=0")
    if args.batch_mode == "token_budget" and args.max_batch_tokens <= 0:
        raise ValueError("token_budget mode requires --max-batch-tokens")
    started_at = time.time()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, local_rank, world_size = reference._distributed_context(torch)
    device = torch.device("cuda", local_rank)
    config, config_sha256, protocol_version, identity_validator = reference._load_eval_protocol(args.config)
    if args.expected_world_size is not None and world_size != args.expected_world_size:
        raise ValueError(
            f"Expected {args.expected_world_size} GPU processes, found {world_size}"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Internal eval output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    challenges = reference.load_challenges(args.challenges)
    challenge_identity = identity_validator(challenges)
    if challenge_identity["challenge_semantic_sha256"] != config["dataset"]["challenge_semantic_sha256"]:
        raise ValueError("Challenge data does not match the frozen internal config")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(device)
    adapter_runtime = None
    if args.adapter is not None:
        metadata = reference.load_lora_adapter_metadata(args.adapter)
        if Path(metadata["base_model_name_or_path"]).name != args.model.resolve().name:
            raise ValueError("LoRA adapter base model does not match --model")
        adapter_runtime = reference.attach_native_lora_adapter(model, args.adapter, torch, metadata)
    model.eval()
    token_contract = reference._verify_token_contract(tokenizer, model)
    contexts, flat = prepare_local_work(
        challenges, config, tokenizer, rank=rank, world_size=world_size
    )
    max_new_tokens = int(config["prompt"]["max_new_tokens"])
    batches = make_batches(
        contexts,
        flat,
        batch_mode=args.batch_mode,
        max_batch_tokens=args.max_batch_tokens,
        max_batch_size=args.max_batch_size,
        max_new_tokens=max_new_tokens,
    )
    torch.cuda.reset_peak_memory_stats(device)
    decoded: dict[PromptUnit, dict[str, Any]] = {}
    batch_stats = []
    decode_started = time.perf_counter()
    for batch_index, batch in enumerate(batches):
        outputs, stats = constrained_greedy_vectorized(
            model,
            [unit.token_ids for unit in batch],
            token_contract,
            max_new_tokens,
            device,
            torch,
        )
        decoded.update(zip(batch, outputs))
        stats["batch_index"] = batch_index
        batch_stats.append(stats)
        print(
            f"rank={rank} batch={batch_index + 1}/{len(batches)} size={stats['batch_size']} "
            f"live_tokens={stats['max_live_tokens']} seconds={stats['elapsed_seconds']:.3f}",
            flush=True,
        )
    decode_seconds = time.perf_counter() - decode_started
    records = assemble_records(
        contexts,
        decoded,
        rank,
        int(config["dataset"]["test_case_count"]),
    )
    reference._write_jsonl(args.output_dir / f"predictions.rank{rank}.jsonl", records)
    rank_performance = {
        "format_version": "nvarc_fast_eval_rank_performance_v1",
        "rank": rank,
        "batch_mode": args.batch_mode,
        "max_batch_tokens": args.max_batch_tokens,
        "max_batch_size": args.max_batch_size,
        "test_case_count": len(contexts),
        "prompt_count": len(flat),
        "batch_count": len(batches),
        "decode_seconds": decode_seconds,
        "max_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "batches": batch_stats,
    }
    reference._write_json(args.output_dir / f"performance.rank{rank}.json", rank_performance)
    reference._barrier(torch, world_size)
    if rank == 0:
        reference._merge_and_score(
            args,
            config,
            config_sha256,
            challenge_identity,
            token_contract,
            world_size,
            started_at,
            torch,
            transformers,
            adapter_runtime,
            protocol_version,
            identity_validator,
        )
        performance = aggregate_performance(args.output_dir, world_size, started_at)
        reference._write_json(args.output_dir / "performance.json", performance)
        manifest_path = args.output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evaluator_artifacts"][Path(__file__).name] = sha256_file(Path(__file__))
        manifest["fast_evaluator"] = {
            "batch_mode": args.batch_mode,
            "max_batch_tokens": args.max_batch_tokens,
            "max_batch_size": args.max_batch_size,
            "performance_sha256": sha256_file(args.output_dir / "performance.json"),
        }
        reference._write_json(manifest_path, manifest)
    reference._barrier(torch, world_size)
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
        raise
