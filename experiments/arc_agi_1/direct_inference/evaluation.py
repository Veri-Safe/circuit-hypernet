from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from nvarc_qwen3.arc_eval_protocol import load_challenges, load_solutions

from s2_hypernet.nvarc_pg.checkpoint import inspect_frozen_hf_checkpoint
from s2_hypernet.nvarc_pg.decode import constrained_greedy_batch, verify_token_contract
from s2_hypernet.nvarc_pg.eval_aggregation import group_query_predictions
from s2_hypernet.nvarc_pg.eval_manifest import build_query_views, query_manifest_sha256
from s2_hypernet.nvarc_pg.eval_protocol import (
    load_profile_eval_protocol,
    validate_profile_dataset_identity,
)
from s2_hypernet.nvarc_pg.eval_scoring import score_grouped_predictions
from s2_hypernet.nvarc_pg.io import read_jsonl, write_jsonl_atomic
from s2_hypernet.nvarc_pg.runtime import (
    cleanup_distributed,
    resolve_process_group_timeout_seconds,
    set_reproducible_seed,
    setup_distributed,
)

from .artifacts import load_vector_bank
from .arc59_interventions import (
    load_intervention,
    scoped_group_gating,
    validate_intervention_evaluation,
)
from .batches import validate_logical_batch_manifest
from .checkpoint import S4TrainerState, attention_backend_identity
from .config import load_config, workspace_relative_path
from .contracts import (
    AUTHOR4_VALIDATION_QUERY_COUNT,
    CANONICAL_VALIDATION_QUERY_COUNT,
    CHECKPOINT_FORMAT,
    EVALUATION_FORMAT,
    FULL_EVAL_QUERY_COUNT,
    FULL_EVAL_TASK_COUNT,
    semantic_sha256,
    sha256_file,
    validate_full_evaluation_request,
)
from .join import load_eval_join_bundle, load_join_bundle
from .lora_scale import (
    GeneratedLoRANormAccumulator,
    scale_generated_lora_state,
    validate_lora_scale_multiplier,
)
from .runtime import build_evaluation_system


AssignmentMode = Literal["matched", "absent", "shuffled"]
DEFAULT_EVALUATION_PROCESS_GROUP_TIMEOUT_SECONDS = 3600.0
# S4 evaluation has no DDP gradient collectives.  Keep its coordination
# barriers off the CUDA/NCCL process group so a slow decode cannot trigger the
# NCCL watchdog while another rank is still working.
EVALUATION_PROCESS_GROUP_BACKEND = "gloo"
DISTRIBUTED_SHARDING = "balanced_prompt_tokens_v1"


def _evaluation_process_group_timeout_seconds() -> float:
    timeout = resolve_process_group_timeout_seconds(
        env_name="S4_EVAL_PROCESS_GROUP_TIMEOUT_SECONDS",
        default=DEFAULT_EVALUATION_PROCESS_GROUP_TIMEOUT_SECONDS,
    )
    if timeout is None:
        raise RuntimeError("S4 evaluation process-group timeout unexpectedly unset")
    return timeout


def validation_query_views(
    all_query_views: Sequence[Any],
    *,
    scope: str,
) -> tuple[Any, ...]:
    if scope == "full":
        return tuple(all_query_views)
    if scope == "canonical":
        return tuple(view for view in all_query_views if bool(view.is_canonical))
    if scope == "author4":
        return tuple(
            view
            for view in all_query_views
            if not bool(view.is_canonical) and int(view.view_index) < 4
        )
    raise ValueError(f"Unsupported S4 validation scope: {scope}")


def circuit_assignment(
    query_count: int, *, mode: AssignmentMode, seed: int
) -> tuple[int, ...]:
    if query_count <= 0:
        raise ValueError("Evaluation query_count must be positive")
    identity = np.arange(query_count)
    if mode in {"matched", "absent"}:
        return tuple(int(value) for value in identity)
    if mode != "shuffled":
        raise ValueError(f"Unsupported S4 assignment mode: {mode}")
    rng = np.random.default_rng(seed)
    for _ in range(100):
        candidate = rng.permutation(query_count)
        if np.all(candidate != identity):
            return tuple(int(value) for value in candidate)
    offset = 1 + seed % (query_count - 1)
    return tuple(int(value) for value in np.roll(identity, offset))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _barrier(local_rank: int) -> None:
    backend = dist.get_backend()
    backend_name = getattr(backend, "value", backend)
    if str(backend_name).lower() == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def _generate_lora(
    system: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    circuit_vectors: torch.Tensor,
    circuit_present: torch.Tensor,
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        list(prompts),
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.inference_mode():
        backbone = getattr(system.base_model, "model", None)
        if backbone is None:
            outputs = system.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            condition = outputs.hidden_states[-1].detach()
        else:
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            condition = outputs.last_hidden_state.detach()
        return system.parameter_generator(
            condition,
            condition_attention_mask=attention_mask,
            circuit_vector=circuit_vectors.to(device=device, dtype=torch.float32),
            circuit_present=circuit_present.to(device=device, dtype=torch.bool),
        )


def _local_batches(
    query_views: Sequence[Any], *, rank: int, world_size: int
) -> tuple[tuple[Any, ...], ...]:
    task_ranks = _task_rank_assignment(query_views, world_size=world_size)
    local = tuple(
        view
        for view in query_views
        if task_ranks[int(view.global_test_index)] == rank
    )
    batches: list[tuple[Any, ...]] = []
    pending: list[Any] = []
    pending_test: int | None = None
    for view in local:
        if pending and int(view.global_test_index) != pending_test:
            batches.append(tuple(pending))
            pending = []
        pending.append(view)
        pending_test = int(view.global_test_index)
    if pending:
        batches.append(tuple(pending))
    return tuple(batches)


def _task_rank_assignment(
    query_views: Sequence[Any], *, world_size: int
) -> dict[int, int]:
    """Assign whole test-input batches to ranks using a deterministic cost proxy.

    A test input's author/canonical views must stay together because they are
    evaluated as one batch.  Prompt-token padding is the dominant predictable
    cost of that batch, so longest batches are placed first on the currently
    lightest rank.  The final prediction stream is still restored by query
    index; this only changes which rank performs each task.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    grouped: dict[int, list[Any]] = {}
    for view in query_views:
        grouped.setdefault(int(view.global_test_index), []).append(view)

    task_costs: dict[int, int] = {}
    for task_index, views in grouped.items():
        prompt_lengths = [
            max(
                1,
                int(
                    getattr(
                        view,
                        "prompt_tokens",
                        len(str(getattr(view, "prompt", ""))),
                    )
                ),
            )
            for view in views
        ]
        # The batched forwards pad every view to the longest prompt in this
        # task, so max_length * batch_size is a stable, protocol-independent
        # estimate for scheduling.
        task_costs[task_index] = max(prompt_lengths) * len(views)

    loads = [0] * world_size
    assignments: dict[int, int] = {}
    for task_index, cost in sorted(
        task_costs.items(), key=lambda item: (-item[1], item[0])
    ):
        assigned_rank = min(
            range(world_size), key=lambda candidate: (loads[candidate], candidate)
        )
        assignments[task_index] = assigned_rank
        loads[assigned_rank] += cost
    return assignments


def _wait_for_rank_predictions(
    destination: Path,
    *,
    query_views: Sequence[Any],
    world_size: int,
    timeout_seconds: float = 86_400.0,
) -> list[dict[str, Any]]:
    task_ranks = _task_rank_assignment(query_views, world_size=world_size)
    paths = tuple(
        destination / f"query_predictions.rank{rank}.jsonl"
        for rank in range(world_size)
    )
    deadline = time.monotonic() + timeout_seconds
    while True:
        missing = [rank for rank, path in enumerate(paths) if not path.is_file()]
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for S4 eval ranks: {missing}")
        time.sleep(5.0)
    records: list[dict[str, Any]] = []
    for rank, path in enumerate(paths):
        values = read_jsonl(path)
        expected = tuple(
            int(view.query_index)
            for view in query_views
            if task_ranks[int(view.global_test_index)] == rank
        )
        actual = tuple(int(value["query_index"]) for value in values)
        if actual != expected:
            raise ValueError(f"S4 eval rank {rank} query order changed")
        records.extend(values)
    records.sort(key=lambda value: int(value["query_index"]))
    return records


def _checkpoint_identity(
    *,
    config: Any,
    s2: Any,
    base_manifest_sha256: str,
    train_join_metadata: Mapping[str, Any],
    fresh_init_sha256: str,
) -> S4TrainerState:
    return S4TrainerState(
        format=CHECKPOINT_FORMAT,
        global_step=0,
        rows_consumed=0,
        variant_id=config.variant_id,
        training_config_sha256=config.training_semantic_sha256,
        base_manifest_sha256=base_manifest_sha256,
        layout_sha256=s2.target_layout.layout_sha256,
        train_root_manifest_sha256=config.train_artifacts.root_manifest_sha256,
        train_join_sha256=str(train_join_metadata["join_sha256"]),
        batch_manifest_sha256=sha256_file(config.batch_manifest),
        fresh_init_sha256=fresh_init_sha256,
        attention_backend=attention_backend_identity(),
    )


def run_evaluation(
    *,
    config_path: str | Path,
    checkpoint: str | Path,
    train_join_path: str | Path,
    eval_join_path: str | Path,
    output_dir: str | Path,
    mode: AssignmentMode,
    max_eval_tasks: int | None = None,
    lora_scale_multiplier: float = 1.0,
    lora_norm_diagnostics: bool = False,
    lora_norm_max_queries_per_rank: int | None = None,
    experiment_label: str | None = None,
    validation_scope: str = "full",
    circuit_intervention: str | Path | None = None,
) -> None:
    validate_full_evaluation_request(max_eval_tasks)
    if validation_scope not in {"full", "canonical", "author4"}:
        raise ValueError(f"Unsupported S4 validation scope: {validation_scope}")
    config = load_config(config_path)
    lora_scale_multiplier = validate_lora_scale_multiplier(lora_scale_multiplier)
    intervention = None
    if circuit_intervention is not None:
        validate_intervention_evaluation(
            variant_id=config.variant_id,
            mode=mode,
            validation_scope=validation_scope,
            lora_scale_multiplier=lora_scale_multiplier,
        )
        intervention = load_intervention(circuit_intervention)
    from s2_hypernet.nvarc_pg.config import load_experiment_config

    s2 = load_experiment_config(config.s2_config)
    context = setup_distributed(
        config.effective_world_size,
        process_group_timeout_seconds=_evaluation_process_group_timeout_seconds(),
        backend=EVALUATION_PROCESS_GROUP_BACKEND,
    )
    system = None
    intervention_scope = ExitStack()
    destination = Path(output_dir).resolve()
    try:
        set_reproducible_seed(config.seed, rank=context.rank)
        train_join, train_join_metadata = load_join_bundle(train_join_path)
        if len(train_join) != config.train_artifacts.expected_events:
            raise ValueError("Training join is incomplete during evaluation")
        eval_join, eval_join_metadata = load_eval_join_bundle(eval_join_path)
        if (
            eval_join_metadata.get("eval_artifact_contract_sha256")
            != config.eval_artifact_contract_sha256
        ):
            raise ValueError("Evaluation join belongs to another artifact contract")
        if (
            eval_join_metadata.get("final_root_manifest_sha256")
            != config.eval_artifacts.root_manifest_sha256
        ):
            raise ValueError("Evaluation join belongs to another S3 root")
        validate_logical_batch_manifest(config.batch_manifest, config=config)
        eval_bank = load_vector_bank(config.eval_artifacts)

        protocol, protocol_sha = load_profile_eval_protocol(s2.evaluation)
        challenges = load_challenges(s2.evaluation.challenges)
        if len(challenges) != FULL_EVAL_TASK_COUNT:
            raise ValueError(
                "S4 score-bearing evaluation requires exactly "
                f"{FULL_EVAL_TASK_COUNT} tasks, found {len(challenges)}"
            )
        dataset_identity = validate_profile_dataset_identity(
            s2.evaluation.protocol_kind,
            challenges,
        )

        # Tokenizer is needed to materialize the frozen query manifest, while the
        # model checkpoint identity must be established before S4 PG weights load.
        from s2_hypernet.nvarc_pg.runtime import load_tokenizer

        manifest_tokenizer = load_tokenizer(s2)
        all_query_views = build_query_views(
            challenges,
            config=protocol,
            tokenizer=manifest_tokenizer,
        )
        query_sha = query_manifest_sha256(all_query_views)
        if query_sha != eval_join_metadata.get("query_manifest_sha256"):
            raise ValueError(
                "Evaluation query manifest changed after circuit extraction"
            )
        if len(all_query_views) != len(eval_join):
            raise ValueError("Evaluation circuit coverage is incomplete")
        if len(all_query_views) != FULL_EVAL_QUERY_COUNT:
            raise ValueError(
                "S4 full evaluation query count changed: "
                f"expected {FULL_EVAL_QUERY_COUNT}, found {len(all_query_views)}"
            )
        selected_task_ids = tuple(sorted(challenges))
        query_views = validation_query_views(
            all_query_views,
            scope=validation_scope,
        )
        if validation_scope == "canonical" and (
            len(query_views) != CANONICAL_VALIDATION_QUERY_COUNT
        ):
            raise ValueError(
                "Canonical validation query count changed: "
                f"{len(query_views)} != {CANONICAL_VALIDATION_QUERY_COUNT}"
            )
        if validation_scope == "author4" and (
            len(query_views) != AUTHOR4_VALIDATION_QUERY_COUNT
        ):
            raise ValueError(
                "Author4 validation query count changed: "
                f"{len(query_views)} != {AUTHOR4_VALIDATION_QUERY_COUNT}"
            )
        score_challenges = challenges
        selected_eval_join = tuple(
            eval_join[int(view.query_index)] for view in query_views
        )
        for view, join_record in zip(query_views, selected_eval_join, strict=True):
            if int(join_record.query_index) != int(view.query_index):
                raise ValueError("Evaluation join/query index order changed")
        query_position = {
            int(view.query_index): position for position, view in enumerate(query_views)
        }

        base_manifest = inspect_frozen_hf_checkpoint(
            hf_checkpoint=s2.base.hf_checkpoint,
            tokenizer=s2.base.tokenizer,
            source_run_manifest=s2.base.source_run_manifest,
            profile=s2.profile,
            layout=s2.target_layout,
        )
        base_manifest_sha = semantic_sha256(base_manifest)
        fresh_manifest = json.loads(
            (Path(config.fresh_init_root) / "manifest.json").read_text(encoding="utf-8")
        )
        expected_checkpoint = _checkpoint_identity(
            config=config,
            s2=s2,
            base_manifest_sha256=base_manifest_sha,
            train_join_metadata=train_join_metadata,
            fresh_init_sha256=str(fresh_manifest["aggregate_sha256"]),
        )
        system, tokenizer, _, model = build_evaluation_system(
            config,
            context=context,
            checkpoint=checkpoint,
            expected_checkpoint_identity=expected_checkpoint,
        )
        if intervention is not None:
            intervention_scope.enter_context(scoped_group_gating(model, intervention))
        checkpoint_state = json.loads(
            (Path(checkpoint) / "trainer_state.json").read_text(encoding="utf-8")
        )
        checkpoint_step = int(checkpoint_state["global_step"])
        logical_checkpoint_step = config.logical_step_for_active_step(checkpoint_step)
        training_lora_gate = config.optimization.lora_gate_at_step(
            logical_checkpoint_step
        )
        learned_global_lora_scale = float(
            model.global_lora_scale_value().detach().cpu()
        )
        norm_accumulator = (
            GeneratedLoRANormAccumulator(
                system.base_model,
                target_keys=system.controller.target_keys,
                device=context.device,
                max_queries=lora_norm_max_queries_per_rank,
            )
            if lora_norm_diagnostics
            else None
        )
        token_contract = verify_token_contract(tokenizer, system.base_model)
        assignment = circuit_assignment(
            len(query_views), mode=mode, seed=s2.evaluation.shuffle_seed
        )
        assignment_sha = semantic_sha256(assignment)
        if context.is_rank_zero:
            if destination.exists() and any(destination.iterdir()):
                raise FileExistsError(
                    f"S4 evaluation output is not empty: {destination}"
                )
            destination.mkdir(parents=True, exist_ok=True)
        _barrier(context.local_rank)

        local_batches = _local_batches(
            query_views,
            rank=context.rank,
            world_size=context.world_size,
        )
        local_views = tuple(view for batch in local_batches for view in batch)
        records: list[dict[str, Any]] = []
        complete = 0
        for batch_views in local_batches:
            started = time.perf_counter()
            source_positions = tuple(
                assignment[query_position[int(view.query_index)]]
                for view in batch_views
            )
            source_records = tuple(
                selected_eval_join[position] for position in source_positions
            )
            source_query_indices = tuple(
                int(record.query_index) for record in source_records
            )
            event_keys = tuple(record.event_key for record in source_records)
            vectors = eval_bank.vectors(event_keys)
            if intervention is not None:
                vectors = intervention.apply_vectors(
                    vectors,
                    query_indices=tuple(int(view.query_index) for view in batch_views),
                )
            present = torch.full(
                (len(batch_views),),
                mode != "absent",
                dtype=torch.bool,
            )
            target_prompts = tuple(str(view.prompt) for view in batch_views)
            raw_state = _generate_lora(
                system,
                tokenizer,
                target_prompts,
                circuit_vectors=vectors,
                circuit_present=present,
                device=context.device,
            )
            gated_state = scale_generated_lora_state(
                raw_state,
                training_lora_gate,
            )
            if norm_accumulator is not None:
                norm_accumulator.update(gated_state)
            state = scale_generated_lora_state(
                gated_state,
                lora_scale_multiplier,
            )
            decoded = constrained_greedy_batch(
                system,
                tokenizer,
                target_prompts,
                token_contract=token_contract,
                max_new_tokens=int(protocol["prompt"]["max_new_tokens"]),
                device=context.device,
                condition_prompts=None,
                precomputed_state=state,
            )
            records.extend(
                {
                    **view.manifest_record(),
                    **prediction,
                    "circuit_query_index": source_index,
                    "circuit_event_key": event_key,
                    "circuit_present": mode != "absent",
                    **(intervention.prediction_metadata() if intervention is not None else {}),
                }
                for view, prediction, source_index, event_key in zip(
                    batch_views,
                    decoded,
                    source_query_indices,
                    event_keys,
                    strict=True,
                )
            )
            complete += len(batch_views)
            print(
                f"rank={context.rank} queries={complete}/{len(local_views)} "
                f"batch={len(batch_views)} batch_seconds={time.perf_counter() - started:.3f}",
                flush=True,
            )
        rank_path = destination / f"query_predictions.rank{context.rank}.jsonl"
        write_jsonl_atomic(rank_path, records)
        norm_summary = (
            norm_accumulator.finalize(
                lora_scale_multiplier=lora_scale_multiplier,
            )
            if norm_accumulator is not None
            else None
        )
        if context.is_rank_zero:
            query_records = _wait_for_rank_predictions(
                destination,
                query_views=query_views,
                world_size=context.world_size,
            )
            grouped = group_query_predictions(query_views, query_records)
            write_jsonl_atomic(destination / "query_predictions.jsonl", query_records)
            write_jsonl_atomic(destination / "predictions.jsonl", grouped)
            solutions = load_solutions(s2.evaluation.solutions, challenges)
            dataset_identity = validate_profile_dataset_identity(
                s2.evaluation.protocol_kind,
                challenges,
                solutions,
            )
            score_solutions = {
                task_id: solutions[task_id] for task_id in selected_task_ids
            }
            metrics, submission = score_grouped_predictions(
                score_challenges,
                score_solutions,
                grouped,
            )
            _write_json_atomic(destination / "metrics.json", metrics)
            _write_json_atomic(destination / "submission.json", submission)
            if norm_summary is not None:
                _write_json_atomic(
                    destination / "lora_norm_diagnostics.json",
                    norm_summary,
                )
            _write_json_atomic(
                destination / "manifest.json",
                {
                    "format": EVALUATION_FORMAT,
                    "status": "complete",
                    "variant_id": config.variant_id,
                    "experiment_label": experiment_label,
                    "mode": mode,
                    "checkpoint": workspace_relative_path(checkpoint),
                    "checkpoint_step": int(checkpoint_state["global_step"]),
                    "logical_checkpoint_step": logical_checkpoint_step,
                    "config_sha256": config.semantic_sha256,
                    "training_config_sha256": config.training_semantic_sha256,
                    "attention_backend": attention_backend_identity(),
                    "base_manifest_sha256": base_manifest_sha,
                    "protocol_sha256": protocol_sha,
                    "query_manifest_sha256": query_sha,
                    "eval_join_sha256": eval_join_metadata["join_sha256"],
                    "eval_root_manifest_sha256": config.eval_artifacts.root_manifest_sha256,
                    "circuit_assignment_sha256": assignment_sha,
                    "prompt_condition_assignment": "identity",
                    "circuit_assignment": mode,
                    "dataset_identity": dataset_identity,
                    "world_size": context.world_size,
                    "distributed_sharding": DISTRIBUTED_SHARDING,
                    "query_count": len(query_views),
                    "full_query_count": len(all_query_views),
                    "validation_scope": validation_scope,
                    "validation_selection_metric": (
                        "canonical_exact@1"
                        if validation_scope == "canonical"
                        else "selected_exact@2"
                    ),
                    "max_eval_tasks": None,
                    "evaluated_task_count": len(selected_task_ids),
                    "selected_task_ids_sha256": semantic_sha256(selected_task_ids),
                    "selected_task_ids": list(selected_task_ids),
                    "lora_scale_multiplier": lora_scale_multiplier,
                    "training_lora_gate_multiplier": training_lora_gate,
                    "learned_global_lora_scale": learned_global_lora_scale,
                    "lora_norm_diagnostics": lora_norm_diagnostics,
                    "lora_norm_max_queries_per_rank": (lora_norm_max_queries_per_rank),
                    "missing_circuits": 0,
                    "corrupt_circuits": 0,
                    **({"circuit_intervention": intervention.manifest_record()}
                       if intervention is not None else {}),
                },
            )
            metric_name = (
                "canonical_exact@1"
                if validation_scope == "canonical"
                else "selected_exact@2"
            )
            print(
                f"evaluation_status=complete mode={mode} "
                f"validation_scope={validation_scope} "
                f"lora_scale_multiplier={lora_scale_multiplier} "
                f"{metric_name}={metrics[metric_name]}",
                flush=True,
            )
    finally:
        intervention_scope.close()
        if system is not None:
            system.close()
        cleanup_distributed()
