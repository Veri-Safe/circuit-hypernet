from __future__ import annotations

from contextlib import nullcontext
from collections.abc import Sequence
from typing import Any, Mapping

import torch
from torch import Tensor

from nvarc_qwen3.nvarc_grid_grammar import GridGrammarState, decode_grid_tokens

from .model import FrozenQwenPerQueryLoRA


class TensorGridGrammar:
    """GPU-resident equivalent of a batch of scalar GridGrammarState objects."""

    def __init__(
        self,
        batch_size: int,
        token_contract: Mapping[str, Any],
        device: torch.device,
    ) -> None:
        self.digit_ids = torch.tensor(
            token_contract["digit_token_ids"], device=device, dtype=torch.long,
        )
        self.newline_id = int(token_contract["newline_token_id"])
        self.eos_id = int(token_contract["eos_token_id"])
        self.pad_id = int(token_contract["pad_token_id"])
        # The production scalar decoder masks the full vocabulary and then
        # applies argmax, so equal scores are resolved by the lowest token id.
        self.candidate_ids = torch.tensor(
            sorted((
                *map(int, token_contract["digit_token_ids"]),
                self.newline_id,
                self.eos_id,
            )),
            device=device,
            dtype=torch.long,
        )
        self.candidate_is_digit = (
            self.candidate_ids[:, None] == self.digit_ids[None, :]
        ).any(dim=1)
        self.completed_rows = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.current_width = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.target_width = torch.full(
            (batch_size,), -1, device=device, dtype=torch.long,
        )
        self.finished = torch.zeros(batch_size, device=device, dtype=torch.bool)

    def select_and_consume(self, log_probs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        active = ~self.finished
        row_limit = torch.where(self.target_width >= 0, self.target_width, 30)
        digit_allowed = active & (self.current_width < row_limit)
        row_complete = active & (self.current_width > 0) & (
            (self.target_width < 0) | (self.current_width == self.target_width)
        )
        newline_allowed = row_complete & (self.completed_rows + 1 < 30)
        allowed = torch.where(
            self.candidate_is_digit[None, :],
            digit_allowed[:, None],
            torch.where(
                (self.candidate_ids == self.newline_id)[None, :],
                newline_allowed[:, None],
                row_complete[:, None],
            ),
        )
        candidate_scores = log_probs.index_select(1, self.candidate_ids)
        offsets = candidate_scores.masked_fill(
            ~allowed, float("-inf"),
        ).argmax(dim=1)
        selected = self.candidate_ids.index_select(0, offsets)
        selected_logprobs = log_probs.gather(1, selected[:, None]).squeeze(1)
        selected = torch.where(active, selected, self.pad_id)
        digit = active & (selected[:, None] == self.digit_ids[None, :]).any(dim=1)
        newline = active & (selected == self.newline_id)
        eos = active & (selected == self.eos_id)
        first_row_end = (newline | eos) & (self.target_width < 0)
        self.target_width = torch.where(
            first_row_end, self.current_width, self.target_width,
        )
        self.current_width = torch.where(
            newline | eos,
            torch.zeros_like(self.current_width),
            self.current_width + digit.to(self.current_width.dtype),
        )
        self.completed_rows += (newline | eos).to(self.completed_rows.dtype)
        self.finished |= eos
        return selected, selected_logprobs, active


def verify_token_contract(tokenizer: Any, model: Any) -> dict[str, Any]:
    expected_literals = {str(value): value for value in range(10)}
    expected_literals.update({"\n": 10, "user": 11, "assistant": 12})
    literal_ids: dict[str, int] = {}
    for literal, expected_id in expected_literals.items():
        encoded = tokenizer.encode(literal, add_special_tokens=False)
        if encoded != [expected_id]:
            raise ValueError(
                f"Tokenizer contract mismatch for {literal!r}: {encoded} != {[expected_id]}"
            )
        literal_ids[literal] = expected_id
    for token, expected_id in {"<|im_start|>": 14, "<|im_end|>": 15}.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if actual != expected_id:
            raise ValueError(f"Tokenizer contract mismatch for {token}: {actual} != {expected_id}")
    if tokenizer.pad_token_id != 13 or tokenizer.eos_token_id != 15:
        raise ValueError("NVARC tokenizer pad/eos contract changed")
    if int(model.config.vocab_size) != 16:
        raise ValueError("NVARC Qwen model vocab must contain exactly 16 tokens")
    return {
        "digit_token_ids": [literal_ids[str(value)] for value in range(10)],
        "newline_token_id": literal_ids["\n"],
        "pad_token_id": 13,
        "eos_token_id": 15,
        "vocab_size": 16,
    }


def generate_lora_for_prompts(
    system: FrozenQwenPerQueryLoRA,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    device: torch.device,
) -> Mapping[str, Tensor]:
    """Generate one independent LoRA per condition prompt in a real batch."""

    if not prompts:
        raise ValueError("prompts must not be empty")
    # Match training: real condition tokens keep positions 0..length-1.
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
        )


def generate_lora_for_prompt(
    system: FrozenQwenPerQueryLoRA,
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device,
) -> Mapping[str, Tensor]:
    """Compatibility wrapper for callers that generate one LoRA."""

    return generate_lora_for_prompts(system, tokenizer, (prompt,), device=device)


def constrained_greedy_batch(
    system: FrozenQwenPerQueryLoRA,
    tokenizer: Any,
    target_prompts: Sequence[str],
    *,
    token_contract: Mapping[str, Any],
    max_new_tokens: int,
    device: torch.device,
    condition_prompts: Sequence[str] | None,
    precomputed_state: Mapping[str, Tensor] | None = None,
) -> list[dict[str, Any]]:
    """Decode a batch while keeping one generated LoRA per batch item."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not target_prompts:
        raise ValueError("target_prompts must not be empty")
    if condition_prompts is not None and len(condition_prompts) != len(target_prompts):
        raise ValueError("condition_prompts must align one-to-one with target_prompts")
    if condition_prompts is not None and precomputed_state is not None:
        raise ValueError("Set condition_prompts or precomputed_state, not both")
    state = precomputed_state
    if condition_prompts is not None:
        state = generate_lora_for_prompts(
            system,
            tokenizer,
            condition_prompts,
            device=device,
        )
    activation = system.activate(state) if state is not None else nullcontext()
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        list(target_prompts),
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    grammars = [GridGrammarState() for _ in target_prompts]
    generated: list[list[int]] = [[] for _ in target_prompts]
    logprob_sums = [0.0 for _ in target_prompts]
    digit_ids = tuple(int(value) for value in token_contract["digit_token_ids"])
    newline_id = int(token_contract["newline_token_id"])
    eos_id = int(token_contract["eos_token_id"])
    pad_id = int(token_contract["pad_token_id"])

    with torch.inference_mode(), activation:
        outputs = system.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        for _ in range(max_new_tokens):
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            active_before = [not grammar.finished for grammar in grammars]
            allowed_mask_rows: list[list[bool]] = []
            for grammar in grammars:
                allowed_mask = [False] * log_probs.shape[-1]
                if grammar.finished:
                    allowed_mask[pad_id] = True
                    allowed_mask_rows.append(allowed_mask)
                    continue
                allowed = grammar.allowed_token_ids(digit_ids, newline_id, eos_id)
                if not allowed:
                    raise RuntimeError(f"Grid grammar reached a dead end: {grammar}")
                for token_id in allowed:
                    allowed_mask[int(token_id)] = True
                allowed_mask_rows.append(allowed_mask)
            allowed_mask_tensor = torch.tensor(
                allowed_mask_rows,
                device=device,
                dtype=torch.bool,
            )
            selected_tokens = torch.argmax(
                log_probs.masked_fill(~allowed_mask_tensor, float("-inf")),
                dim=-1,
            )
            selected_logprobs = log_probs.gather(
                1,
                selected_tokens.unsqueeze(1),
            ).squeeze(1)
            selected_rows = torch.stack(
                (selected_tokens.to(selected_logprobs.dtype), selected_logprobs),
                dim=1,
            ).cpu().tolist()
            next_tokens = [int(row[0]) for row in selected_rows]
            for batch_index, grammar in enumerate(grammars):
                if grammar.finished:
                    continue
                token_id = next_tokens[batch_index]
                generated[batch_index].append(token_id)
                logprob_sums[batch_index] += float(selected_rows[batch_index][1])
                grammar.consume(token_id, digit_ids, newline_id, eos_id)
            if all(grammar.finished for grammar in grammars):
                break
            # Reuse the device-resident argmax result instead of copying the
            # same token ids CPU -> GPU again for every decode step.
            next_token = selected_tokens.unsqueeze(1)
            new_mask = torch.tensor(
                active_before,
                device=device,
                dtype=attention_mask.dtype,
            ).unsqueeze(1)
            attention_mask = torch.cat(
                (attention_mask, new_mask),
                dim=1,
            )
            outputs = system.base_model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

    digit_by_id = {token_id: value for value, token_id in enumerate(digit_ids)}
    return [
        {
            "token_ids": tokens,
            "grid": decode_grid_tokens(tokens, digit_by_id, newline_id, eos_id),
            "normalized_logprob": (
                logprob_sums[index] / len(tokens) if tokens else float("-inf")
            ),
            "new_tokens": len(tokens),
            "finished": grammars[index].finished,
        }
        for index, tokens in enumerate(generated)
    ]


def constrained_greedy_batch_vectorized(
    system: FrozenQwenPerQueryLoRA,
    tokenizer: Any,
    target_prompts: Sequence[str],
    *,
    token_contract: Mapping[str, Any],
    max_new_tokens: int,
    device: torch.device,
    condition_prompts: Sequence[str] | None,
    precomputed_state: Mapping[str, Tensor] | None = None,
) -> list[dict[str, Any]]:
    """Exact greedy decoder with grammar state and result buffers on GPU."""

    if max_new_tokens <= 0 or not target_prompts:
        raise ValueError("Vectorized decode requires prompts and positive max_new_tokens")
    if condition_prompts is not None and len(condition_prompts) != len(target_prompts):
        raise ValueError("condition_prompts must align one-to-one with target_prompts")
    if condition_prompts is not None and precomputed_state is not None:
        raise ValueError("Set condition_prompts or precomputed_state, not both")
    state = precomputed_state
    if condition_prompts is not None:
        state = generate_lora_for_prompts(
            system, tokenizer, condition_prompts, device=device,
        )
    activation = system.activate(state) if state is not None else nullcontext()
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        list(target_prompts), add_special_tokens=False, padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    original_mask = encoded["attention_mask"].to(device)
    batch_size, prompt_width = input_ids.shape
    attention_mask = torch.zeros(
        (batch_size, prompt_width + max_new_tokens),
        device=device,
        dtype=original_mask.dtype,
    )
    attention_mask[:, :prompt_width] = original_mask
    grammar = TensorGridGrammar(batch_size, token_contract, device)
    pad_id = int(token_contract["pad_token_id"])
    generated = torch.full(
        (batch_size, max_new_tokens), pad_id, device=device, dtype=torch.long,
    )
    selected_logprobs = torch.zeros(
        (batch_size, max_new_tokens), device=device, dtype=torch.float32,
    )
    token_count = torch.zeros(batch_size, device=device, dtype=torch.long)

    with torch.inference_mode(), activation:
        outputs = system.base_model(
            input_ids=input_ids,
            attention_mask=original_mask,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        steps = 0
        for step in range(max_new_tokens):
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
            selected, logprobs, active = grammar.select_and_consume(log_probs)
            generated[:, step] = selected
            selected_logprobs[:, step] = torch.where(
                active, logprobs, torch.zeros_like(logprobs),
            )
            token_count += active.to(token_count.dtype)
            steps = step + 1
            if bool(grammar.finished.all().item()):
                break
            attention_mask[:, prompt_width + step] = active.to(attention_mask.dtype)
            outputs = system.base_model(
                input_ids=selected[:, None],
                attention_mask=attention_mask[:, : prompt_width + step + 1],
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

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
        results.append({
            "token_ids": tokens,
            "grid": decode_grid_tokens(
                tokens,
                digit_by_id,
                int(token_contract["newline_token_id"]),
                int(token_contract["eos_token_id"]),
            ),
            "normalized_logprob": (
                sum(float(value) for value in logprob_cpu[index][:count]) / count
                if count else float("-inf")
            ),
            "new_tokens": count,
            "finished": bool(finished_cpu[index]),
        })
    return results


def constrained_greedy_query(
    system: FrozenQwenPerQueryLoRA,
    tokenizer: Any,
    target_prompt: str,
    *,
    token_contract: Mapping[str, Any],
    max_new_tokens: int,
    device: torch.device,
    condition_prompt: str | None,
) -> dict[str, Any]:
    """Compatibility wrapper for single-query decoding."""

    condition_prompts = None if condition_prompt is None else (condition_prompt,)
    return constrained_greedy_batch(
        system,
        tokenizer,
        (target_prompt,),
        token_contract=token_contract,
        max_new_tokens=max_new_tokens,
        device=device,
        condition_prompts=condition_prompts,
    )[0]
