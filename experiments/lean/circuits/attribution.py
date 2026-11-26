"""Native S5 feature decomposition and exact direct-effect row provider."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from ..upstream_runtime.hook_utils import hook_recorder
from .traces import CircuitTrace
from .transcoder import RectangularReLUTranscoder


ROOT_TARGET = -1


@dataclass(frozen=True)
class FullOutputObjective:
    target_token_ids: Tensor
    target_positions: Tensor
    target_probabilities: Tensor
    target_entropies: Tensor
    target_log_probabilities: Tensor
    scalar: Tensor

    @property
    def target_count(self) -> int:
        return int(self.target_token_ids.numel())


def build_full_output_objective(
    logits: Tensor,
    *,
    condition_length: int,
    target_token_ids: Sequence[int],
) -> FullOutputObjective:
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("S5 circuit objective requires batch size one")
        logits = logits[0]
    targets = torch.tensor(
        tuple(int(item) for item in target_token_ids),
        device=logits.device,
        dtype=torch.long,
    )
    if targets.numel() == 0:
        raise ValueError("full-output objective needs at least one target")
    positions = torch.arange(
        int(condition_length) - 1,
        int(condition_length) + int(targets.numel()) - 1,
        device=logits.device,
        dtype=torch.long,
    )
    if int(positions[-1]) >= logits.shape[0]:
        raise ValueError("causal replay is shorter than the requested output target")
    selected = logits.index_select(0, positions).float()
    log_probs = torch.log_softmax(selected, dim=-1)
    probabilities = log_probs.exp()
    target_log_probs = log_probs.gather(1, targets[:, None]).squeeze(1)
    entropies = -(probabilities * log_probs).sum(dim=-1)
    return FullOutputObjective(
        target_token_ids=targets,
        target_positions=positions,
        target_probabilities=target_log_probs.exp(),
        target_entropies=entropies,
        target_log_probabilities=target_log_probs,
        scalar=target_log_probs.sum(),
    )


@dataclass
class S5DecomposedForward:
    logits: Tensor
    objective: FullOutputObjective
    replay_token_ids: Tensor
    feature_tensors: tuple[Tensor, ...]
    error_gates: tuple[Tensor, ...]
    error_vectors: tuple[Tensor, ...]
    token_gates: Tensor
    active_features: Tensor
    activation_values: Tensor

    @property
    def feature_count(self) -> int:
        return int(self.active_features.shape[0])

    @property
    def n_layers(self) -> int:
        return len(self.feature_tensors)

    @property
    def token_count(self) -> int:
        return int(self.replay_token_ids.numel())

    @property
    def error_count(self) -> int:
        return self.n_layers * self.token_count

    @property
    def source_count(self) -> int:
        return self.feature_count + self.error_count + self.token_count


@dataclass
class S5LinearizedForward:
    """Forward-exact local replacement replay with CodeCircuit gradient flow."""

    logits: Tensor
    objective: FullOutputObjective
    replay_token_ids: Tensor
    mlp_inputs: tuple[Tensor, ...]
    feature_tensors: tuple[Tensor, ...]
    error_gates: tuple[Tensor, ...]
    error_vectors: tuple[Tensor, ...]
    token_embedding_leaf: Tensor
    active_features: Tensor
    activation_values: Tensor
    transcoders: tuple[RectangularReLUTranscoder, ...]

    @property
    def feature_count(self) -> int:
        return int(self.active_features.shape[0])

    @property
    def n_layers(self) -> int:
        return len(self.mlp_inputs)

    @property
    def token_count(self) -> int:
        return int(self.replay_token_ids.numel())

    @property
    def error_count(self) -> int:
        return self.n_layers * self.token_count

    @property
    def source_count(self) -> int:
        return self.feature_count + self.error_count + self.token_count


def _forward_tail_from_tokens(
    model: torch.nn.Module,
    token_ids: Tensor,
    *,
    token_gates: Tensor,
) -> tuple[Tensor, Tensor | None]:
    if model.config.enable_bigram_table:
        raise ValueError("initial S5 circuit provider does not support Bigram ON")
    length = int(token_ids.shape[1])
    token_embedding = model.transformer.wte(token_ids)
    position_embedding = model.transformer.wpe.weight[:length].unsqueeze(0)
    x = model.transformer.drop(token_embedding * token_gates[None, :, None])
    positional = (
        model.transformer.drop(position_embedding)
        if model.config.dropout_cat_pos_emb
        else position_embedding
    )
    output = model.forward_tail(
        x,
        n=0,
        targets=None,
        additional_logits_bias=None,
        include_resid_mid=False,
        pos_emb_to_cat=positional,
    )
    return output[0], positional


def decomposed_forward(
    model: torch.nn.Module,
    transcoders: Sequence[RectangularReLUTranscoder],
    trace: CircuitTrace,
    *,
    device: torch.device,
    requires_grad: bool,
    feature_overrides: Mapping[tuple[int, int, int], float] | None = None,
) -> S5DecomposedForward:
    trace.validate()
    if len(transcoders) != int(model.config.n_layer):
        raise ValueError("one Transcoder is required for every S5 layer")
    for transcoder in transcoders:
        if (
            transcoder.d_in != int(model.config.d_model + model.config.d_pos_emb)
            or transcoder.d_out != int(model.config.d_model)
        ):
            raise ValueError("Transcoder dimensions do not match the S5 MLP interface")
        transcoder.to(device=device, dtype=torch.float32).eval()
        for parameter in transcoder.parameters():
            parameter.requires_grad_(False)

    replay = torch.tensor(
        [trace.replay_token_ids], device=device, dtype=torch.long
    )
    token_gates = torch.ones(
        replay.shape[1], device=device, dtype=torch.float32, requires_grad=requires_grad
    )
    layer_inputs: dict[int, Tensor] = {}
    feature_tensors: dict[int, Tensor] = {}
    error_gates: dict[int, Tensor] = {}
    error_vectors: dict[int, Tensor] = {}
    interventions: dict[str, Any] = {}
    overrides = dict(feature_overrides or {})

    for layer, transcoder in enumerate(transcoders):
        key_in = f"{layer}.mlp.act_in"
        key_out = f"{layer}.mlp.resid_delta"

        def remember(value: Tensor, *, layer_index: int = layer) -> Tensor:
            layer_inputs[layer_index] = value
            return value

        def replace(
            original: Tensor,
            *,
            layer_index: int = layer,
            layer_transcoder: RectangularReLUTranscoder = transcoder,
        ) -> Tensor:
            if layer_index not in layer_inputs:
                raise RuntimeError("MLP decomposition ran before input capture")
            x = layer_inputs[layer_index].float()
            original_features = layer_transcoder.encode(x)
            baseline_reconstruction = layer_transcoder.decode(original_features)
            error = (original.float() - baseline_reconstruction).detach()
            used_features = original_features
            relevant = [
                (position, feature, value)
                for (which_layer, position, feature), value in overrides.items()
                if which_layer == layer_index
            ]
            if relevant:
                used_features = original_features.clone()
                for position, feature, value in relevant:
                    if not 0 <= position < used_features.shape[1]:
                        raise ValueError("feature intervention position is outside replay")
                    if not 0 <= feature < used_features.shape[2]:
                        raise ValueError("feature intervention ID is outside dictionary")
                    used_features[0, position, feature] = float(value)
            gate = torch.ones(
                original.shape[:2],
                device=original.device,
                dtype=torch.float32,
                requires_grad=requires_grad,
            )
            output = layer_transcoder.decode(used_features) + gate[..., None] * error
            feature_tensors[layer_index] = used_features
            error_gates[layer_index] = gate
            error_vectors[layer_index] = error
            return output.to(dtype=original.dtype)

        interventions[key_in] = remember
        interventions[key_out] = replace

    gradient_context = torch.enable_grad() if requires_grad else torch.inference_mode()
    with gradient_context, hook_recorder(regex=r"a^", interventions=interventions):
        logits, _position = _forward_tail_from_tokens(
            model, replay, token_gates=token_gates
        )
        objective = build_full_output_objective(
            logits,
            condition_length=trace.condition_length,
            target_token_ids=trace.generated_token_ids,
        )
    if set(feature_tensors) != set(range(len(transcoders))):
        raise RuntimeError("not every S5 layer produced a Transcoder decomposition")

    records = []
    activation_parts = []
    for layer in range(len(transcoders)):
        active = torch.nonzero(feature_tensors[layer][0] > 0, as_tuple=False)
        if active.numel():
            layer_column = torch.full(
                (active.shape[0], 1), layer, device=active.device, dtype=torch.long
            )
            records.append(torch.cat((layer_column, active.long()), dim=1))
            activation_parts.append(
                feature_tensors[layer][0, active[:, 0], active[:, 1]]
            )
    active_features = (
        torch.cat(records, dim=0)
        if records
        else torch.empty((0, 3), dtype=torch.long, device=device)
    )
    activation_values = (
        torch.cat(activation_parts)
        if activation_parts
        else torch.empty(0, dtype=torch.float32, device=device)
    )
    return S5DecomposedForward(
        logits=logits,
        objective=objective,
        replay_token_ids=replay[0],
        feature_tensors=tuple(feature_tensors[layer] for layer in range(len(transcoders))),
        error_gates=tuple(error_gates[layer] for layer in range(len(transcoders))),
        error_vectors=tuple(error_vectors[layer] for layer in range(len(transcoders))),
        token_gates=token_gates,
        active_features=active_features,
        activation_values=activation_values,
    )


def _frozen_rms_norm(module: torch.nn.Module, x: Tensor) -> Tensor:
    """Preserve native RMSNorm values while freezing its denominator in backward."""

    native = module(x)
    eps = module.eps
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    denominator = torch.sqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    linear = x / denominator.detach().to(dtype=x.dtype)
    linear = linear * module.weight
    return native.detach() + linear - linear.detach()


def _frozen_attention_pattern(attention: torch.nn.Module, x: Tensor) -> Tensor:
    """Preserve native SDPA values with gradients only through V and output projection."""

    if attention.config.afrac is not None:
        raise ValueError("linearized dense provider requires activation sparsity disabled")
    if attention.dropout != 0.0:
        raise ValueError("linearized circuit provider requires zero attention dropout")
    batch, length, _width = x.shape
    q, k, v = attention.c_attn(x).split(
        attention.n_head * attention.d_head, dim=2
    )
    q = q.view(batch, length, attention.n_head, attention.d_head).transpose(1, 2)
    k = k.view(batch, length, attention.n_head, attention.d_head).transpose(1, 2)
    v = v.view(batch, length, attention.n_head, attention.d_head).transpose(1, 2)
    scale = 1.0 / math.sqrt(attention.d_head)
    if attention.config.sink:
        native_aggregate = attention.attn_imp(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )
        scores = (q.detach() @ k.detach().transpose(-2, -1)) * scale
        sink = attention.attn_imp.sink_logit.detach().to(scores).view(
            1, attention.n_head, 1, 1
        )
        sink = sink.expand(batch, -1, length, -1)
        scores = torch.cat((sink, scores), dim=-1)
        allow = torch.zeros(
            (length, length + 1), device=x.device, dtype=torch.bool
        )
        allow[:, 0] = True
        allow[:, 1:] = torch.ones(
            (length, length), device=x.device, dtype=torch.bool
        ).tril()
        scores = scores.masked_fill(
            ~allow.view(1, 1, length, length + 1), torch.finfo(scores.dtype).min
        )
        pattern = torch.softmax(scores, dim=-1).detach()
        values = torch.cat(
            (
                torch.zeros(
                    (batch, attention.n_head, 1, attention.d_head),
                    device=v.device,
                    dtype=v.dtype,
                ),
                v,
            ),
            dim=2,
        )
    else:
        native_aggregate = (
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=True,
                scale=scale,
            )
            if attention.flash
            else None
        )
        scores = (q.detach() @ k.detach().transpose(-2, -1)) * scale
        allow = torch.ones((length, length), device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(
            ~allow.view(1, 1, length, length), torch.finfo(scores.dtype).min
        )
        pattern = torch.softmax(scores, dim=-1).detach()
        values = v
        if native_aggregate is None:
            native_aggregate = pattern @ v
    linear_aggregate = pattern @ values
    aggregate = (
        native_aggregate.detach()
        + linear_aggregate
        - linear_aggregate.detach()
    )
    joined = aggregate.transpose(1, 2).contiguous().view(
        batch, length, attention.n_head * attention.d_head
    )
    return attention.resid_dropout(attention.c_proj(joined))


def linearized_forward(
    model: torch.nn.Module,
    transcoders: Sequence[RectangularReLUTranscoder],
    trace: CircuitTrace,
    *,
    device: torch.device,
) -> S5LinearizedForward:
    """Run the forward-exact CodeCircuit local linear replacement model."""

    trace.validate()
    if len(transcoders) != int(model.config.n_layer):
        raise ValueError("one Transcoder is required for every S5 layer")
    if not model.config.cat_pos_emb or model.config.residual_activation_type != "identity":
        raise ValueError("S5 linearized provider requires concat position and identity residual")
    if model.config.enable_bigram_table:
        raise ValueError("initial S5 linearized provider supports Bigram OFF only")
    replay = torch.tensor([trace.replay_token_ids], device=device, dtype=torch.long)
    length = int(replay.shape[1])
    token_embedding_leaf = (
        model.transformer.wte(replay).detach().requires_grad_(True)
    )
    position = model.transformer.wpe.weight[:length].unsqueeze(0).detach()
    x = model.transformer.drop(token_embedding_leaf)
    mlp_inputs: list[Tensor] = []
    feature_tensors: list[Tensor] = []
    error_gates: list[Tensor] = []
    error_vectors: list[Tensor] = []
    active_records: list[Tensor] = []
    activation_parts: list[Tensor] = []
    frozen_transcoders = []

    with torch.enable_grad():
        for layer, (block, transcoder) in enumerate(
            zip(model.transformer.h, transcoders, strict=True)
        ):
            transcoder.to(device=device, dtype=torch.float32).eval()
            for parameter in transcoder.parameters():
                parameter.requires_grad_(False)
            frozen_transcoders.append(transcoder)
            resid_read = _frozen_rms_norm(block.ln_1, x)
            pos_read = block.ln_p1(position).detach()
            attention_input = torch.cat((resid_read, pos_read), dim=-1)
            x = x + _frozen_attention_pattern(block.attn, attention_input)

            mlp_resid = _frozen_rms_norm(block.ln_2, x)
            mlp_pos = block.ln_p2(position).detach()
            mlp_input = torch.cat((mlp_resid, mlp_pos), dim=-1)
            with torch.no_grad():
                native_output = block.mlp(mlp_resid.detach(), mlp_pos)
                baseline_features = transcoder.encode(mlp_input.detach().float())
                baseline_reconstruction = transcoder.decode(baseline_features)
                error = native_output.float() - baseline_reconstruction
            feature_leaf = baseline_features.detach().requires_grad_(True)
            error_gate = torch.ones(
                native_output.shape[:2],
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            )
            replacement = transcoder.decode(feature_leaf) + error_gate[..., None] * error
            x = x + replacement.to(dtype=x.dtype)
            mlp_inputs.append(mlp_input)
            feature_tensors.append(feature_leaf)
            error_gates.append(error_gate)
            error_vectors.append(error.detach())

            active = torch.nonzero(feature_leaf[0] > 0, as_tuple=False)
            if active.numel():
                active_records.append(
                    torch.cat(
                        (
                            torch.full(
                                (active.shape[0], 1),
                                layer,
                                device=device,
                                dtype=torch.long,
                            ),
                            active.long(),
                        ),
                        dim=1,
                    )
                )
                activation_parts.append(
                    feature_leaf[0, active[:, 0], active[:, 1]]
                )

        final = _frozen_rms_norm(model.transformer.ln_f, x)
        logits = model.lm_head(final) + model.final_logits_bias
        objective = build_full_output_objective(
            logits,
            condition_length=trace.condition_length,
            target_token_ids=trace.generated_token_ids,
        )
    active_features = (
        torch.cat(active_records, dim=0)
        if active_records
        else torch.empty((0, 3), dtype=torch.long, device=device)
    )
    activations = (
        torch.cat(activation_parts)
        if activation_parts
        else torch.empty(0, dtype=torch.float32, device=device)
    )
    return S5LinearizedForward(
        logits=logits,
        objective=objective,
        replay_token_ids=replay[0],
        mlp_inputs=tuple(mlp_inputs),
        feature_tensors=tuple(feature_tensors),
        error_gates=tuple(error_gates),
        error_vectors=tuple(error_vectors),
        token_embedding_leaf=token_embedding_leaf,
        active_features=active_features,
        activation_values=activations,
        transcoders=tuple(frozen_transcoders),
    )


class S5DirectEffectRowProvider:
    """Compute exact gradient-times-source direct-effect rows for S3 streaming."""

    def __init__(
        self,
        context: S5LinearizedForward,
        *,
        row_batch_size: int = 8,
        use_batched_vjp: bool = True,
        root_scalar: Tensor | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if row_batch_size <= 0:
            raise ValueError("row_batch_size must be positive")
        if not context.objective.scalar.requires_grad:
            raise ValueError("direct-effect provider requires a gradient-enabled context")
        self.context = context
        self.row_batch_size = int(row_batch_size)
        self.use_batched_vjp = bool(use_batched_vjp)
        self.progress_callback = progress_callback
        self.root_scalar = (
            context.objective.scalar if root_scalar is None else root_scalar
        )
        if self.root_scalar.ndim != 0 or not self.root_scalar.requires_grad:
            raise ValueError("direct-effect root scalar must be a live scalar")
        self.feature_count = context.feature_count
        self.source_count = context.source_count
        self.rows_requested = 0
        self.max_target_block_size = 0
        self.batched_vjp_attempts = 0
        self.batched_vjp_successes = 0
        self.batched_vjp_fallbacks = 0
        self.loop_vjp_calls = 0
        self.batched_vjp_seconds = 0.0
        self.loop_vjp_seconds = 0.0
        self.last_batched_vjp_error: dict[str, str] | None = None

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-safe execution counters without changing row semantics."""

        return {
            "feature_count": self.feature_count,
            "source_count": self.source_count,
            "row_batch_size": self.row_batch_size,
            "use_batched_vjp": self.use_batched_vjp,
            "rows_requested": self.rows_requested,
            "max_target_block_size": self.max_target_block_size,
            "batched_vjp_attempts": self.batched_vjp_attempts,
            "batched_vjp_successes": self.batched_vjp_successes,
            "batched_vjp_fallbacks": self.batched_vjp_fallbacks,
            "loop_vjp_calls": self.loop_vjp_calls,
            "batched_vjp_seconds": self.batched_vjp_seconds,
            "loop_vjp_seconds": self.loop_vjp_seconds,
            "last_batched_vjp_error": self.last_batched_vjp_error,
        }

    def _emit_progress(self, event: dict[str, Any]) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(
            {
                **event,
                "provider": self.diagnostics(),
            }
        )

    def _source_tensors(self) -> tuple[Tensor, ...]:
        return (
            *self.context.feature_tensors,
            *self.context.error_gates,
            self.context.token_embedding_leaf,
        )

    def _target_scalar(self, target: int) -> Tensor:
        if target == ROOT_TARGET:
            return self.root_scalar
        if not 0 <= target < self.feature_count:
            raise ValueError(f"feature target is outside active feature table: {target}")
        layer, position, feature = self.context.active_features[target].tolist()
        transcoder = self.context.transcoders[layer]
        preactivation = F.linear(
            self.context.mlp_inputs[layer][:, position],
            transcoder.W_enc[feature : feature + 1],
            transcoder.b_enc[feature : feature + 1],
        )
        return torch.relu(preactivation).squeeze()

    def _effects_from_grads(
        self, grads: tuple[Tensor | None, ...], *, batch_size: int
    ) -> Tensor:
        cursor = 0
        feature_parts = []
        for layer, features in enumerate(self.context.feature_tensors):
            grad = grads[cursor]
            cursor += 1
            if grad is None:
                grad = torch.zeros(
                    (batch_size, *features.shape),
                    device=features.device,
                    dtype=features.dtype,
                )
            if grad.ndim == features.ndim:
                grad = grad.unsqueeze(0)
            active = self.context.active_features
            mask = active[:, 0] == layer
            records = active[mask]
            if records.numel():
                positions = records[:, 1].long()
                ids = records[:, 2].long()
                values = features[0, positions, ids]
                feature_parts.append(grad[:, 0, positions, ids] * values[None, :])
        feature_effects = (
            torch.cat(feature_parts, dim=1)
            if feature_parts
            else torch.empty(
                (batch_size, 0), device=self.context.logits.device, dtype=torch.float32
            )
        )
        error_parts = []
        for gate in self.context.error_gates:
            grad = grads[cursor]
            cursor += 1
            if grad is None:
                grad = torch.zeros(
                    (batch_size, *gate.shape), device=gate.device, dtype=gate.dtype
                )
            if grad.ndim == gate.ndim:
                grad = grad.unsqueeze(0)
            error_parts.append(grad.reshape(batch_size, -1))
        error_effects = torch.cat(error_parts, dim=1)
        token_grad = grads[cursor]
        if token_grad is None:
            token_grad = torch.zeros(
                (batch_size, *self.context.token_embedding_leaf.shape),
                device=self.context.token_embedding_leaf.device,
            )
        if token_grad.ndim == self.context.token_embedding_leaf.ndim:
            token_grad = token_grad.unsqueeze(0)
        token_effects = (
            token_grad
            * self.context.token_embedding_leaf.detach().unsqueeze(0)
        ).sum(dim=-1).reshape(batch_size, -1)
        return torch.cat((feature_effects, error_effects, token_effects), dim=1).float()

    def _rows_batched(self, targets: list[int]) -> Tensor:
        outputs = torch.stack([self._target_scalar(target) for target in targets])
        identity = torch.eye(len(targets), device=outputs.device, dtype=outputs.dtype)
        grads = torch.autograd.grad(
            outputs,
            self._source_tensors(),
            grad_outputs=identity,
            retain_graph=True,
            allow_unused=True,
            is_grads_batched=True,
            materialize_grads=False,
        )
        return self._effects_from_grads(grads, batch_size=len(targets))

    def _rows_loop(self, targets: list[int]) -> Tensor:
        result = []
        for target in targets:
            self.loop_vjp_calls += 1
            started = time.perf_counter()
            grads = torch.autograd.grad(
                self._target_scalar(target),
                self._source_tensors(),
                retain_graph=True,
                allow_unused=True,
                materialize_grads=False,
            )
            self.loop_vjp_seconds += time.perf_counter() - started
            result.append(self._effects_from_grads(grads, batch_size=1)[0])
        return torch.stack(result)

    def rows(self, target_feature_indices: Sequence[int]) -> Tensor:
        targets = [int(item) for item in target_feature_indices]
        self.rows_requested += len(targets)
        if not targets:
            return torch.empty((0, self.source_count), dtype=torch.float32)
        if targets.count(ROOT_TARGET) > 1:
            raise ValueError("a row request may contain the root only once")
        output = []
        for start in range(0, len(targets), self.row_batch_size):
            block = targets[start : start + self.row_batch_size]
            self.max_target_block_size = max(self.max_target_block_size, len(block))
            mode = "loop"
            if self.use_batched_vjp:
                self.batched_vjp_attempts += 1
                started = time.perf_counter()
                try:
                    rows = self._rows_batched(block)
                except RuntimeError as error:
                    self.batched_vjp_seconds += time.perf_counter() - started
                    self.batched_vjp_fallbacks += 1
                    self.last_batched_vjp_error = {
                        "type": type(error).__name__,
                        "message": str(error)[:500],
                    }
                    self._emit_progress(
                        {
                            "stage": "vjp_fallback_started",
                            "target_count": len(block),
                            "error": self.last_batched_vjp_error,
                        }
                    )
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        raise
                    rows = self._rows_loop(block)
                    mode = "loop_fallback"
                    elapsed_seconds = time.perf_counter() - started
                else:
                    self.batched_vjp_seconds += time.perf_counter() - started
                    self.batched_vjp_successes += 1
                    mode = "batched"
                    elapsed_seconds = time.perf_counter() - started
            else:
                started = time.perf_counter()
                rows = self._rows_loop(block)
                elapsed_seconds = time.perf_counter() - started
            for offset, target in enumerate(block):
                if target != ROOT_TARGET:
                    rows[offset, target] = 0.0
            output.append(rows.detach().float().cpu())
            self._emit_progress(
                {
                    "stage": "vjp_block_complete",
                    "target_count": len(block),
                    "mode": mode,
                    "seconds": elapsed_seconds,
                }
            )
        result = torch.cat(output, dim=0)
        if result.shape != (len(targets), self.source_count):
            raise AssertionError("S5 direct-effect row shape changed")
        if not torch.isfinite(result).all():
            raise ValueError("S5 direct-effect rows contain non-finite values")
        return result


@torch.inference_mode()
def feature_intervention_logits(
    model: torch.nn.Module,
    transcoders: Sequence[RectangularReLUTranscoder],
    trace: CircuitTrace,
    interventions: Sequence[tuple[int, int, int, float]],
    *,
    device: torch.device,
) -> Tensor:
    override = {
        (int(layer), int(position), int(feature)): float(value)
        for layer, position, feature, value in interventions
    }
    result = decomposed_forward(
        model,
        transcoders,
        trace,
        device=device,
        requires_grad=False,
        feature_overrides=override,
    )
    return result.logits.detach()
