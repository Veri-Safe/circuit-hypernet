from __future__ import annotations

from collections import OrderedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel

from .config import ParameterGeneratorConfig
from .lora_tokenizer import Tokenizer2DBatchedLoRA


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    if dim % 2 != 0:
        raise ValueError("RoPE dim must be even.")
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos = torch.cos(out)
    sin = torch.sin(out)
    return torch.stack((cos, -sin, sin, cos), dim=-1).reshape(*out.shape, 2, 2)


def apply_rope(xq: Tensor, xk: Tensor, q_freqs: Tensor, k_freqs: Tensor | None = None) -> tuple[Tensor, Tensor]:
    if k_freqs is None:
        k_freqs = q_freqs
    q_freqs = q_freqs.to(device=xq.device, dtype=xq.dtype)
    k_freqs = k_freqs.to(device=xk.device, dtype=xk.dtype)
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = q_freqs[..., 0] * xq_[..., 0] + q_freqs[..., 1] * xq_[..., 1]
    xk_out = k_freqs[..., 0] * xk_[..., 0] + k_freqs[..., 1] * xk_[..., 1]
    return xq_out.reshape_as(xq), xk_out.reshape_as(xk)


class PosEmbedND(nn.Module):
    def __init__(self, dim: int, theta: int = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(
        self,
        ids: Tensor | None = None,
        axes_dim: list[int] | None = None,
        axes_lengths: list[int] | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        if axes_dim is None:
            raise ValueError("axes_dim is required.")
        if sum(axes_dim) != self.dim:
            raise ValueError(f"sum(axes_dim)={sum(axes_dim)} must equal dim={self.dim}.")
        if ids is None:
            if axes_lengths is None:
                raise ValueError("axes_lengths is required when ids is None.")
            ids = self._create_ids(axes_lengths=axes_lengths, device=device)

        parts = [rope(ids[..., idx], axes_dim[idx], self.theta) for idx in range(len(axes_dim))]
        return torch.cat(parts, dim=-3).unsqueeze(1)

    def _create_ids(self, axes_lengths: list[int], device: torch.device | None = None) -> Tensor:
        ranges = [torch.arange(length, device=device) for length in axes_lengths]
        grids = torch.meshgrid(*ranges, indexing="ij")
        ids = torch.stack([grid.flatten() for grid in grids], dim=-1)
        return ids.unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _project_qkv(self, q_input: Tensor, kv_input: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        if kv_input is None:
            kv_input = q_input
        batch_size, q_len, _ = q_input.shape
        kv_len = kv_input.shape[1]
        q = self.w_q(q_input).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(kv_input).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(kv_input).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(self, x: Tensor, freqs: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        batch_size, seq_length, _ = x.shape
        q, k, v = self._project_qkv(x)
        q, k = apply_rope(q, k, freqs)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch_size, seq_length, self.d_model)
        return self.w_o(out)


class MultiHeadCrossAttention(MultiHeadAttention):
    def forward(
        self,
        q_input: Tensor,
        kv_input: Tensor,
        q_freqs: Tensor,
        kv_freqs: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, q_len, _ = q_input.shape
        q, k, v = self._project_qkv(q_input, kv_input)
        q, k = apply_rope(q, k, q_freqs, kv_freqs)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch_size, q_len, self.d_model)
        return self.w_o(out)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.self_attn_n = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.self_attn_hw = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.cross_attn = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_model * 4)
        self.norm1 = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.norm2 = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.norm3 = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.norm4 = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)

    def forward(
        self,
        x: Tensor,
        encoder_hidden_states: Tensor,
        hw_freqs: Tensor,
        n_freqs: Tensor,
        l_freqs: Tensor,
        all_freqs: Tensor,
        condition_attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, seq_n, seq_hw, channels = x.shape

        norm_x = self.norm1(x)
        norm_x = norm_x.permute(0, 2, 1, 3).reshape(batch_size * seq_hw, seq_n, channels)
        out = self.self_attn_n(norm_x, freqs=n_freqs)
        out = out.reshape(batch_size, seq_hw, seq_n, channels).permute(0, 2, 1, 3)
        x = x + out

        norm_x = self.norm2(x)
        norm_x = norm_x.reshape(batch_size * seq_n, seq_hw, channels)
        out = self.self_attn_hw(norm_x, freqs=hw_freqs)
        out = out.reshape(batch_size, seq_n, seq_hw, channels)
        x = x + out

        norm_x = self.norm3(x)
        norm_x = norm_x.reshape(batch_size, seq_n * seq_hw, channels)
        cross_mask = None
        if condition_attention_mask is not None:
            cross_mask = condition_attention_mask.to(dtype=torch.bool, device=norm_x.device)[:, None, None, :]
        out = self.cross_attn(
            norm_x,
            encoder_hidden_states,
            q_freqs=all_freqs,
            kv_freqs=l_freqs,
            attn_mask=cross_mask,
        )
        out = out.reshape(batch_size, seq_n, seq_hw, channels)
        x = x + out

        x = x + self.feed_forward(self.norm4(x))
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_base_model_layers: int,
        num_token_per_layer: int,
        lora_rank: int,
        dim_accumulation: int,
        output_dim: int,
        head_dim: int,
        num_blocks: int,
        lora_A_token_count: int,
        lora_B_token_count: int,
        condition_injection_mode: str = "none",
        condition_summary_scale_init: float = 0.01,
    ):
        super().__init__()
        if lora_rank % dim_accumulation != 0:
            raise ValueError("lora_rank must be divisible by dim_accumulation.")
        if d_model % dim_accumulation != 0:
            raise ValueError("d_model must be divisible by dim_accumulation.")
        if condition_injection_mode not in {"none", "additive", "film"}:
            raise ValueError("condition_injection_mode must be one of: none, additive, film.")

        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = d_model // head_dim
        self.dim_accumulation = dim_accumulation
        self.dim_wo_accumulation = d_model // dim_accumulation
        self.seq_n = num_base_model_layers
        self.seq_h = num_token_per_layer
        self.seq_w = lora_rank
        self.output_dim = output_dim
        self.lora_A_token_count = lora_A_token_count
        self.lora_B_token_count = lora_B_token_count
        self.condition_injection_mode = condition_injection_mode

        self.pos_embed = PosEmbedND(dim=head_dim, theta=10000)
        self.norm_input = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.norm_tokens = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.hw_pos = nn.Embedding(num_token_per_layer * lora_rank, self.dim_wo_accumulation)
        self.layer_pos = nn.Embedding(num_base_model_layers, self.dim_wo_accumulation)
        if condition_injection_mode == "additive":
            self.condition_summary_proj = nn.Linear(d_model, d_model)
            self.condition_summary_scale = nn.Parameter(torch.tensor(float(condition_summary_scale_init)))
        elif condition_injection_mode == "film":
            self.condition_summary_film = nn.Linear(d_model, d_model * 2)
            self.reset_condition_injection()
        self.blocks = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=self.num_heads) for _ in range(num_blocks)])
        self.norm_final = nn.RMSNorm(d_model, elementwise_affine=True, eps=1e-6)
        self.proj_out_A = nn.Linear(self.dim_wo_accumulation, output_dim)
        self.proj_out_B = nn.Linear(self.dim_wo_accumulation, output_dim)
        self.gradient_checkpointing = False

    def reset_condition_injection(self) -> None:
        if self.condition_injection_mode == "film":
            nn.init.zeros_(self.condition_summary_film.weight)
            nn.init.zeros_(self.condition_summary_film.bias)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    @torch.no_grad()
    def _get_pos_embed(self, device: torch.device, cond_seq_length: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rank_after_accum = self.seq_w // self.dim_accumulation
        hw_freqs = self.pos_embed(
            axes_dim=[self.head_dim // 2, self.head_dim // 2],
            axes_lengths=[self.seq_h, rank_after_accum],
            device=device,
        )
        n_freqs = self.pos_embed(
            axes_dim=[self.head_dim],
            axes_lengths=[self.seq_n],
            device=device,
        )
        all_freqs = self.pos_embed(
            axes_dim=[self.head_dim // 2, self.head_dim // 2],
            axes_lengths=[self.seq_n, self.seq_h * rank_after_accum],
            device=device,
        )
        diag_ids = torch.arange(cond_seq_length, device=device)
        diag_ids = torch.stack([diag_ids, diag_ids], dim=-1).unsqueeze(0)
        diag_ids = diag_ids + max(self.seq_h, rank_after_accum)
        l_freqs = self.pos_embed(
            ids=diag_ids,
            axes_dim=[self.head_dim // 2, self.head_dim // 2],
            device=device,
        )
        return hw_freqs, n_freqs, l_freqs, all_freqs

    def get_tokens(self, batch_size: int, device: torch.device) -> Tensor:
        hw = self.hw_pos.weight.to(device)
        layer = self.layer_pos.weight.to(device)
        tokens = hw[None, None, :, :] + layer[None, :, None, :]
        tokens = tokens.expand(batch_size, -1, -1, -1)
        tokens = tokens.reshape(
            batch_size,
            self.seq_n,
            self.seq_h,
            self.seq_w // self.dim_accumulation,
            self.dim_accumulation,
            self.dim_wo_accumulation,
        )
        tokens = tokens.permute(0, 1, 2, 3, 4, 5).reshape(
            batch_size,
            self.seq_n,
            self.seq_h * (self.seq_w // self.dim_accumulation),
            self.d_model,
        )
        return self.norm_tokens(tokens)

    def condition_summary(
        self,
        encoder_hidden_states: Tensor,
        condition_attention_mask: Tensor | None = None,
    ) -> Tensor:
        if condition_attention_mask is None:
            return encoder_hidden_states.mean(dim=1)

        mask = condition_attention_mask.to(
            device=encoder_hidden_states.device,
            dtype=encoder_hidden_states.dtype,
        ).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (encoder_hidden_states * mask).sum(dim=1) / denom

    def apply_condition_summary(
        self,
        x: Tensor,
        encoder_hidden_states: Tensor,
        condition_attention_mask: Tensor | None = None,
    ) -> Tensor:
        if self.condition_injection_mode == "none":
            return x

        summary = self.condition_summary(
            encoder_hidden_states=encoder_hidden_states,
            condition_attention_mask=condition_attention_mask,
        )
        if self.condition_injection_mode == "additive":
            delta = self.condition_summary_proj(summary).to(dtype=x.dtype)
            scale = self.condition_summary_scale.to(device=x.device, dtype=x.dtype)
            return x + scale * delta[:, None, None, :]

        if self.condition_injection_mode == "film":
            gamma, beta = self.condition_summary_film(summary).to(dtype=x.dtype).chunk(2, dim=-1)
            return x * (1.0 + gamma[:, None, None, :]) + beta[:, None, None, :]

        raise RuntimeError(f"Unknown condition_injection_mode={self.condition_injection_mode}")

    def post_process(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x = self.norm_final(x)
        x = x.reshape(
            batch_size,
            self.seq_n,
            self.seq_h,
            self.seq_w // self.dim_accumulation,
            self.dim_accumulation,
            self.dim_wo_accumulation,
        )
        x = x.reshape(batch_size, self.seq_n, self.seq_h, self.seq_w, self.dim_wo_accumulation)
        x_A, x_B = torch.split(x, [self.lora_A_token_count, self.lora_B_token_count], dim=2)
        x_A = self.proj_out_A(x_A)
        x_B = self.proj_out_B(x_B)
        return torch.cat([x_A, x_B], dim=2)

    def forward(self, encoder_hidden_states: Tensor, condition_attention_mask: Tensor | None = None) -> Tensor:
        if encoder_hidden_states.dim() == 4 and encoder_hidden_states.shape[1] == 1:
            encoder_hidden_states = encoder_hidden_states.squeeze(1)
        encoder_hidden_states = self.norm_input(encoder_hidden_states)
        hw_freqs, n_freqs, l_freqs, all_freqs = self._get_pos_embed(
            device=encoder_hidden_states.device,
            cond_seq_length=encoder_hidden_states.shape[1],
        )
        x = self.get_tokens(batch_size=encoder_hidden_states.shape[0], device=encoder_hidden_states.device)
        x = self.apply_condition_summary(
            x=x,
            encoder_hidden_states=encoder_hidden_states,
            condition_attention_mask=condition_attention_mask,
        )
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                if condition_attention_mask is None:
                    def block_forward(
                        x_arg: Tensor,
                        encoder_arg: Tensor,
                        hw_arg: Tensor,
                        n_arg: Tensor,
                        l_arg: Tensor,
                        all_arg: Tensor,
                        block_module: TransformerBlock = block,
                    ) -> Tensor:
                        return block_module(
                            x=x_arg,
                            encoder_hidden_states=encoder_arg,
                            hw_freqs=hw_arg,
                            n_freqs=n_arg,
                            l_freqs=l_arg,
                            all_freqs=all_arg,
                            condition_attention_mask=None,
                        )

                    x = checkpoint(
                        block_forward,
                        x,
                        encoder_hidden_states,
                        hw_freqs,
                        n_freqs,
                        l_freqs,
                        all_freqs,
                        use_reentrant=False,
                    )
                else:
                    def block_forward(
                        x_arg: Tensor,
                        encoder_arg: Tensor,
                        hw_arg: Tensor,
                        n_arg: Tensor,
                        l_arg: Tensor,
                        all_arg: Tensor,
                        mask_arg: Tensor,
                        block_module: TransformerBlock = block,
                    ) -> Tensor:
                        return block_module(
                            x=x_arg,
                            encoder_hidden_states=encoder_arg,
                            hw_freqs=hw_arg,
                            n_freqs=n_arg,
                            l_freqs=l_arg,
                            all_freqs=all_arg,
                            condition_attention_mask=mask_arg,
                        )

                    x = checkpoint(
                        block_forward,
                        x,
                        encoder_hidden_states,
                        hw_freqs,
                        n_freqs,
                        l_freqs,
                        all_freqs,
                        condition_attention_mask,
                        use_reentrant=False,
                    )
            else:
                x = block(
                    x=x,
                    encoder_hidden_states=encoder_hidden_states,
                    hw_freqs=hw_freqs,
                    n_freqs=n_freqs,
                    l_freqs=l_freqs,
                    all_freqs=all_freqs,
                    condition_attention_mask=condition_attention_mask,
                )
        return self.post_process(x)


class ParameterGenerator(PreTrainedModel):
    config_class = ParameterGeneratorConfig
    _no_split_modules = ["TransformerBlock"]

    def __init__(self, config: ParameterGeneratorConfig):
        super().__init__(config)
        self.tokenizer = Tokenizer2DBatchedLoRA(
            token_dim=config.token_dim,
            rank=config.rank,
            alpha=config.alpha,
            pg_mapping=config.pg_mapping,
        )
        self.lora_A_token_count = self.tokenizer.lora_A_token_count
        self.lora_B_token_count = self.tokenizer.lora_B_token_count
        self.model = TransformerModel(
            d_model=config.d_model,
            num_base_model_layers=config.num_base_model_layers,
            num_token_per_layer=self.lora_A_token_count + self.lora_B_token_count,
            lora_rank=config.rank,
            output_dim=config.output_dim,
            head_dim=config.head_dim,
            num_blocks=config.num_pg_layers,
            dim_accumulation=config.dim_accumulation,
            lora_A_token_count=self.lora_A_token_count,
            lora_B_token_count=self.lora_B_token_count,
            condition_injection_mode=config.condition_injection_mode,
            condition_summary_scale_init=config.condition_summary_scale_init,
        )
        self.layer_num = config.num_base_model_layers
        self.target_layer_indices = list(config.target_layer_indices)
        self.prefix = config.prefix
        self.hidden_in = nn.Linear(config.input_dim, config.d_model)
        self.post_init()
        self.model.reset_condition_injection()
        self._init_near_zero_residual()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        self.model.disable_gradient_checkpointing()

    def _init_near_zero_residual(self) -> None:
        std = float(self.config.near_zero_b_init_std)
        if std <= 0:
            nn.init.zeros_(self.model.proj_out_B.weight)
        else:
            nn.init.normal_(self.model.proj_out_B.weight, mean=0.0, std=std)
        nn.init.zeros_(self.model.proj_out_B.bias)

    def get_lora_count(self) -> int:
        return (
            self.config.output_dim
            * (self.lora_A_token_count + self.lora_B_token_count)
            * self.config.num_base_model_layers
            * self.config.rank
        )

    def forward(
        self,
        condition: Tensor,
        condition_attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        embeddings = self.hidden_in(condition)
        output = self.model(
            encoder_hidden_states=embeddings,
            condition_attention_mask=condition_attention_mask,
        )

        all_layer_state_dict: dict[str, Tensor] = {}
        shape_state_dict = self.tokenizer.shape_state_dict
        for seq_idx, layer_idx in enumerate(self.target_layer_indices):
            layer_output = output[:, seq_idx]
            layer_state_dict: OrderedDict[str, Tensor] = self.tokenizer.detokenize(
                shape_state_dict,
                layer_output,
            )
            for key, value in layer_state_dict.items():
                all_layer_state_dict[f"{self.prefix}{layer_idx}.{key}"] = value

        all_layer_state_dict["scale"] = torch.tensor(
            self.tokenizer.lora_scale,
            device=output.device,
            dtype=output.dtype,
        )
        return all_layer_state_dict
