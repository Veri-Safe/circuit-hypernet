from __future__ import annotations

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from s2_hypernet.parameter_generator import TransformerModel


class SafeRawWeightTransformerModel(TransformerModel):
    """Inject raw-weight conditioning after a prompt-aware PG block."""

    def __init__(
        self,
        *args,
        raw_weight_conditioner: nn.Module,
        injection_after_block: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.raw_weight_conditioner = raw_weight_conditioner
        if injection_after_block != 1:
            raise ValueError("Safe raw-weight v2 freezes injection_after_block=1")
        if len(self.blocks) <= injection_after_block:
            raise ValueError("Safe raw-weight injection requires a later PG block")
        self.injection_after_block = int(injection_after_block)

    def _run_block(
        self,
        block: nn.Module,
        *,
        x: Tensor,
        encoder_hidden_states: Tensor,
        hw_freqs: Tensor,
        n_freqs: Tensor,
        l_freqs: Tensor,
        all_freqs: Tensor,
        condition_attention_mask: Tensor | None,
    ) -> Tensor:
        if not (self.gradient_checkpointing and self.training):
            return block(
                x=x,
                encoder_hidden_states=encoder_hidden_states,
                hw_freqs=hw_freqs,
                n_freqs=n_freqs,
                l_freqs=l_freqs,
                all_freqs=all_freqs,
                condition_attention_mask=condition_attention_mask,
            )

        if condition_attention_mask is None:

            def block_forward(
                x_arg: Tensor,
                encoder_arg: Tensor,
                hw_arg: Tensor,
                n_arg: Tensor,
                l_arg: Tensor,
                all_arg: Tensor,
            ) -> Tensor:
                return block(
                    x=x_arg,
                    encoder_hidden_states=encoder_arg,
                    hw_freqs=hw_arg,
                    n_freqs=n_arg,
                    l_freqs=l_arg,
                    all_freqs=all_arg,
                    condition_attention_mask=None,
                )

            return checkpoint(
                block_forward,
                x,
                encoder_hidden_states,
                hw_freqs,
                n_freqs,
                l_freqs,
                all_freqs,
                use_reentrant=False,
            )

        def block_forward_with_mask(
            x_arg: Tensor,
            encoder_arg: Tensor,
            hw_arg: Tensor,
            n_arg: Tensor,
            l_arg: Tensor,
            all_arg: Tensor,
            mask_arg: Tensor,
        ) -> Tensor:
            return block(
                x=x_arg,
                encoder_hidden_states=encoder_arg,
                hw_freqs=hw_arg,
                n_freqs=n_arg,
                l_freqs=l_arg,
                all_freqs=all_arg,
                condition_attention_mask=mask_arg,
            )

        return checkpoint(
            block_forward_with_mask,
            x,
            encoder_hidden_states,
            hw_freqs,
            n_freqs,
            l_freqs,
            all_freqs,
            condition_attention_mask,
            use_reentrant=False,
        )

    def forward(
        self,
        encoder_hidden_states: Tensor,
        condition_attention_mask: Tensor | None = None,
    ) -> Tensor:
        if encoder_hidden_states.dim() == 4 and encoder_hidden_states.shape[1] == 1:
            encoder_hidden_states = encoder_hidden_states.squeeze(1)
        encoder_hidden_states = self.norm_input(encoder_hidden_states)
        hw_freqs, n_freqs, l_freqs, all_freqs = self._get_pos_embed(
            device=encoder_hidden_states.device,
            cond_seq_length=encoder_hidden_states.shape[1],
        )
        x = self.get_tokens(
            batch_size=encoder_hidden_states.shape[0],
            device=encoder_hidden_states.device,
        )
        x = self.apply_condition_summary(
            x=x,
            encoder_hidden_states=encoder_hidden_states,
            condition_attention_mask=condition_attention_mask,
        )
        for block_index, block in enumerate(self.blocks, start=1):
            x = self._run_block(
                block,
                x=x,
                encoder_hidden_states=encoder_hidden_states,
                hw_freqs=hw_freqs,
                n_freqs=n_freqs,
                l_freqs=l_freqs,
                all_freqs=all_freqs,
                condition_attention_mask=condition_attention_mask,
            )
            if block_index == self.injection_after_block:
                x = self.raw_weight_conditioner(x)
        return self.post_process(x)
