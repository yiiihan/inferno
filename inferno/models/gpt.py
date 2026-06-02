"""GPT / nanoGPT-style autoregressive language model.

This implementation follows the same structure as
[``inferno.models.vit``][inferno.models.VisionTransformer] and borrows
architectural choices from `nanochat <https://github.com/karpathy/nanochat>`_:

- RMSNorm with no learnable parameters
- Rotary positional embeddings (RoPE) inside attention
- QK norm after Q/K projections
- relu² activation in MLP
- No bias in linear layers
- Causal self-attention

while using inferno's BNN-compatible layers throughout.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MLP
from .. import bnn
from ..bnn import params
from ..bnn.modules.attention import _scaled_dot_product_attention_non_fused
from ._utils import _check_cov

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers (architectural choices from nanochat)
# ---------------------------------------------------------------------------


class _ReLUSquared(nn.Module):
    """relu²(x) = relu(x)^2 activation, as used in nanochat."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm with no learnable parameters (nanochat-style)."""
    return F.rms_norm(x, (x.size(-1),))


def _precompute_rotary_embeddings(
    context_length: int,
    head_dim: int,
    base: int = 10000,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables.

    Returns cos and sin of shape ``(context_length, head_dim // 2)``.
    These broadcast naturally over ``(*sample, batch, num_heads, seq_len, head_dim // 2)``.

    :param context_length: Maximum sequence length.
    :param head_dim: Per-head embedding dimension.
    :param base: RoPE base frequency.
    :param device: Target device.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    t = torch.arange(context_length, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (context_length, head_dim // 2)
    return freqs.cos(), freqs.sin()


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary positional embeddings to ``x``.

    :param x: Shape ``(..., num_heads, seq_len, head_dim)``.
    :param cos: Shape ``(seq_len, head_dim // 2)``.
    :param sin: Shape ``(seq_len, head_dim // 2)``.
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


# ---------------------------------------------------------------------------
# GPT Causal Self-Attention (RoPE + QK norm built-in)
# ---------------------------------------------------------------------------


class GPTCausalSelfAttention(bnn.MultiheadAttention):
    """Causal self-attention with RoPE and QK norm.

    Subclasses :class:`inferno.bnn.MultiheadAttention` and overrides ``forward``
    to inject RoPE and QK norm between the Q/K projections and the attention
    computation. cos/sin tables are precomputed at init and stored as buffers
    so they don't need to be threaded through the ``bnn.Sequential`` call chain.

    Always uses ``bias=False`` and ``is_causal=True``.

    :param embed_dim: Embedding dimension.
    :param num_heads: Number of attention heads.
    :param context_length: Maximum sequence length (used to precompute RoPE).
    :param dropout: Attention dropout probability.
    :param fused_attn: Use fused attention or explicit (for interpretability).
    :param parametrization: BNN parametrization.
    :param cov: Covariance structure for probabilistic layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
        fused_attn: bool = True,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: (
            params.FactorizedCovariance | dict[params.FactorizedCovariance] | None
        ) = None,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=False,
            fused_attn=fused_attn,
            parametrization=parametrization,
            cov=cov,
        )
        cos, sin = _precompute_rotary_embeddings(context_length, self.head_dim)
        self.register_buffer("rope_cos", cos)  # (context_length, head_dim // 2)
        self.register_buffer("rope_sin", sin)  # (context_length, head_dim // 2)

    def forward(
        self,
        query: Float[Tensor, "*sample batch_size seq_length embed_dim"],
        key: Float[Tensor, "*sample batch_size seq_length embed_dim"] | None,
        value: Float[Tensor, "*sample batch_size seq_length embed_dim"] | None,
        attn_mask: Float[Tensor, "batch query_token keyval_token"] | None = None,
        is_causal: bool = True,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample batch_size seq_length embed_dim"]:
        if key is None:
            key = query
        if value is None:
            value = query

        # Step 1: Q, K, V projections (BNN linear layers)
        query = self.q_proj(
            query,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        key = self.k_proj(
            key,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        value = self.v_proj(
            value,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )

        # Step 2: Split heads -> (..., num_heads, seq_len, head_dim)
        query = query.unflatten(-1, [self.num_heads, self.head_dim]).transpose(-2, -3)
        key = key.unflatten(-1, [self.num_heads, self.head_dim]).transpose(-2, -3)
        value = value.unflatten(-1, [self.num_heads, self.head_dim]).transpose(-2, -3)

        # Step 3: RoPE — cos/sin broadcast over (*sample, batch, num_heads, seq_len, d)
        seq_len = query.shape[-2]
        cos = self.rope_cos[:seq_len]  # (seq_len, head_dim // 2)
        sin = self.rope_sin[:seq_len]
        query = _apply_rotary_emb(query, cos, sin)
        key = _apply_rotary_emb(key, cos, sin)

        # Step 4: QK norm (nanochat-style)
        query = _rms_norm(query)
        key = _rms_norm(key)

        # Step 5: Scaled dot-product attention (always causal)
        if self.fused_attn:
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_output, _ = _scaled_dot_product_attention_non_fused(
                query,
                key,
                value,
                attn_mask=attn_mask,
                is_causal=True,
            )

        # Step 6: Merge heads -> (..., seq_len, embed_dim)
        attn_output = attn_output.transpose(-2, -3).flatten(-2)

        # Step 7: Output projection
        if self.out_proj is not None:
            attn_output = self.out_proj(
                attn_output,
                sample_shape=sample_shape,
                generator=generator,
                input_contains_samples=True,
                parameter_samples=parameter_samples,
            )

        return attn_output


# ---------------------------------------------------------------------------
# GPT MLP Block
# ---------------------------------------------------------------------------


class GPTMLPBlock(MLP):
    """GPT MLP block: linear -> relu² -> linear, no bias.

    Follows nanochat's choice of relu² over GELU and no bias.

    :param in_dim: Input and output dimension.
    :param mlp_dim: Intermediate (hidden) dimension.
    :param dropout: Dropout probability.
    :param parametrization: BNN parametrization.
    :param cov: Covariance structure for probabilistic layers.
    """

    def __init__(
        self,
        in_dim: int,
        mlp_dim: int,
        dropout: float,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: params.FactorizedCovariance | None = None,
    ):
        super().__init__(
            in_dim,
            [mlp_dim],
            in_dim,
            activation_layer=_ReLUSquared,
            inplace=None,
            bias=False,
            dropout=dropout,
            parametrization=parametrization,
            cov=cov,
        )


# ---------------------------------------------------------------------------
# GPT Block
# ---------------------------------------------------------------------------


class GPTBlock(bnn.BNNMixin, nn.Module):
    """GPT transformer decoder block.

    Pre-norm with RMSNorm, causal self-attention with RoPE + QK norm, and MLP
    with relu² activation. Mirrors ``EncoderBlock`` in ``vit.py``.

    :param num_heads: Number of attention heads.
    :param hidden_dim: Embedding / hidden dimension.
    :param mlp_dim: MLP intermediate dimension.
    :param context_length: Maximum sequence length (passed to attention for RoPE).
    :param dropout: Dropout probability.
    :param attention_dropout: Attention dropout probability.
    :param fused_attn: Use fused attention (faster) or explicit (for interpretability).
    :param parametrization: BNN parametrization.
    :param cov: Covariance structure for probabilistic layers.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        context_length: int,
        dropout: float,
        attention_dropout: float,
        fused_attn: bool = True,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: (
            params.FactorizedCovariance | dict[params.FactorizedCovariance] | None
        ) = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        cov = _check_cov(cov, ["self_attention", "mlp"])

        self.self_attention = GPTCausalSelfAttention(
            hidden_dim,
            num_heads,
            context_length=context_length,
            dropout=attention_dropout,
            fused_attn=fused_attn,
            parametrization=parametrization,
            cov=cov["self_attention"],
        )
        self.dropout = nn.Dropout(dropout)

        self.mlp = GPTMLPBlock(
            hidden_dim,
            mlp_dim,
            dropout,
            cov=cov["mlp"],
            parametrization=parametrization,
        )

    def forward(
        self,
        input: Float[Tensor, "*sample batch_size seq_length hidden_dim"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample batch_size seq_length hidden_dim"]:
        # Pre-norm + causal self-attention (RoPE + QK norm inside) + residual
        x = _rms_norm(input)
        x = self.self_attention(
            x,
            None,
            None,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        x = self.dropout(x)
        x = x + input  # broadcasts when input lacks sample dims (first block)

        # Pre-norm + MLP + residual
        # After self_attention, x has sample dims => input_contains_samples=True
        y = _rms_norm(x)
        y = self.mlp(
            y,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=True,
            parameter_samples=parameter_samples,
        )
        return x + y


# ---------------------------------------------------------------------------
# GPT Decoder
# ---------------------------------------------------------------------------


class GPTDecoder(bnn.BNNMixin, nn.Module):
    """GPT decoder: token embedding + transformer blocks with RoPE.

    Mirrors ``Encoder`` in ``vit.py``. The token embedding (``nn.Embedding``)
    is not BNN-wrapped since it sits at the input boundary. Positional
    information is encoded via RoPE inside each attention block — no separate
    positional embedding parameter.

    :param vocab_size: Vocabulary size.
    :param context_length: Maximum sequence length.
    :param num_layers: Number of transformer blocks.
    :param num_heads: Number of attention heads.
    :param hidden_dim: Embedding / hidden dimension.
    :param mlp_dim: MLP intermediate dimension.
    :param dropout: Dropout probability.
    :param attention_dropout: Attention dropout probability.
    :param fused_attn: Use fused or explicit attention.
    :param parametrization: BNN parametrization.
    :param cov: Covariance structure for probabilistic layers.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        fused_attn: bool = True,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: (
            params.FactorizedCovariance
            | dict[params.FactorizedCovariance]
            | dict[dict[params.FactorizedCovariance]]
            | None
        ) = None,
    ):
        super().__init__()
        cov = _check_cov(cov, [f"layers.block_{i}" for i in range(num_layers)])

        # Token embedding — not BNN-wrapped (input boundary)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks (RoPE handled inside each block's attention)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"block_{i}"] = GPTBlock(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                context_length=context_length,
                dropout=dropout,
                attention_dropout=attention_dropout,
                fused_attn=fused_attn,
                parametrization=parametrization,
                cov=cov[f"layers.block_{i}"],
            )
        self.layers = bnn.Sequential(layers)

    def reset_parameters(self) -> None:
        """Reset parameters. Needed because GPTDecoder has direct parameters."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.layers.parametrization = self.parametrization
        self.layers.reset_parameters()

    def parameters_and_lrs(
        self,
        lr: float,
        optimizer: Literal["SGD", "Adam"],
    ) -> list[dict[str, Tensor | float]]:
        """Get parameters and their learning rates.

        :param lr: Global learning rate.
        :param optimizer: Optimizer being used.
        """
        param_groups = [
            {
                "name": "token_embedding",
                "params": list(self.token_embedding.parameters()),
                "lr": lr,
            },
        ]
        param_groups += self.layers.parameters_and_lrs(lr=lr, optimizer=optimizer)
        return param_groups

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_length"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample batch_size seq_length hidden_dim"]:
        # Token ids are always (batch, seq_len) — no sample dims on the input.
        x = self.token_embedding(input)  # (batch, seq_len, hidden_dim)
        x = self.dropout(x)

        # Transformer blocks — input has no sample dims yet
        x = self.layers(
            x,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=False,
            parameter_samples=parameter_samples,
        )

        # Final RMSNorm — works on any number of leading dims
        return _rms_norm(x)


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------


class GPT(bnn.BNNMixin, nn.Module):
    """GPT autoregressive language model.

    Architecture follows nanochat's modern choices (RMSNorm, RoPE, QK norm,
    relu², no bias) built on inferno's BNN layers. For a plain deterministic
    model, leave ``cov=None`` (the default).

    The covariance can be specified as ``None`` (deterministic), a single
    :class:`~inferno.bnn.params.FactorizedCovariance` (same covariance in all
    layers), or a nested dictionary to target specific sub-modules.  For
    example, a last-layer-only probabilistic LM head:

    .. code-block:: python

        cov = params.LowRankCovariance(rank=2)
        model = GPT(
            vocab_size=50257,
            context_length=1024,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            cov={"lm_head": cov},   # decoder stays None (deterministic)
        )

    Or a fully probabilistic model with the same covariance everywhere:

    .. code-block:: python

        model = GPT(..., cov=params.DiagonalCovariance())

    Note that any modules omitted from the covariance dict default to ``None``.

    :param vocab_size: Vocabulary size (e.g. 50257 for GPT-2 tokenizer).
    :param context_length: Maximum sequence length (e.g. 1024).
    :param num_layers: Number of transformer blocks.
    :param num_heads: Number of attention heads.
    :param hidden_dim: Embedding / hidden dimension.
    :param mlp_dim: MLP intermediate dimension (typically 4 * hidden_dim).
    :param dropout: Dropout probability.
    :param attention_dropout: Attention dropout probability.
    :param fused_attn: Use fused attention (faster) or explicit (for interpretability).
    :param parametrization: BNN parametrization.
    :param cov: Covariance structure for probabilistic layers.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        fused_attn: bool = True,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: (
            params.FactorizedCovariance
            | dict[params.FactorizedCovariance]
            | dict[dict[params.FactorizedCovariance]]
            | None
        ) = None,
    ):
        super().__init__(parametrization=parametrization)
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.fused_attn = fused_attn

        cov = _check_cov(cov, ["decoder", "lm_head"])

        self.decoder = GPTDecoder(
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            fused_attn=fused_attn,
            parametrization=parametrization,
            cov=cov["decoder"],
        )

        # LM head: hidden_dim -> vocab_size, no bias (nanochat-style)
        self.lm_head = bnn.Linear(
            hidden_dim,
            vocab_size,
            bias=False,
            parametrization=parametrization,
            cov=cov["lm_head"],
            layer_type="output",
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters of the module and propagate parametrization."""
        self.decoder.parametrization = self.parametrization
        self.lm_head.parametrization = self.parametrization
        self.decoder.reset_parameters()
        self.lm_head.reset_parameters()

    def parameters_and_lrs(
        self,
        lr: float,
        optimizer: Literal["SGD", "Adam"],
    ) -> list[dict[str, Tensor | float]]:
        """Get parameters and their learning rates.

        :param lr: Global learning rate.
        :param optimizer: Optimizer being used.
        """
        param_groups = []
        param_groups += self.decoder.parameters_and_lrs(lr=lr, optimizer=optimizer)
        param_groups += self.lm_head.parameters_and_lrs(lr=lr, optimizer=optimizer)
        return param_groups

    def representation(
        self,
        input: Float[Tensor, "batch_size seq_length"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample batch_size seq_length hidden_dim"]:
        """Hidden states from the decoder (before the LM head)."""
        return self.decoder(
            input,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_length"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample batch_size seq_length vocab_size"]:
        x = self.representation(
            input,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        # After decoder, x has sample dims => input_contains_samples=True
        return self.lm_head(
            x,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=True,
            parameter_samples=parameter_samples,
        )

    @torch.inference_mode()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive text generation.

        :param tokens: Prompt token ids, shape ``(batch, seq_len)``.
        :param max_new_tokens: Number of tokens to generate.
        :param temperature: Sampling temperature. Use ``0.0`` for greedy decoding.
        :param top_k: If set, restricts sampling to the top-k logits.
        :returns: Token ids including the prompt, shape ``(batch, seq_len + max_new_tokens)``.
        """
        for _ in range(max_new_tokens):
            # Crop to context_length if the sequence has grown too long
            tokens_cond = (
                tokens
                if tokens.size(1) <= self.context_length
                else tokens[:, -self.context_length :]
            )
            logits = self(tokens_cond)       # (batch, seq_len, vocab_size)
            logits = logits[:, -1, :]        # (batch, vocab_size) — last position only

            if temperature == 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens


# ---------------------------------------------------------------------------
# Named variants (standard GPT-2 sizes)
# ---------------------------------------------------------------------------


class GPT2_Nano(GPT):
    """GPT-2 Nano (~10M transformer body): 6 layers, 6 heads, hidden_dim=384.

    Designed for small datasets like TinyStories where GPT-2 Small is
    over-parameterised. The transformer body is ~10M parameters; the
    embedding / LM-head add ~19M each for the GPT-2 vocab (50257 tokens).

    :param **kwargs: Passed to :class:`GPT`. Must include ``vocab_size`` and
        ``context_length``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            num_layers=6,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
            **kwargs,
        )


class GPT2_Small(GPT):
    """GPT-2 Small (~117M parameters): 12 layers, 12 heads, hidden_dim=768.

    :param **kwargs: Passed to :class:`GPT`. Must include ``vocab_size`` and
        ``context_length``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            **kwargs,
        )


class GPT2_Medium(GPT):
    """GPT-2 Medium (~345M parameters): 24 layers, 16 heads, hidden_dim=1024.

    :param **kwargs: Passed to :class:`GPT`. Must include ``vocab_size`` and
        ``context_length``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            **kwargs,
        )


class GPT2_Large(GPT):
    """GPT-2 Large (~762M parameters): 36 layers, 20 heads, hidden_dim=1280.

    :param **kwargs: Passed to :class:`GPT`. Must include ``vocab_size`` and
        ``context_length``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            num_layers=36,
            num_heads=20,
            hidden_dim=1280,
            mlp_dim=5120,
            **kwargs,
        )


class GPT2_XL(GPT):
    """GPT-2 XL (~1.5B parameters): 48 layers, 25 heads, hidden_dim=1600.

    :param **kwargs: Passed to :class:`GPT`. Must include ``vocab_size`` and
        ``context_length``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            num_layers=48,
            num_heads=25,
            hidden_dim=1600,
            mlp_dim=6400,
            **kwargs,
        )
