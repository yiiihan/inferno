"""Vision Transformers.

This implementation largely follows
[``torchvision.models.vision_transformer``](https://github.com/pytorch/vision/blob/1e53952f57462e4c28103835cf1f9e504dbea84b/torchvision/models/vision_transformer.py#L536).
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import TYPE_CHECKING, Callable, Iterator, Literal, NamedTuple

import torch
import torch.nn as nn
import torchvision
from torchvision.utils import _log_api_usage_once

from . import MLP
from .. import bnn
from ..bnn import params
from ..bnn.modules.bnn_mixin import (
    named_parameter_groups_of_torch_module,
    parameters_and_lrs_of_torch_module,
    reset_parameters_of_torch_module,
)
from ._utils import _check_cov

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

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
            activation_layer=nn.GELU,
            inplace=None,
            dropout=dropout,
            parametrization=parametrization,
            cov=cov,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(bnn.BNNMixin, nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        fused_attn: bool = True,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: (
            params.FactorizedCovariance | dict[params.FactorizedCovariance] | None
        ) = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        cov = _check_cov(cov, ["self_attention", "mlp"])

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = bnn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=attention_dropout,
            fused_attn=fused_attn,
            parametrization=parametrization,
            cov=cov["self_attention"],
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)

        self.mlp = MLPBlock(
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
    ) -> Float[Tensor, "*sample *batch *out_feature"]:
        x = self.ln_1(input)
        x = self.self_attention(
            x,
            x,
            x,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(
            y,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=True,
            parameter_samples=parameter_samples,
        )
        return x + y


class Encoder(bnn.BNNMixin, nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        fused_attn: bool = True,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        pre_layernorm: bool = False,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: (
            params.FactorizedCovariance
            | dict[params.FactorizedCovariance]
            | dict[dict[params.FactorizedCovariance]]
            | None
        ) = None,
    ):
        super().__init__()
        cov = _check_cov(cov, [f"layers.encoder_layer_{i}" for i in range(num_layers)])

        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(
            torch.empty(1, seq_length, hidden_dim).normal_(std=0.02)
        )  # from BERT
        self.pre_ln = norm_layer(hidden_dim) if pre_layernorm else None
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                fused_attn=fused_attn,
                norm_layer=norm_layer,
                parametrization=parametrization,
                cov=cov[f"layers.encoder_layer_{i}"],
            )
        self.layers = bnn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def reset_parameters(self) -> None:
        """Reset the parameters of the module and set the parametrization of all children
        to the parametrization of the module.

        Needs to be implemented because Encoder has direct parameters.
        """

        # direct parameters
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)  # from BERT

        # child modules
        self.layers.parametrization = self.parametrization
        self.layers.reset_parameters()

        reset_parameters_of_torch_module(self.ln, parametrization=self.parametrization)

        if self.pre_ln is not None:
            reset_parameters_of_torch_module(self.pre_ln, parametrization=self.parametrization)

    def named_parameter_groups(
        self,
        optimizer: Literal["SGD", "Adam", "NGD"],
        lr: float | None = None,
        prefix: str = "",
    ) -> Iterator[tuple[str, dict[str, Tensor | float]]]:
        """Return the parameters of a module sorted into named groups.

        Needs to be implemented because Encoder has direct parameters.

        :param optimizer: The optimizer for which to return parameter groups.
        :param lr: The global learning rate.
        :param prefix: Prefix to add to the names of the parameter groups.
        """
        prefix = prefix + "." if prefix != "" else prefix

        # direct parameters
        if optimizer in ("SGD", "Adam"):
            yield prefix + "pos_embedding", {
                "param_names": [prefix + "pos_embedding"],
                "module": prefix[:-1] if prefix.endswith(".") else prefix,
                "params": [self.pos_embedding],
                "lr": lr,
            }
        elif optimizer == "NGD":
            yield prefix + "pos_embedding", {
                "param_names": [prefix + "pos_embedding"],
                "module": prefix[:-1] if prefix.endswith(".") else prefix,
                "params": [self.pos_embedding],
            }

        # child modules
        yield from self.layers.named_parameter_groups(
            optimizer=optimizer, lr=lr, prefix=prefix + "layers"
        )
        yield from named_parameter_groups_of_torch_module(
            self.ln,
            optimizer=optimizer,
            lr=lr,
            parametrization=self.parametrization,
            prefix=prefix + "ln",
        )

        if self.pre_ln is not None:
            yield from named_parameter_groups_of_torch_module(
                self.pre_ln,
                optimizer=optimizer,
                lr=lr,
                parametrization=self.parametrization,
                prefix=prefix + "pre_ln",
            )

    def parameters_and_lrs(
        self,
        lr: float,
        optimizer: Literal["SGD", "Adam", "NGD"] = "SGD",
    ) -> list[dict[str, Tensor | float]]:
        return list(self.parameter_groups(optimizer=optimizer, lr=lr))

    def forward(
        self,
        input: Float[Tensor, "*sample batch_size seq_length hidden_dim"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch *out_feature"]:
        num_sample_dims = 0 if sample_shape is None else len(sample_shape)

        if sample_shape is not None:
            input = input + self.pos_embedding.expand(
                *sample_shape, *self.pos_embedding.shape
            )
        else:
            input = input + self.pos_embedding

        if self.pre_ln is not None:
            input = bnn.batched_forward(self.pre_ln, num_batch_dims=num_sample_dims + 1)(input)

        output = self.dropout(input)
        output = self.layers(
            output,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        output = bnn.batched_forward(self.ln, num_batch_dims=num_sample_dims + 1)(
            output
        )
        return output


class VisionTransformer(bnn.BNNMixin, nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929.

    The covariance can be specified as ``None`` (resulting in a non-stochastic model),
    as an instance of [``inferno.bnn.params.FactorizedCovariance``][] (resulting in the same covariance in all layers),
    or as a nested dictionary with the keys indicating the module. For example, the following will place
    a low rank covariance in the ``conv_proj``, the last layer of the encoder, and the output head:
    ```
    cov = params.LowRankCovariance(rank=2)
    last_layer_cov = {
        "conv_proj": copy.deepcopy(cov),
        "encoder": {
            "layers.encoder_layer_1": {
                "self_attention": {
                    "q": copy.deepcopy(cov),
                    "k": copy.deepcopy(cov),
                    "v": copy.deepcopy(cov),
                    "out": copy.deepcopy(cov),
                },
                "mlp": copy.deepcopy(cov)
            },
        },
        "fc": copy.deepcopy(cov),
    }
    model = VisionTransformer(
        in_size=32,
        patch_size=16,
        num_layers=2,
        num_heads=2,
        hidden_dim=10,
        mlp_dim=10,
        cov=last_layer_cov,
    )
    ```
    Note that any modules omitted from the covariance specification will default to ``None``
    (in this example, any modules part of ``last_layer_cov["encoder"]["layers.encoder_layer_0"]``).

    :param in_size: Size of the input (i.e. image size).
    :param patch_size: Size of the patch.
    :param num_layers: Number of layers in the encoder.
    :param num_heads: Number of heads.
    :param hidden_dim: Hidden size in encoder.
    :param mlp_dim: Dimension of MLP block.
    :param dropout: Dropout probability.
    :param attention_dropout: Attention dropout probability.
    :param out_size: Size of the output (i.e. number of classes).
    :param representation_size: Size of pre-logits layer before output head.
    :param norm_layer: Normalization layer to use.
    :param fused_attn: Should fused attention be used, which is more efficient, or should attention weights
        be computed explicitly (e.g. for interpretability).
    :param conv_stem_configs: Currently not supported.
    :param parametrization: The parametrization to use. Defines the initialization
        and learning rate scaling for the parameters of the module.
    :param cov: Covariance structure of the probabilistic layers.
    """

    def __init__(
        self,
        in_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        out_size: int = 1000,
        representation_size: int | None = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        fused_attn: bool = True,
        conv_stem_configs: list[NamedTuple] | None = None,
        pre_layernorm: bool = False,
        normalize_representation: bool = False,
        parametrization: params.Parametrization = params.MaximalUpdate(),
        cov: (
            params.FactorizedCovariance
            | dict[params.FactorizedCovariance]
            | dict[dict[params.FactorizedCovariance]]
            | None
        ) = None,
    ):
        super().__init__(parametrization=parametrization)
        _log_api_usage_once(self)
        torch._assert(
            in_size % patch_size == 0, "Input shape indivisible by patch size!"
        )
        self.in_size = in_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.out_size = out_size
        self.representation_size = representation_size
        if norm_layer is nn.BatchNorm2d:
            raise ValueError(
                "BatchNorm is currently not supported due to incompatibility of "
                "torch.vmap with the 'running_stats' tracked by BatchNorm."
                "See also: https://pytorch.org/docs/stable/func.batch_norm.html#patching-batch-norm."
            )
        self.norm_layer = norm_layer
        self.normalize_representation = normalize_representation
        cov = _check_cov(cov, ["conv_proj", "encoder", "pre_head.pre_logits", "fc"])
        self.fused_attn = fused_attn

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            raise NotImplementedError(
                "conv_stem_configs currently not supported in inferno "
                "because there is no Conv2dNormActivation implementation."
            )
        else:
            self.conv_proj = bnn.Conv2d(
                in_channels=3,
                out_channels=hidden_dim,
                kernel_size=patch_size,
                stride=patch_size,
                cov=cov["conv_proj"],
                parametrization=parametrization,
                layer_type="input",
            )

        seq_length = (in_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
            fused_attn=self.fused_attn,
            pre_layernorm=pre_layernorm,
            parametrization=parametrization,
            cov=cov["encoder"],
        )
        self.seq_length = seq_length

        if self.representation_size is None:
            self.pre_head = None
        else:
            pre_head_layers: OrderedDict[str, nn.Module] = OrderedDict()
            pre_head_layers["pre_logits"] = bnn.Linear(
                hidden_dim,
                representation_size,
                parametrization=parametrization,
                cov=cov["pre_head.pre_logits"],
                layer_type="hidden",
            )
            pre_head_layers["act"] = nn.Tanh()
            self.pre_head = bnn.Sequential(pre_head_layers)

        self.fc = bnn.Linear(
            (
                self.representation_size
                if self.representation_size is not None
                else hidden_dim
            ),
            out_size,
            parametrization=parametrization,
            cov=cov["fc"],
            layer_type="output",
        )

        # Reset parameters (note this replaces torchvision initialization)
        self.reset_parameters()

    @classmethod
    def from_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        weights: torchvision.models.Weights,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        """Load a VisionTransformer model with pretrained weights.

        Depending on the ``in_size`` and ``out_size`` parameters, the first and last
        layers of the model are not initialized with the pretrained weights.

        :param in_size: Size of the input (i.e. image size).
        :param out_size: Size of the output (i.e. number of classes).
        :param weights: Pretrained weights to use.
        :param freeze: Whether to freeze the pretrained weights.
        """
        # Load and preprocess the pretrained weights
        pretrained_weights = weights.get_state_dict(progress=True)
        if in_size != 224:
            # Remove the first layer (conv_proj) from the pretrained weights
            del pretrained_weights["conv_proj.weight"]
            del pretrained_weights["conv_proj.bias"]

            # Remove the positional embeddings
            del pretrained_weights["encoder.pos_embedding"]

        if out_size != pretrained_weights["heads.head.weight"].shape[0]:
            # Remove the last layer (head) from the pretrained weights
            del pretrained_weights["heads.head.weight"]
            del pretrained_weights["heads.head.bias"]

        # Model
        model = cls(
            *args,
            **kwargs,
            in_size=in_size,
            out_size=out_size,
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            pretrained_weights, strict=False
        )

        if freeze:
            # Freeze the pretrained weights
            for name, param in model.named_parameters():
                if name.replace(".params", "") in pretrained_weights:
                    param.requires_grad = False

        return model

    @classmethod
    def from_clip_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        clip_model_name: str,
        freeze: bool = False,
        pre_layernorm: bool = True,
        *args,
        **kwargs,
    ):
        """Load a VisionTransformer model with CLIP pretrained weights from HuggingFace.

        The fc (classification head) is always randomly initialised since CLIP
        has no classification head.  All other layers are loaded from CLIP's
        vision encoder.

        :param in_size: Size of the input (i.e. image size).
        :param out_size: Number of output classes.
        :param clip_model_name: HuggingFace model identifier, e.g.
            ``"openai/clip-vit-base-patch32"``.
        :param freeze: Whether to freeze the loaded weights.
        """
        import re
        try:
            from transformers import CLIPModel
        except ImportError:
            raise ImportError(
                "The `transformers` package is required to load CLIP weights. "
                "Install it with: pip install transformers"
            )

        clip_model = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)
        clip_sd = clip_model.vision_model.state_dict()

        # Derive patch_size from CLIP's patch embedding weight: [out_ch, in_ch, kH, kW]
        clip_patch_size = clip_sd["embeddings.patch_embedding.weight"].shape[-1]

        # Map CLIP vision encoder keys to inferno ViT keys.
        inferno_sd = {}
        for clip_key, value in clip_sd.items():
            if clip_key == "embeddings.class_embedding":
                inferno_sd["class_token"] = value.reshape(1, 1, -1)
            elif clip_key == "embeddings.patch_embedding.weight":
                inferno_sd["conv_proj.weight"] = value
            elif clip_key == "embeddings.position_embedding.weight":
                # CLIP: [seq, hidden] → inferno: [1, seq, hidden]
                # Interpolate spatial positions if target resolution differs from CLIP.
                clip_seq, hidden = value.shape
                target_seq = (in_size // clip_patch_size) ** 2 + 1
                if clip_seq != target_seq:
                    cls_pos = value[:1]  # [1, hidden]
                    spatial = value[1:]  # [clip_seq-1, hidden]
                    clip_grid = int(round((clip_seq - 1) ** 0.5))
                    target_grid = int(round((target_seq - 1) ** 0.5))
                    spatial = spatial.reshape(1, clip_grid, clip_grid, hidden).permute(0, 3, 1, 2)
                    spatial = torch.nn.functional.interpolate(
                        spatial.float(), size=(target_grid, target_grid), mode="bicubic", align_corners=False
                    ).to(value.dtype)
                    spatial = spatial.permute(0, 2, 3, 1).reshape(target_seq - 1, hidden)
                    value = torch.cat([cls_pos, spatial], dim=0)
                inferno_sd["encoder.pos_embedding"] = value.unsqueeze(0)
            elif clip_key in ("pre_layrnorm.weight", "pre_layrnorm.bias"):
                suffix = clip_key.split(".")[-1]
                inferno_sd[f"encoder.pre_ln.{suffix}"] = value
            elif clip_key in ("post_layernorm.weight", "post_layernorm.bias"):
                suffix = clip_key.split(".")[-1]
                inferno_sd[f"encoder.ln.{suffix}"] = value
            else:
                m = re.match(r"encoder\.layers\.(\d+)\.(.+)", clip_key)
                if m:
                    i, rest = m.group(1), m.group(2)
                    if rest.startswith("self_attn."):
                        attn_rest = rest[len("self_attn."):]
                        inferno_sd[
                            f"encoder.layers.encoder_layer_{i}.self_attention.{attn_rest}"
                        ] = value
                    elif rest.startswith("layer_norm1."):
                        inferno_sd[
                            f"encoder.layers.encoder_layer_{i}.ln_1.{rest.split('.')[-1]}"
                        ] = value
                    elif rest.startswith("layer_norm2."):
                        inferno_sd[
                            f"encoder.layers.encoder_layer_{i}.ln_2.{rest.split('.')[-1]}"
                        ] = value
                    elif rest == "mlp.fc1.weight":
                        inferno_sd[f"encoder.layers.encoder_layer_{i}.mlp.0.weight"] = value
                    elif rest == "mlp.fc1.bias":
                        inferno_sd[f"encoder.layers.encoder_layer_{i}.mlp.0.bias"] = value
                    elif rest == "mlp.fc2.weight":
                        inferno_sd[f"encoder.layers.encoder_layer_{i}.mlp.3.weight"] = value
                    elif rest == "mlp.fc2.bias":
                        inferno_sd[f"encoder.layers.encoder_layer_{i}.mlp.3.bias"] = value

        model = cls(*args, **kwargs, in_size=in_size, out_size=out_size, pre_layernorm=pre_layernorm)
        model.load_state_dict(inferno_sd, strict=False)

        # CLIP patch_embedding has no bias; zero-init to avoid a random offset
        if model.conv_proj.params.bias is not None:
            nn.init.zeros_(model.conv_proj.params.bias)

        if freeze:
            for name, param in model.named_parameters():
                if name.replace(".params", "") in inferno_sd:
                    param.requires_grad = False

        return model

    def load_state_dict(
        self, state_dict: dict, strict: bool = True, assign: bool = False
    ):
        # Ensure correct key mapping for heads if torchvision state_dict is loaded.
        torchvision_to_inferno_key_map = {
            "heads.pre_logits.weight": "pre_head.pre_logits.weight",
            "heads.pre_logits.bias": "pre_head.pre_logits.bias",
            "heads.head.weight": "fc.weight",
            "heads.head.bias": "fc.bias",
        }
        state_dict = {
            torchvision_to_inferno_key_map.get(k, k): v for k, v in state_dict.items()
        }

        return super().load_state_dict(state_dict, strict, assign)

    def reset_parameters(self) -> None:
        """Reset the parameters of the module and set the parametrization of all children
        to the parametrization of the module.

        Needs to be implemented because VisionTransformer has direct parameters.
        """

        # Direct parameters
        nn.init.zeros_(self.class_token)

        # Child modules
        self.conv_proj.parametrization = self.parametrization
        self.encoder.parametrization = self.parametrization
        if self.pre_head is not None:
            self.pre_head.parametrization = self.parametrization
        self.fc.parametrization = self.parametrization

        self.conv_proj.reset_parameters()
        self.encoder.reset_parameters()
        if self.pre_head is not None:
            self.pre_head.reset_parameters()
        self.fc.reset_parameters()

    def named_parameter_groups(
        self,
        optimizer: Literal["SGD", "Adam", "NGD"],
        lr: float | None = None,
        prefix: str = "",
    ) -> Iterator[tuple[str, dict[str, Tensor | float]]]:
        """Return the parameters of a module sorted into named groups.

        Needs to be implemented because VisionTransformer has direct parameters.

        :param optimizer: The optimizer for which to return parameter groups.
        :param lr: The global learning rate.
        :param prefix: Prefix to add to the names of the parameter groups.
        """
        prefix = prefix + "." if prefix != "" else prefix

        # Direct parameters
        if optimizer in ("SGD", "Adam"):
            yield prefix + "class_token", {
                "param_names": [prefix + "class_token"],
                "module": prefix[:-1] if prefix.endswith(".") else prefix,
                "params": [self.class_token],
                "lr": lr,
            }
        elif optimizer == "NGD":
            yield prefix + "class_token", {
                "param_names": [prefix + "class_token"],
                "module": prefix[:-1] if prefix.endswith(".") else prefix,
                "params": [self.class_token],
            }

        # Child modules
        yield from self.conv_proj.named_parameter_groups(
            optimizer=optimizer, lr=lr, prefix=prefix + "conv_proj"
        )
        yield from self.encoder.named_parameter_groups(
            optimizer=optimizer, lr=lr, prefix=prefix + "encoder"
        )
        if self.pre_head is not None:
            yield from self.pre_head.named_parameter_groups(
                optimizer=optimizer, lr=lr, prefix=prefix + "pre_head"
            )
        yield from self.fc.named_parameter_groups(
            optimizer=optimizer, lr=lr, prefix=prefix + "fc"
        )

    def parameters_and_lrs(
        self,
        lr: float,
        optimizer: Literal["SGD", "Adam", "NGD"] = "SGD",
    ) -> list[dict[str, Tensor | float]]:
        return list(self.parameter_groups(optimizer=optimizer, lr=lr))

    def _process_input(
        self,
        input: torch.Tensor,
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> torch.Tensor:
        num_sample_dims = 0 if sample_shape is None else len(sample_shape)

        if input_contains_samples:
            n, c, h, w = input.shape[num_sample_dims:]
        else:
            n, c, h, w = input.shape

        p = self.patch_size
        torch._assert(
            h == self.in_size,
            f"Wrong image height! Expected {self.in_size} but got {h}!",
        )
        torch._assert(
            w == self.in_size,
            f"Wrong image width! Expected {self.in_size} but got {w}!",
        )
        n_h = h // p
        n_w = w // p

        # (*sample_shape, n, c, h, w) -> (*sample_shape, n, hidden_dim, n_h, n_w)
        x = self.conv_proj(
            input,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        if sample_shape is not None:
            x = x.reshape(*sample_shape, n, self.hidden_dim, n_h * n_w)
        else:
            x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (*sample_shape, n, hidden_dim, (n_h * n_w)) -> (*sample_shape, n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.transpose(-2, -1)

        return x

    def representation(
        self,
        input: Float[Tensor, "*sample batch *in_feature"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch *out_feature"]:
        """Representation of the model."""
        num_sample_dims = 0 if sample_shape is None else len(sample_shape)

        # Reshape and permute the input tensor
        x = self._process_input(
            input,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )
        n = x.shape[num_sample_dims]

        # Expand the class token to the full batch
        if sample_shape is not None:
            batch_class_token = self.class_token.expand(*sample_shape, n, -1, -1)
        else:
            batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=-2)

        x = self.encoder(
            x,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=True,
            parameter_samples=parameter_samples,
        )

        # Classifier "token" as used by standard language architectures
        x = torch.select(x, num_sample_dims + 1, 0)

        if self.pre_head is not None:
            x = self.pre_head(
                x,
                sample_shape=sample_shape,
                generator=generator,
                input_contains_samples=True,
                parameter_samples=parameter_samples,
            )

        if self.normalize_representation:
            x = torch.nn.functional.normalize(x, dim=-1)

        return x

    def forward(
        self,
        input: Float[Tensor, "*sample batch *in_feature"],
        /,
        sample_shape: torch.Size | None = torch.Size([]),
        generator: torch.Generator | None = None,
        input_contains_samples: bool = False,
        parameter_samples: dict[str, Float[Tensor, "*sample parameter"]] | None = None,
    ) -> Float[Tensor, "*sample *batch *out_feature"]:
        x = self.representation(
            input,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=input_contains_samples,
            parameter_samples=parameter_samples,
        )

        x = self.fc(
            x,
            sample_shape=sample_shape,
            generator=generator,
            input_contains_samples=True,
            parameter_samples=parameter_samples,
        )
        return x


class ViT_B_16(VisionTransformer):
    """ViT_B_16

    :param **kwargs: Additional keyword arguments passed on to [``VisionTransformer``][inferno.models.VisionTransformer].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        weights: torchvision.models.Weights = torchvision.models.ViT_B_16_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            in_size=in_size,
            out_size=out_size,
            weights=weights,
            freeze=freeze,
            *args,
            **kwargs,
        )

    @classmethod
    def from_clip_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_clip_pretrained_weights(
            in_size=in_size,
            out_size=out_size,
            clip_model_name=clip_model_name,
            freeze=freeze,
            *args,
            **kwargs,
        )


class ViT_B_32(VisionTransformer):
    """ViT_B_32

    :param **kwargs: Additional keyword arguments passed on to [``VisionTransformer``][inferno.models.VisionTransformer].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            patch_size=32,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        weights: torchvision.models.Weights = torchvision.models.ViT_B_32_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            in_size=in_size,
            out_size=out_size,
            weights=weights,
            freeze=freeze,
            *args,
            **kwargs,
        )

    @classmethod
    def from_clip_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_clip_pretrained_weights(
            in_size=in_size,
            out_size=out_size,
            clip_model_name=clip_model_name,
            freeze=freeze,
            *args,
            **kwargs,
        )


class ViT_L_16(VisionTransformer):
    """ViT_L_16

    :param **kwargs: Additional keyword arguments passed on to [``VisionTransformer``][inferno.models.VisionTransformer].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            patch_size=16,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        weights: torchvision.models.Weights = torchvision.models.ViT_L_16_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            in_size=in_size,
            out_size=out_size,
            weights=weights,
            freeze=freeze,
            *args,
            **kwargs,
        )


class ViT_L_32(VisionTransformer):
    """ViT_L_32

    :param **kwargs: Additional keyword arguments passed on to [``VisionTransformer``][inferno.models.VisionTransformer].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            patch_size=32,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        weights: torchvision.models.Weights = torchvision.models.ViT_L_32_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            in_size=in_size,
            out_size=out_size,
            weights=weights,
            freeze=freeze,
            *args,
            **kwargs,
        )


class ViT_H_14(VisionTransformer):
    """ViT_H_14

    :param **kwargs: Additional keyword arguments passed on to [``VisionTransformer``][inferno.models.VisionTransformer].
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            patch_size=14,
            num_layers=32,
            num_heads=16,
            hidden_dim=1280,
            mlp_dim=5120,
            **kwargs,
        )

    @classmethod
    def from_pretrained_weights(
        cls,
        in_size: int,
        out_size: int,
        weights: torchvision.models.Weights = torchvision.models.ViT_H_14_Weights.DEFAULT,
        freeze: bool = False,
        *args,
        **kwargs,
    ):
        return super().from_pretrained_weights(
            in_size=in_size,
            out_size=out_size,
            weights=weights,
            freeze=freeze,
            *args,
            **kwargs,
        )
