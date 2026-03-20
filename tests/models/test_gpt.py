from collections.abc import MutableMapping
import itertools

import torch
from torch import testing

import inferno
from inferno.bnn import params

import pytest


# ---------------------------------------------------------------------------
# Small model config used across tests (fast, not realistic)
# ---------------------------------------------------------------------------

_VOCAB = 64
_CTX = 16
_LAYERS = 2
_HEADS = 2
_HIDDEN = 16
_MLP = 32
_BATCH = 4
_SEQ = 8


def _make_gpt(cov=None, **kwargs):
    return inferno.models.GPT(
        vocab_size=_VOCAB,
        context_length=_CTX,
        num_layers=_LAYERS,
        num_heads=_HEADS,
        hidden_dim=_HIDDEN,
        mlp_dim=_MLP,
        cov=cov,
        **kwargs,
    )


def _tokens(batch=_BATCH, seq=_SEQ):
    return torch.randint(0, _VOCAB, (batch, seq))


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_output_shape_deterministic():
    model = _make_gpt(cov=None)
    model.eval()
    with torch.no_grad():
        out = model(_tokens())
    assert out.shape == (_BATCH, _SEQ, _VOCAB)


def test_output_shape_with_sample_shape():
    sample_shape = (5,)
    model = _make_gpt(cov=params.LowRankCovariance(2))
    model.eval()
    with torch.no_grad():
        out = model(_tokens(), sample_shape=sample_shape)
    assert out.shape == (*sample_shape, _BATCH, _SEQ, _VOCAB)


def test_output_shape_with_2d_sample_shape():
    sample_shape = (3, 2)
    model = _make_gpt(cov=params.DiagonalCovariance())
    model.eval()
    with torch.no_grad():
        out = model(_tokens(), sample_shape=sample_shape)
    assert out.shape == (*sample_shape, _BATCH, _SEQ, _VOCAB)


# ---------------------------------------------------------------------------
# sample_shape=None is equivalent to forward with mean parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cov",
    [
        params.DiagonalCovariance(),
        params.LowRankCovariance(2),
        {"lm_head": params.DiagonalCovariance()},
    ],
)
def test_sample_shape_none_equals_mean_params(cov):
    """sample_shape=None should give the same result as a deterministic model
    with the same mean weights."""
    torch.manual_seed(0)
    model = _make_gpt(cov=cov)
    det_model = _make_gpt(cov=None)
    det_model.load_state_dict(model.state_dict(), strict=False)

    tokens = _tokens()
    model.eval()
    det_model.eval()
    with torch.no_grad():
        out_none = model(tokens, sample_shape=None)
        out_det = det_model(tokens)

    testing.assert_close(out_none, out_det)


# ---------------------------------------------------------------------------
# Covariance specification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cov,valid",
    [
        (None, True),
        (params.DiagonalCovariance(), True),
        (params.LowRankCovariance(2), True),
        # last-layer only
        ({"lm_head": params.DiagonalCovariance()}, True),
        # decoder only
        ({"decoder": params.LowRankCovariance(2)}, True),
        # per-block specification
        (
            {
                "decoder": {
                    "layers.block_0": params.DiagonalCovariance(),
                    "layers.block_1": params.LowRankCovariance(2),
                },
                "lm_head": params.KroneckerCovariance(),
            },
            True,
        ),
        # per-sub-module inside a block
        (
            {
                "decoder": {
                    "layers.block_1": {
                        "self_attention": params.DiagonalCovariance(),
                        "mlp": params.LowRankCovariance(2),
                    }
                }
            },
            True,
        ),
        # invalid key
        ({"wrong_key": params.DiagonalCovariance()}, False),
    ],
)
def test_covariance_spec(cov, valid):
    if not valid:
        with pytest.raises(ValueError):
            _make_gpt(cov=cov)
        return

    model = _make_gpt(cov=cov)

    if cov is None:
        for name, module in model.named_modules():
            if ".".join(name.split(".")[-2:]) == "params.cov":
                assert module is None

    elif isinstance(cov, params.FactorizedCovariance):
        for name, module in model.named_modules():
            if ".".join(name.split(".")[-2:]) == "params.cov":
                assert isinstance(module, type(cov))

    elif isinstance(cov, dict):

        def flatten_dict(d, parent=""):
            items = []
            for k, v in d.items():
                new_key = f"{parent}.{k}" if parent else k
                if isinstance(v, MutableMapping):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        _flat = flatten_dict(cov)
        # map "q" -> "q_proj" etc. (same hack as test_vit)
        cov_flat = {}
        for key in _flat:
            new_key = key
            for old, new in [
                ("self_attention.q", "self_attention.q_proj"),
                ("self_attention.k", "self_attention.k_proj"),
                ("self_attention.v", "self_attention.v_proj"),
                ("self_attention.out", "self_attention.out_proj"),
            ]:
                new_key = new_key.replace(old, new, 1)
            cov_flat[new_key] = _flat[key]

        for name, module in model.named_modules():
            if ".".join(name.split(".")[-2:]) != "params.cov":
                continue
            name_prefix = ".".join(name.split(".")[:-2])
            prefixes = list(
                itertools.accumulate(
                    name_prefix.split("."), lambda x, y: f"{x}.{y}"
                )
            )
            for prefix in reversed(prefixes):
                if prefix in cov_flat:
                    spec = cov_flat[prefix]
                    if spec is None:
                        assert module is None
                    else:
                        assert isinstance(module, type(spec))
                    break
            else:
                assert module is None


# ---------------------------------------------------------------------------
# parameters_and_lrs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("optimizer", ["SGD", "Adam"])
@pytest.mark.parametrize(
    "cov",
    [None, params.DiagonalCovariance(), {"lm_head": params.LowRankCovariance(2)}],
)
def test_parameters_and_lrs_covers_all_params(cov, optimizer):
    """Every trainable parameter must appear in exactly one param group.

    Non-trainable parameters (e.g. temperature, requires_grad=False) are
    excluded — they are tuned separately by TemperatureScaler, not the optimizer.
    """
    model = _make_gpt(cov=cov)
    param_groups = model.parameters_and_lrs(lr=1e-3, optimizer=optimizer)

    grouped = {id(p) for g in param_groups for p in g["params"]}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert grouped == trainable, (
        f"Mismatch: {len(trainable) - len(grouped)} trainable params not in any group"
    )


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def test_generate_shape():
    model = _make_gpt(cov=None)
    model.eval()
    tokens = _tokens(batch=2, seq=4)
    with torch.no_grad():
        out = model.generate(tokens, max_new_tokens=6)
    assert out.shape == (2, 4 + 6)


def test_generate_greedy():
    """temperature=0 should be deterministic."""
    model = _make_gpt(cov=None)
    model.eval()
    tokens = _tokens(batch=1, seq=4)
    with torch.no_grad():
        out1 = model.generate(tokens, max_new_tokens=5, temperature=0.0)
        out2 = model.generate(tokens, max_new_tokens=5, temperature=0.0)
    testing.assert_close(out1, out2)


def test_generate_top_k():
    model = _make_gpt(cov=None)
    model.eval()
    tokens = _tokens(batch=2, seq=4)
    with torch.no_grad():
        out = model.generate(tokens, max_new_tokens=4, temperature=1.0, top_k=5)
    assert out.shape == (2, 8)


def test_generate_context_cropping():
    """Sequences longer than context_length should be cropped before forward."""
    model = _make_gpt(cov=None)
    model.eval()
    # Start with a sequence longer than context_length
    tokens = _tokens(batch=1, seq=_CTX + 2)
    with torch.no_grad():
        out = model.generate(tokens, max_new_tokens=3)
    assert out.shape == (1, _CTX + 2 + 3)


# ---------------------------------------------------------------------------
# Representation (hidden states)
# ---------------------------------------------------------------------------


def test_representation_shape():
    model = _make_gpt(cov=None)
    model.eval()
    with torch.no_grad():
        rep = model.representation(_tokens())
    assert rep.shape == (_BATCH, _SEQ, _HIDDEN)


def test_representation_shape_with_samples():
    sample_shape = (4,)
    model = _make_gpt(cov=params.DiagonalCovariance())
    model.eval()
    with torch.no_grad():
        rep = model.representation(_tokens(), sample_shape=sample_shape)
    assert rep.shape == (*sample_shape, _BATCH, _SEQ, _HIDDEN)
