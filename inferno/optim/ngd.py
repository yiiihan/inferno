from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Union

import numpy as np
import torch
from torch.optim.optimizer import ParamsT

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


def precondition(
    grad: Float[torch.Tensor, "param ..."],
    precond_factor: Float[torch.Tensor, "param ... rank"],
    dampening: float = 0.0,
    normalize: bool = True,
):
    needs_squeezing = False
    if grad.ndim < 3 and precond_factor.ndim >= 3:
        needs_squeezing = True
        grad = grad.unsqueeze(-1)

    param_dim = grad.shape[-2]
    cov_rank = precond_factor.shape[-1]

    # For a low-rank preconditioner, only precondition in the subspace corresponding to
    # non-zero eigenvalues of the preconditioner factor.
    if cov_rank < param_dim:

        try:
            U, S, _ = torch.linalg.svd(precond_factor, full_matrices=True)

            # TODO: should we set a threshold here so we don't zero gradients due to numerical instabilities?
            # Or: should we actually set the lr for the ones eigenvalues relative to the largest eigenvalue?
            # Seems like we get runoff effect after hitting edge of stability which is worse than for other optimizers.
            precond_factor = U * torch.concat(
                [
                    S,
                    torch.ones(
                        (*S.shape[:-1], param_dim - cov_rank),
                        dtype=grad.dtype,
                        device=grad.device,
                    )
                    * S[cov_rank - 1],  # TODO: Scale with smallest non-zero eigenvalue?
                ],
                dim=-1,
            )
        except torch._C._LinAlgError:
            # When the covariance factor has a large condition number, we assume its smallest singular values are very small,
            # so we don't need to ensure the rest of the spectrum gets scaled appropriately.
            # TODO: Alternatively, just add dampening to preconditioned_grad here? We need smoother penalization of error in SVD
            # to retain good performance here, especially when precond_factor = 0, then normalization blows up the gradient.
            pass

        # Lower bounded spectrum
        # U, L, V = torch.linalg.svd(precond_factor)
        # L[L < 1.0] = self.precond_dampening

        # L, Q = torch.linalg.eigh(precond_factor @ precond_factor.mT)
        # L[L < 1.0] = self.precond_dampening
        # preconditioned_grad = Q @ torch.diag_embed(L) @ Q.mT @ grad

    # Precondition gradient
    preconditioned_grad = precond_factor @ (precond_factor.mT @ grad) + dampening * grad

    # Normalization
    if normalize:
        preconditioned_grad = preconditioned_grad / torch.linalg.vector_norm(
            preconditioned_grad, ord=2
        )
        # TODO: is this normalization working correctly even for covariance matrices, or should we be normalizing only column-wise?

    if not needs_squeezing:
        return preconditioned_grad
    else:
        return preconditioned_grad.squeeze(-1)


class NGD(torch.optim.SGD):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        precond_dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        normalize: bool = True,
        *,
        maximize: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )
        self.precond_dampening = precond_dampening
        self.normalize = normalize

        # Sort param_groups by module
        self.param_groups.sort(key=lambda x: x["module"])

    def step(self, closure=None):
        """Perform a single optimization step.

        :param closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate through parameter groups by module
        for _, module_param_groups in itertools.groupby(
            self.param_groups, key=lambda x: x["module"]
        ):
            for param_group in module_param_groups:

                # Ignore parameters which do not have a gradient
                param_group["params"] = [
                    p for p in param_group["params"] if p.grad is not None
                ]

                stacked_cov_params = None
                if not any([".cov" in name for name in param_group["param_names"]]):
                    # Mean parameters
                    is_mean_param = True
                    if (
                        "cov_params" in param_group
                        and param_group["cov_params"] is not None
                    ):
                        cov_params = [
                            p.reshape(-1, p.shape[-1])
                            for p in param_group["cov_params"]
                        ]
                        stacked_cov_params = torch.concat(cov_params, dim=-2)

                    grads = [p.grad.reshape(-1, 1) for p in param_group["params"]]
                else:
                    # Covariance parameters
                    is_mean_param = False
                    cov_params = [
                        p.reshape(-1, p.shape[-1]) for p in param_group["params"]
                    ]
                    stacked_cov_params = torch.concat(cov_params, dim=-2)

                    grads = [
                        p.grad.reshape(-1, p.shape[-1]) for p in param_group["params"]
                    ]

                stacked_grad = torch.concat(grads, dim=0)

                # Precondition mean parameters with a covariance, and covariance parameters
                if stacked_cov_params is not None:
                    with torch.no_grad():
                        stacked_grad = precondition(
                            stacked_grad,
                            precond_factor=stacked_cov_params,
                            dampening=self.precond_dampening,
                            normalize=self.normalize,
                        )

                # Update parameters
                start_idx = 0
                momentum_buffer_list = []
                lr = param_group["lr"]

                for i, param in enumerate(param_group["params"]):

                    # Split (preconditioned) stacked gradient into individual parameter gradients
                    if is_mean_param:
                        end_idx = start_idx + int(np.prod(param.shape))
                    else:
                        end_idx = start_idx + int(np.prod(param.shape[0:-1]))

                    grad = stacked_grad[start_idx:end_idx, ...].reshape(param.shape)
                    if not is_mean_param:
                        # NOTE: NGD requires factor 2 for covariance parameters
                        # grad = 2 * grad
                        grad = 0.5 * grad  # TODO: or a factor 1/2?

                    start_idx = end_idx

                    # Weight decay
                    if param_group["weight_decay"] != 0:
                        grad = grad.add(param, alpha=param_group["weight_decay"])

                    # Momentum
                    if param_group["momentum"] != 0:
                        momentum_buffer_list.append(
                            self.state[param].get("momentum_buffer")
                        )

                        if momentum_buffer_list[i] is None:
                            momentum_buffer_list[i] = grad.detach().clone()
                        else:
                            momentum_buffer_list[i].mul_(param_group["momentum"]).add_(
                                grad, alpha=1 - param_group["dampening"]
                            )

                        if param_group["nesterov"]:
                            grad = grad.add(
                                momentum_buffer_list[i], alpha=param_group["momentum"]
                            )
                        else:
                            grad = momentum_buffer_list[i]

                    # Update parameters
                    param.data.add_(
                        grad, alpha=-lr if not param_group["maximize"] else lr
                    )

                    # Update momentum_buffer in state
                    if param_group["momentum"] != 0:
                        self.state[param]["momentum_buffer"] = momentum_buffer_list[i]

        return loss
