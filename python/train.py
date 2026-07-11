"""CPU PyTorch twin of the browser SVD-by-GD demo.

Minimize mean squared reconstruction error with global-RMS SGD (one scalar
Adam-style second moment), then thin-QR retract U and V onto the Stiefel
manifold. Armijo backtracking on the retract keeps a reached Eckart–Young Â
from walking away. σ = softplus(raw). Sign/σ-order is display-only.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

# Match web/src/svdGrad.ts
LR_BETA2 = 0.999
ARMIJO_C = 1e-4
ARMIJO_MAX_BT = 12


def retract_stiefel_(X: torch.Tensor) -> None:
    """In-place thin QR: replace X with Q so X^T X = I."""
    Q, _ = torch.linalg.qr(X, mode="reduced")
    X.copy_(Q)


def grad_second_moment_fp_floor(data_scale: float) -> float:
    """Float64 floor on mean(g²); same ULP scale as the web FP appendix."""
    ulp = torch.finfo(torch.float64).eps * max(data_scale, torch.finfo(torch.float64).eps)
    return float(ulp * ulp)


def order_factors_for_display(
    U: torch.Tensor, sigma: torch.Tensor, V: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort by descending σ; flip so max-|U| entry per column is ≥ 0. Does not change Â."""
    order = torch.argsort(sigma, descending=True)
    Uo = U[:, order].clone()
    Vo = V[:, order].clone()
    so = sigma[order].clone()
    for j in range(Uo.shape[1]):
        i = int(torch.argmax(Uo[:, j].abs()).item())
        if Uo[i, j] < 0:
            Uo[:, j].neg_()
            Vo[:, j].neg_()
    return Uo, so, Vo


def mean_grad_sq(params: list[torch.Tensor]) -> float:
    total = 0.0
    count = 0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        total += float(torch.sum(g * g).item())
        count += g.numel()
    return total / count if count else 0.0


def sum_grad_sq(params: list[torch.Tensor]) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(torch.sum(p.grad * p.grad).item())
    return total


def train(
    n: int = 5,
    rank: int = 3,
    lr: float = 0.01,
    steps: int = 800,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    k = max(1, min(rank, n))
    A = torch.randn(n, n)
    U = torch.randn(n, k, requires_grad=True)
    raw = torch.randn(k, requires_grad=True)
    V = torch.randn(n, k, requires_grad=True)
    with torch.no_grad():
        U.mul_(0.3)
        raw.mul_(0.3)
        V.mul_(0.3)
        retract_stiefel_(U)
        retract_stiefel_(V)

    params = [U, raw, V]
    v = 0.0
    v_fp = grad_second_moment_fp_floor(float(A.abs().max().item()))

    for t in range(1, steps + 1):
        for p in params:
            if p.grad is not None:
                p.grad = None
        sigma = F.softplus(raw)
        A_hat = (U * sigma) @ V.T
        mse = torch.mean((A - A_hat) ** 2)
        mse.backward()

        mean_sq = mean_grad_sq(params)
        sum_sq = sum_grad_sq(params)
        v = LR_BETA2 * v + (1.0 - LR_BETA2) * mean_sq
        v_hat = v / (1.0 - LR_BETA2**t)
        v_use = max(v_hat, mean_sq, v_fp)
        eta = lr / (math.sqrt(v_use) + math.sqrt(v_fp))
        L0 = float(mse.item())
        dir_deriv = -eta * sum_sq

        with torch.no_grad():
            U0, raw0, V0 = U.clone(), raw.clone(), V.clone()
            gU, gRaw, gV = U.grad.clone(), raw.grad.clone(), V.grad.clone()
            alpha = 1.0
            accepted = False
            for _ in range(ARMIJO_MAX_BT):
                U.copy_(U0 - alpha * eta * gU)
                raw.copy_(raw0 - alpha * eta * gRaw)
                V.copy_(V0 - alpha * eta * gV)
                retract_stiefel_(U)
                retract_stiefel_(V)
                sigma_try = F.softplus(raw)
                A_try = (U * sigma_try) @ V.T
                L_try = float(torch.mean((A - A_try) ** 2).item())
                if L_try <= L0 + ARMIJO_C * alpha * dir_deriv:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                U.copy_(U0)
                raw.copy_(raw0)
                V.copy_(V0)

        if t % 100 == 0 or t == steps:
            with torch.no_grad():
                recon = torch.sum((A - (U * F.softplus(raw)) @ V.T) ** 2).item()
            print(f"step {t:4d}  η={eta:.4e}  recon={recon:.4e}")

    with torch.no_grad():
        sigma = F.softplus(raw)
        A_hat = (U * sigma) @ V.T
        recon_err = torch.sum((A - A_hat) ** 2).item()
        _U_disp, sigma_disp, _V_disp = order_factors_for_display(U, sigma, V)
        U_true, S_true, Vh_true = torch.linalg.svd(A, full_matrices=False)
        A_svd = (U_true[:, :k] * S_true[:k]) @ Vh_true[:k, :]
        svd_err = torch.sum((A - A_svd) ** 2).item()

    print(f"final ‖A−Â_gd‖_F²  = {recon_err:.6e}")
    print(f"trunc SVD ‖A−Â‖_F² = {svd_err:.6e}")
    print(f"learned σ (display) = {sigma_disp.detach().cpu().numpy()}")
    print(f"true    σ            = {S_true[:k].cpu().numpy()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01, help="base η₀")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(args.n, args.rank, args.lr, args.steps, args.seed)


if __name__ == "__main__":
    main()
