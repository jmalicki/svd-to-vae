"""CPU PyTorch twin of the browser SVD-by-GD demo.

Minimize ||A - U diag(σ) V^T||_F^2 with plain SGD, then thin-QR retract U and V
onto the Stiefel manifold after each step. σ = softplus(raw).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def retract_stiefel_(X: torch.Tensor) -> None:
    """In-place thin QR: replace X with Q so X^T X = I."""
    Q, _ = torch.linalg.qr(X, mode="reduced")
    X.copy_(Q)


def fix_svd_signs_and_order_(U: torch.Tensor, V: torch.Tensor, raw: torch.Tensor) -> None:
    """Sort by descending softplus(raw); flip so max-|U| entry per column is ≥ 0."""
    sigma = F.softplus(raw)
    order = torch.argsort(sigma, descending=True)
    U.copy_(U[:, order])
    V.copy_(V[:, order])
    raw.copy_(raw[order])
    for j in range(U.shape[1]):
        i = int(torch.argmax(U[:, j].abs()).item())
        if U[i, j] < 0:
            U[:, j].neg_()
            V[:, j].neg_()


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
        fix_svd_signs_and_order_(U, V, raw)

    opt = torch.optim.SGD([U, raw, V], lr=lr)

    for t in range(steps):
        opt.zero_grad()
        sigma = F.softplus(raw)
        A_hat = (U * sigma) @ V.T
        recon = torch.sum((A - A_hat) ** 2)
        recon.backward()
        opt.step()
        with torch.no_grad():
            retract_stiefel_(U)
            retract_stiefel_(V)
            fix_svd_signs_and_order_(U, V, raw)
        if t % 100 == 0 or t == steps - 1:
            print(f"step {t:4d}  recon={recon.item():.4e}")

    with torch.no_grad():
        sigma = F.softplus(raw)
        A_hat = (U * sigma) @ V.T
        recon_err = torch.sum((A - A_hat) ** 2).item()
        U_true, S_true, Vh_true = torch.linalg.svd(A, full_matrices=False)
        A_svd = (U_true[:, :k] * S_true[:k]) @ Vh_true[:k, :]
        svd_err = torch.sum((A - A_svd) ** 2).item()

    print(f"final ‖A−Â_gd‖_F²  = {recon_err:.6e}")
    print(f"trunc SVD ‖A−Â‖_F² = {svd_err:.6e}")
    print(f"learned σ = {sigma.detach().cpu().numpy()}")
    print(f"true    σ = {S_true[:k].cpu().numpy()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(args.n, args.rank, args.lr, args.steps, args.seed)


if __name__ == "__main__":
    main()
