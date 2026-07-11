"""CPU PyTorch twin of the browser SVD-by-GD demo.

Loss:
  L = ||A - U diag(σ) V^T||_F^2 + λ ( ||U^T U - I||_F^2 + ||V^T V - I||_F^2 )
with σ = softplus(raw).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def soft_ortho_svd_loss(
    A: torch.Tensor,
    U: torch.Tensor,
    raw_sigma: torch.Tensor,
    V: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma = F.softplus(raw_sigma)
    A_hat = (U * sigma) @ V.T
    recon = torch.sum((A - A_hat) ** 2)
    k = U.shape[1]
    I = torch.eye(k, dtype=A.dtype, device=A.device)
    ortho = torch.sum((U.T @ U - I) ** 2) + torch.sum((V.T @ V - I) ** 2)
    total = recon + lam * ortho
    return total, recon, ortho


def train(
    n: int = 5,
    rank: int = 3,
    lam: float = 1.0,
    lr: float = 0.08,
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

    opt = torch.optim.Adam([U, raw, V], lr=lr)

    for t in range(steps):
        opt.zero_grad()
        loss, recon, ortho = soft_ortho_svd_loss(A, U, raw, V, lam)
        loss.backward()
        opt.step()
        if t % 100 == 0 or t == steps - 1:
            print(
                f"step {t:4d}  L={loss.item():.4e}  recon={recon.item():.4e}  ortho={ortho.item():.4e}"
            )

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
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train(args.n, args.rank, args.lam, args.lr, args.steps, args.seed)


if __name__ == "__main__":
    main()
