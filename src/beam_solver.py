"""
beam_solver.py

Finite-difference Euler–Bernoulli beam solver (clamped–free cantilever)
Method of lines in space + RK4 in time.

Running this file directly generates:
- plots/tip_displacement_clean.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class BeamParams:
    L: float = 1.0          # beam length
    EI: float = 1.0         # flexural rigidity
    rhoA: float = 1.0       # mass per unit length

    @property
    def c2(self) -> float:
        return self.EI / self.rhoA

    @property
    def c(self) -> float:
        return np.sqrt(self.c2)


def _repo_root() -> str:
    # src/ -> repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ensure_plots_dir() -> str:
    root = _repo_root()
    outdir = os.path.join(root, "plots")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def build_D4_clamped_free(N: int, L: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a finite-difference matrix D4 approximating u_xxxx on nodes j=1..N
    (we exclude j=0 because u(0)=0 is clamped and held fixed).

    Grid: x_j = j*dx, j=0..N, dx=L/N
    Unknown vector: u = [u_1, u_2, ..., u_N]^T  (length N)

    Boundary handling:
    - Clamped at x=0: u_0 = 0, and slope u_x(0)=0 -> ghost u_{-1}=u_1
    - Free at x=L: moment/shear u_xx(L)=0, u_xxx(L)=0
      implemented via ghost relations:
        u_{N+1} = 2u_N - u_{N-1}
        u_{N+2} = 3u_N - 2u_{N-1}
    """
    if N < 6:
        raise ValueError("Use N >= 6 for stable stencils near boundaries.")

    dx = L / N
    x_full = np.linspace(0.0, L, N + 1)

    # Unknown indices correspond to full-grid j=1..N
    # D4 maps u_unknown -> approx u_xxxx at j=1..N (same length N)
    D4 = np.zeros((N, N), dtype=float)

    def u_val(j: int, u_unknown: np.ndarray) -> float:
        """
        Get u_j with ghost/boundary handling, given unknown u_1..u_N.
        This is used only for constructing rows algebraically.
        """
        # full-grid mapping: unknown index k = j-1 for j=1..N
        if j == 0:
            return 0.0
        if 1 <= j <= N:
            return u_unknown[j - 1]

        # left ghost: u_{-1} = u_1
        if j == -1:
            return u_unknown[0]

        # right ghosts from free-end BCs:
        # u_{N+1} = 2u_N - u_{N-1}
        # u_{N+2} = 3u_N - 2u_{N-1}
        if j == N + 1:
            return 2.0 * u_unknown[N - 1] - u_unknown[N - 2]
        if j == N + 2:
            return 3.0 * u_unknown[N - 1] - 2.0 * u_unknown[N - 2]

        raise ValueError(f"Unexpected ghost index j={j}")

    # Build D4 row-by-row using the 5-point stencil centered at j:
    # u_xxxx(x_j) ≈ (u_{j-2} - 4u_{j-1} + 6u_j - 4u_{j+1} + u_{j+2})/dx^4
    #
    # We express each row as linear combination of unknowns u_1..u_N
    # by probing basis vectors.
    I = np.eye(N)

    for j_full in range(1, N + 1):  # j_full = 1..N
        row = np.zeros(N)
        for k in range(N):
            u_basis = I[:, k]
            val = (
                u_val(j_full - 2, u_basis)
                - 4.0 * u_val(j_full - 1, u_basis)
                + 6.0 * u_val(j_full, u_basis)
                - 4.0 * u_val(j_full + 1, u_basis)
                + u_val(j_full + 2, u_basis)
            ) / (dx**4)
            row[k] = val
        D4[j_full - 1, :] = row

    # Return x for unknowns (j=1..N) and D4
    x_unknown = x_full[1:]
    return x_unknown, D4


def rk4_step(f, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_beam(
    params: BeamParams,
    N: int,
    dt: float,
    T: float,
    u0_func,
    v0_func,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      t: (Nt+1,)
      x_full: (N+1,) full grid including x=0
      u_hist: (Nt+1, N+1) displacement including boundary node u(0,t)=0
    """
    x_unknown, D4 = build_D4_clamped_free(N=N, L=params.L)
    c2 = params.c2

    # Unknowns u_1..u_N
    u0_unknown = u0_func(x_unknown)
    v0_unknown = v0_func(x_unknown)

    y = np.concatenate([u0_unknown, v0_unknown])  # [u; v]
    n_unknown = N

    def F(_t, yvec):
        u = yvec[:n_unknown]
        v = yvec[n_unknown:]
        a = -c2 * (D4 @ u)
        return np.concatenate([v, a])

    Nt = int(np.ceil(T / dt))
    t = np.linspace(0.0, Nt * dt, Nt + 1)

    u_hist_unknown = np.zeros((Nt + 1, n_unknown))
    u_hist_unknown[0, :] = u0_unknown

    for n in range(Nt):
        y = rk4_step(F, t[n], y, dt)
        u_hist_unknown[n + 1, :] = y[:n_unknown]

    # Build full u including u0=0 at x=0
    x_full = np.linspace(0.0, params.L, N + 1)
    u_hist_full = np.zeros((Nt + 1, N + 1))
    u_hist_full[:, 0] = 0.0
    u_hist_full[:, 1:] = u_hist_unknown

    return t, x_full, u_hist_full


def _default_u0(L: float, U_tip: float = 1.0):
    def u0(x):
        # smooth, nonzero, satisfies u(0)=0
        return U_tip * (x / L) ** 2
    return u0


def _default_v0():
    def v0(x):
        return np.zeros_like(x)
    return v0


if __name__ == "__main__":
    outdir = ensure_plots_dir()

    params = BeamParams(L=1.0, EI=1.0, rhoA=1.0)
    N = 200
    dt = 5e-4
    T = 2.0

    t, x, u = simulate_beam(
        params=params,
        N=N,
        dt=dt,
        T=T,
        u0_func=_default_u0(L=params.L, U_tip=1.0),
        v0_func=_default_v0(),
    )

    # Tip displacement is u(x=L,t) = last node
    tip = u[:, -1]

    plt.figure()
    plt.plot(t, tip, label="Tip displacement")
    plt.xlabel("t")
    plt.ylabel("u(L,t)")
    plt.title("Tip displacement vs time")
    plt.grid(True)
    plt.legend()

    fname = os.path.join(outdir, "tip_displacement_clean.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
