"""
mode_analysis.py

Computes discrete mode shapes and frequencies from the FD operator,
and generates plots used in the report:

- plots/mode_shapes.png
- plots/tip_fft_all_modes.png
- plots/wing_shapes_over_time.png

Run:
  python src/mode_analysis.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from beam_solver import BeamParams, ensure_plots_dir, build_D4_clamped_free, simulate_beam


def cantilever_mode_shape(x: np.ndarray, L: float, beta: float) -> np.ndarray:
    """
    Analytical cantilever mode shape (unnormalized).
    beta solves cosh(beta)cos(beta) = -1
    """
    xi = x / L
    b = beta

    # Standard closed form:
    # Phi = cosh(b*xi) - cos(b*xi) - alpha*(sinh(b*xi) - sin(b*xi))
    alpha = (np.cosh(b) + np.cos(b)) / (np.sinh(b) + np.sin(b))
    Phi = np.cosh(b * xi) - np.cos(b * xi) - alpha * (np.sinh(b * xi) - np.sin(b * xi))
    return Phi


def compute_modes(params: BeamParams, N: int, n_modes: int = 3):
    x_unknown, D4 = build_D4_clamped_free(N=N, L=params.L)

    # Eigenproblem: u_tt + c^2 D4 u = 0 -> omega^2 = c^2 * lambda(D4)
    # D4 may be slightly non-symmetric from boundary handling; use eig and take real parts.
    lam, V = np.linalg.eig(D4)
    lam = np.real(lam)
    V = np.real(V)

    # Keep positive eigenvalues, sort ascending
    idx = np.argsort(lam)
    lam = lam[idx]
    V = V[:, idx]

    lam_pos = lam[lam > 1e-10]
    V_pos = V[:, lam > 1e-10]

    omegas = params.c * np.sqrt(lam_pos)
    modes = V_pos[:, :n_modes]
    omegas = omegas[:n_modes]

    # Normalize each mode by tip displacement (node at x=L -> last unknown)
    for k in range(n_modes):
        tip_val = modes[-1, k]
        if np.abs(tip_val) < 1e-14:
            continue
        modes[:, k] /= tip_val

    return x_unknown, omegas, modes


def save_mode_shapes_plot(x_unknown, modes, outdir):
    # Add x=0 boundary node with u=0 for plot
    x_full = np.concatenate([[0.0], x_unknown])
    plt.figure()

    for k in range(modes.shape[1]):
        u_full = np.concatenate([[0.0], modes[:, k]])
        plt.plot(x_full, u_full, label=f"Mode {k+1}")

    plt.xlabel("x")
    plt.ylabel(r"$\Phi_n(x)$ (normalized)")
    plt.title("Clamped–free mode shapes (FD eigenvectors)")
    plt.grid(True)
    plt.legend()

    fname = os.path.join(outdir, "mode_shapes.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")


def save_fft_plot(params: BeamParams, N: int, outdir):
    """
    Create tip FFT from three separate single-mode initial conditions
    using analytical mode shapes (clean single-frequency input).
    """
    betas = [1.875104068711961, 4.694091132974174, 7.854757438237612]
    T = 6.0
    dt = 5e-4

    plt.figure()

    for i, beta in enumerate(betas):
        # initial condition: exact mode shape
        x_full = np.linspace(0.0, params.L, N + 1)
        Phi = cantilever_mode_shape(x_full, params.L, beta)
        Phi /= Phi[-1]  # normalize to tip=1

        def u0_func(x_unknown):
            # x_unknown is x_full[1:]
            # build from full Phi
            return Phi[1:].copy()

        def v0_func(x_unknown):
            return np.zeros_like(x_unknown)

        t, x, u = simulate_beam(params, N=N, dt=dt, T=T, u0_func=u0_func, v0_func=v0_func)
        tip = u[:, -1]

        # FFT
        tip = tip - np.mean(tip)
        freqs = np.fft.rfftfreq(len(t), d=dt)
        U = np.abs(np.fft.rfft(tip))

        plt.plot(freqs, U, label=f"Mode {i+1}")

    plt.xlim(0, 50)  # adjust if needed
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("FFT magnitude")
    plt.title("Tip displacement FFT (single-mode excitations)")
    plt.grid(True)
    plt.legend()

    fname = os.path.join(outdir, "tip_fft_all_modes.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")


def save_superposition_snapshots(params: BeamParams, N: int, outdir):
    """
    Build a 3-mode superposition snapshot figure using analytical modes.
    (This is a clean 'report-quality' figure without needing animation.)
    """
    betas = [1.875104068711961, 4.694091132974174, 7.854757438237612]
    x = np.linspace(0.0, params.L, N + 1)

    # Choose amplitudes and phases (arbitrary but smooth)
    a = np.array([1.0, 0.35, 0.2])
    phases = np.array([0.0, 0.0, 0.0])

    # Compute omegas from analytical betas
    omegas = np.array([b**2 * np.sqrt(params.EI / (params.rhoA * params.L**4)) for b in betas])

    Phis = []
    for beta in betas:
        Phi = cantilever_mode_shape(x, params.L, beta)
        Phi /= Phi[-1]
        Phis.append(Phi)
    Phis = np.array(Phis)  # shape (3, N+1)

    times = [0.0, 0.25, 0.5, 0.75, 1.0]  # seconds

    plt.figure()
    for tt in times:
        u = np.zeros_like(x)
        for k in range(3):
            u += a[k] * Phis[k] * np.cos(omegas[k] * tt + phases[k])
        plt.plot(x, u, label=f"t={tt:.2f}s")

    plt.xlabel("x")
    plt.ylabel("u(x,t) (scaled)")
    plt.title("Spanwise deflection snapshots (3-mode superposition)")
    plt.grid(True)
    plt.legend()

    fname = os.path.join(outdir, "wing_shapes_over_time.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")


if __name__ == "__main__":
    outdir = ensure_plots_dir()

    params = BeamParams(L=1.0, EI=1.0, rhoA=1.0)
    N = 300

    # Mode shapes from discrete operator
    x_unknown, omegas, modes = compute_modes(params, N=N, n_modes=3)
    save_mode_shapes_plot(x_unknown, modes, outdir)

    # FFT and superposition figures
    save_fft_plot(params, N=300, outdir=outdir)
    save_superposition_snapshots(params, N=300, outdir=outdir)
