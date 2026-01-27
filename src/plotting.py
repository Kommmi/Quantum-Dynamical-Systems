"""Geometric Quantum States (GQS) toolkit.

This package contains utilities for:
- building spin-coherent product states,
- evolving many-body dynamics (e.g., kicked top),
- extracting environment-conditioned (GQS) ensembles,
- computing ensemble distances (Wasserstein/OT),
- and computing sensitivity diagnostics (Gamma).

Modules are split for clarity and easier reuse.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

def plot_entropy_map(S_array, theta_array, phi_array, title):
    X, Y = np.meshgrid(phi_array, theta_array)
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, S_array, 200)
    fig.colorbar(cp, ax=ax, shrink=0.6,)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\theta$')
    plt.show()

def plot_log_dist_vs_time(ln_avg_dist_L_map, theta_array, phi_array, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(theta_array)):
        for j in range(len(phi_array)):
            #ax.plot(ln_avg_dist_L_map[i, j, :], label=f"theta={theta_array[i]:.2f}, phi={phi_array[j]:.2f}")
            ax.plot(ln_avg_dist_L_map[i, j, :])
    ax.set_xlabel("Time")
    ax.set_ylabel("Log Average Distance")
    ax.set_title(title)
    #ax.legend()
    plt.show()

def plot_Gamma_from_file(fname, title=None):
    """
    Load Gamma phase-space data from file and plot using plot_entropy_map.

    Parameters
    ----------
    fname : str
        Path to saved file (.npz expected).
    title : str or None
        Plot title. If None, a default title is used.
    """
    data = np.load(fname, allow_pickle=True)

    # Required arrays
    Gamma_array = data["Gamma_map"]
    theta_array = data["theta_array"]
    phi_array = data["phi_array"]

    if title is None:
        title = r"Average local separation rate $\Gamma$"

    plot_entropy_map(Gamma_array, theta_array, phi_array, title)
    return
