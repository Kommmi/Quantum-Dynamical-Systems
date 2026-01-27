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
from numpy import linalg as LA

def von_neumann_entropy(rho, base=2):
    """
    Von Neumann entropy S(rho) = -Tr(rho log rho).
    """
    # Eigenvalues of Hermitian rho
    evals = np.linalg.eigvalsh(rho)
    evals = np.clip(evals.real, 0.0, 1.0)  # guard against tiny negatives
    evals = evals[evals > 0.0]
    log_fn = np.log2 if base == 2 else np.log
    return float(-np.sum(evals * log_fn(evals)))

def purity(rho):
    """
    Purity Tr(rho^2).
    """
    return float(np.real(np.trace(rho @ rho)))

def single_qubit_entropies_from_state(Psi, d_hilbert, n_chain):
    """
    Return S( rho_site ) for each site, given a global pure state Psi.

    Output: array of length n_chain
    """
    S_sites = np.empty(n_chain, dtype=float)
    for s in range(n_chain):
        rho_s = rho_single_spin(d_hilbert, n_chain, s, Psi)
        S_sites[s] = von_neumann_entropy(rho_s, base=2)
    return S_sites

def mean_single_qubit_entropy_from_state(Psi, d_hilbert, n_chain):
    """
    Average single-qubit entanglement entropy over all sites.
    """
    return float(np.mean(single_qubit_entropies_from_state(Psi, d_hilbert, n_chain)))

def entropies_over_time(Psi0, step_map, d_hilbert, n_chain, n_steps):
    """
    Compute mean single-qubit entropy at each step t=1..n_steps.
    """
    S_t = np.empty(n_steps, dtype=float)
    Psi = np.asarray(Psi0).reshape(-1)

    for t in range(n_steps):
        Psi = step_map(Psi)  # advance one step
        S_t[t] = mean_single_qubit_entropy_from_state(Psi, d_hilbert, n_chain)

    return S_t

def S_vN_single_site_from_psi(Psi, d_hilbert, n_chain, system_site):
    rho_s = rho_single_spin(d_hilbert, n_chain, system_site, Psi)
    return von_neumann_entropy(rho_s, base=2)

def S_vN_mean_over_sites(Psi, d_hilbert, n_chain):
    return float(np.mean([S_vN_single_site_from_psi(Psi, d_hilbert, n_chain, s)
                          for s in range(n_chain)]))

def S_theta_phi_timeavg_floquet(
    nqubit, tau, kappa,
    theta_num, phi_num,
    n_avg_steps,
    burn_in=0,
    hbar=1.0,
    order="free_then_kick",
    renormalize=False
):
    """
    Compute time-averaged mean single-qubit entanglement entropy on a theta-phi grid
    for the kicked-top Floquet dynamics.

    Returns
    -------
    S_map : (theta_num, phi_num) array
    theta_array, phi_array : 1D arrays
    """
    d_hilbert = 2

    # --- Build Floquet operator once ---
    H1, H2 = Hamiltonian_QK(tau, kappa, nqubit)
    U = floquet_operator_from_H(H1, H2, tau, hbar=hbar, order=order)

    def step(Psi):
        Psi = U @ Psi
        if renormalize:
            Psi = Psi / np.linalg.norm(Psi)
        return Psi

    theta_array = np.linspace(0.0, np.pi, theta_num)
    phi_array   = np.linspace(-np.pi, np.pi, phi_num)
    S_map = np.zeros((theta_num, phi_num), dtype=float)

    for i, th in enumerate(theta_array):
        for j, ph in enumerate(phi_array):

            # Your initializer (product/coherent state on each qubit)
            Psi = Initial_state(nqubit, th, ph).reshape(-1)

            # burn-in (optional)
            for _ in range(burn_in):
                Psi = step(Psi)

            # average entanglement over n_avg_steps
            acc = 0.0
            for _ in range(n_avg_steps):
                Psi = step(Psi)
                acc += S_vN_mean_over_sites(Psi, d_hilbert, nqubit)

            S_map[i, j] = acc / n_avg_steps

    return S_map, theta_array, phi_array
