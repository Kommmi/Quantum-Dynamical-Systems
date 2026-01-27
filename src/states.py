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

def Initial_state(nqubit, theta0, phi0):
    """
    Product state of n qubits, each initialized to
    cos(theta/2)|0> + e^{-i phi} sin(theta/2)|1>.
    """
    ket0 = np.array([1.0, 0.0], dtype=complex)
    ket1 = np.array([0.0, 1.0], dtype=complex)

    psi_single = (
        np.cos(theta0 / 2.0) * ket0
        + np.sin(theta0 / 2.0) * np.exp(1j * phi0) * ket1
    )

    Psi = psi_single
    for _ in range(nqubit - 1):
        Psi = np.kron(Psi, psi_single)

    return Psi

def Reduced_state_single_site(d_hilbert, n_chain, system_site, Psi_SE, lambda_E=None, eps=1e-12):
    """
    Compute conditional pure system states |chi_a> for a user-chosen single site in a chain and a
    associated.

    For each environment basis outcome a:
        |chi_a> = (1/sqrt(lambda_E[a])) * sum_k psi_{k,a} |k>
    where lambda_E[a] = sum_k |psi_{k,a}|^2.

    Parameters
    ----------
    d_hilbert : int
        Local Hilbert space dimension (2 for qubits).
    n_chain : int
        Number of subsystems.
    system_site : int
        Index (0-based) of the chosen system site.
    Psi_SE : array-like, complex
        Pure state vector, length d_hilbert**n_chain (or shape (1, d_hilbert**n_chain)).
    lambda_E : array-like or None
        Optional precomputed lambda_E of length d_hilbert**(n_chain-1).
        If None, it is computed internally.
    eps : float
        Threshold to avoid division by ~0.

    Returns
    -------
    chi_S : np.ndarray, shape (d_hilbert**(n_chain-1), d_hilbert), complex
        Row a is the conditional pure state |chi_a> written in the system basis.
        Rows for which lambda_E[a] <= eps are left as all zeros.
    lambda_E : np.ndarray, shape (d_hilbert**(n_chain-1),), float
        The environment probabilities used for normalization.
    """
    d = int(d_hilbert)
    n = int(n_chain)

    if not (0 <= system_site < n):
        raise ValueError(f"system_site must be in [0, {n-1}]")

    Psi = np.asarray(Psi_SE).reshape(-1)
    expected = d**n
    if Psi.size != expected:
        raise ValueError(f"Psi_SE must have length {expected}, got {Psi.size}")

    # (1) Tensor view: one axis per subsystem
    Psi_tensor = Psi.reshape((d,) * n)

    # (2) Put the chosen site first: (system, env...)
    Psi_perm = np.moveaxis(Psi_tensor, system_site, 0)

    # (3) Flatten env indices: A has shape (d, d^(n-1))
    A = Psi_perm.reshape(d, -1)

    # (4) lambda_E[a] = sum_k |A_{k,a}|^2
    if lambda_E is None:
        lambda_E = np.sum(np.abs(A)**2, axis=0)
    else:
        lambda_E = np.asarray(lambda_E).reshape(-1)
        if lambda_E.size != d**(n-1):
            raise ValueError(f"lambda_E must have length {d**(n-1)}, got {lambda_E.size}")

    # (5) chi_a = A[:,a] / sqrt(lambda_E[a])  (column normalization)
    chi = np.zeros_like(A, dtype=complex)  # (d, d^(n-1))
    mask = lambda_E > eps
    chi[:, mask] = A[:, mask] / np.sqrt(lambda_E[mask])

    # Return as (env_state_index, system_dim) like your original chi_S[a]
    chi_S = chi.T  # shape (d^(n-1), d)

    return chi_S, lambda_E

def rho_single_spin(d_hilbert, n_chain, system_site, Psi_SE):
    """
    Single-spin reduced density matrix rho_s for a user-chosen subsystem in an n-spin pure state.

    Computes: rho_s = Tr_env( |Psi><Psi| )

    Parameters
    ----------
    d_hilbert : int
        Local Hilbert space dimension (2 for qubits).
    n_chain : int
        Number of subsystems in the chain.
    system_site : int
        Which subsystem is the "system" (0-based index: 0,...,n_chain-1).
    Psi_SE : array-like, complex
        Pure state vector of length d_hilbert**n_chain.

    Returns
    -------
    rho_s : np.ndarray, shape (d_hilbert, d_hilbert), complex
        Reduced density matrix of the chosen subsystem.
    """
    d = int(d_hilbert)
    n = int(n_chain)

    if not (0 <= system_site < n):
        raise ValueError(f"system_site must be in [0, {n-1}]")

    Psi = np.asarray(Psi_SE).reshape(-1)
    expected = d**n
    if Psi.size != expected:
        raise ValueError(f"Psi_SE must have length {expected}, got {Psi.size}")

    # Reshape to n-index tensor and move system axis to front
    Psi_tensor = Psi.reshape((d,) * n)
    Psi_perm = np.moveaxis(Psi_tensor, system_site, 0)

    # Flatten env indices -> matrix A with shape (d, d^(n-1))
    A = Psi_perm.reshape(d, -1)

    # Partial trace over environment: rho_s = A A^\dagger
    rho_s = A @ A.conj().T

    # Numerical hygiene: enforce Hermiticity (optional)
    rho_s = 0.5 * (rho_s + rho_s.conj().T)

    return rho_s
