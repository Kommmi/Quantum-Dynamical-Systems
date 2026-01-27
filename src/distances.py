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
try:
    import ot  # POT (Python Optimal Transport)
except ImportError as e:
    ot = None

def Psi_Dist(psi0,psi1):
    f1  = np.dot(psi0,np.transpose(np.conj(psi1)))
    f2  = np.dot(psi1,np.transpose(np.conj(psi0)))
    f3  = np.dot(psi0,np.transpose(np.conj(psi0)))
    f4  = np.dot(psi1,np.transpose(np.conj(psi1)))
    d1 = np.abs(np.arccos(np.sqrt(f1*f2/(f3*f4))) )
    return d1

def Dist_ij(chi_alpha_1, chi_alpha_2):
    len_i = np.shape(chi_alpha_1)[0]
    len_j = np.shape(chi_alpha_2)[0]
    D_ij = np.zeros((len_i,len_j))
    for i in range(len_i):
        for j in range(len_j):
            D_ij[i][j]=Psi_Dist(chi_alpha_1[i],chi_alpha_2[j])
    return D_ij

def _mask_chi_lambda(chi, lam, eps=1e-12, renormalize=True):
    lam = np.asarray(lam).reshape(-1)
    chi = np.asarray(chi)
    if chi.shape[0] != lam.size:
        raise ValueError("chi and lambda must have matching first dimension")

    mask = lam > eps
    chi_m = chi[mask]
    lam_m = lam[mask]

    if renormalize:
        s = lam_m.sum()
        if s > 0:
            lam_m = lam_m / s
    return chi_m, lam_m

def Quantum_EMD(chi_1, lambda_1, chi_2, lambda_2, eps=1e-12):
    # Mask *here* (late), keeping only nonzero-probability support
    chi_1, lambda_1 = _mask_chi_lambda(chi_1, lambda_1, eps=eps, renormalize=True)
    chi_2, lambda_2 = _mask_chi_lambda(chi_2, lambda_2, eps=eps, renormalize=True)

    # Pairwise distances only on the reduced supports
    M = Dist_ij(chi_1, chi_2)

    if ot is None:
        raise ImportError(
            "POT (package 'POT', imported as 'ot') is required for Quantum_EMD. "
            "Install with: pip install POT"
        )

    # Wasserstein distance
    return ot.emd2(lambda_1, lambda_2, M)
