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

# Spin-1/2 operators (NOT Pauli matrices): S = sigma_Pauli / 2
sx = np.array([[0, 1.0], [1.0, 0]], dtype=complex) / 2.0
sy = np.array([[0, -1.0j], [1.0j, 0]], dtype=complex) / 2.0
sz = np.array([[1.0, 0], [0, -1.0]], dtype=complex) / 2.0

def Spin(nqubit, s_single):
    """
    Build site operators S_i (embedded) and the collective operator
        J = sum_i S_i
    on the full 2^nqubit Hilbert space.

    IMPORTANT: s_single should be the spin-1/2 operator (Pauli/2),
    i.e., sx, sy, sz as defined above.
    """
    dim = 2**nqubit
    I2 = np.eye(2, dtype=complex)

    spins = []
    J = np.zeros((dim, dim), dtype=complex)

    for i in range(nqubit):
        op = None
        for j in range(nqubit):
            factor = s_single if (i == j) else I2
            op = factor if (op is None) else np.kron(op, factor)
        spins.append(op)
        J += op

    return spins, J
