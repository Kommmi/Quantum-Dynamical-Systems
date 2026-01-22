# Quantum-Dynamical-Systems

This repository studies sensitivity of open quantum systems by tracking how small perturbations in the global quantum state of a spin chain affect the local mixed-state dynamics of a single spin.

The state of the system and environment is given as:
$$|\Psi_{SE}(t)\rangle=\sum_{k=1}^{L_S}\sum_{j=1}^{L_E}\psi_{kj}(t)\|s_k^{(a)}\rangle |e_j^{(\neq a)}\rangle$$

Here, 

We simulate finite qubit chains of length $L\in[3,10]$ evolving under interacting Hamiltonians (continuous or kicked). Starting from two nearby global pure states, we evolve both under identical dynamics and reduce each to a single-spin subsystem. The reduced state is represented as a geometric quantum state—a probability distribution over pure states on complex projective space (the Bloch sphere for a qubit).

To quantify the separation between reduced states, we compute the quantum Earth Mover’s (Wasserstein) distance using the Fubini–Study metric as the transport cost. The resulting time series measures how perturbations in global state preparation propagate to the local subsystem. An averaged growth/decay rate, characterizes amplification, persistence, or suppression of these perturbations in the open quantum system.

This framework provides an ensemble-based, geometry-aware notion of sensitivity for finite quantum systems and connects quantum dynamics, optimal transport, and open-system behavior.
