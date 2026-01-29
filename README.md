# Geometric Transport of Quantum State Ensembles

This repository implements a **geometric, ensemble-based framework** for studying how small differences in global quantum preparations manifest at the level of **local, environment-conditioned quantum states** in interacting many-body systems.

How does a small difference in global coherent preparation manifest at the level of local geometric quantum states under interacting many-body dynamics?  Does interactions with the environment erase information about initial preparation, preserve it, or amplify it at the level of locally accessible quantum-state ensembles.

---

## ✨ Core Idea

We initialize a many-body system in two **nearby spin-coherent product states**, evolve both under the same global Hamiltonian, and compare their **local reduced states** over time.

Rather than working only with density matrices, we represent reduced states as **geometric quantum states (GQS)**:
probability measures on projective Hilbert space arising from environment conditioning.

Distances between these ensembles are quantified using the **Wasserstein (optimal transport) distance**, which captures **geometric deformation and transport of probability mass**, not just operator-level distinguishability.

---

## 🔬 Scientific Question

> **How does a small difference in global coherent preparation manifest at the level of locally accessible quantum states under interacting many-body dynamics?**

More concretely:
- Do interactions with the environment erase information about the initial preparation?
- Do local geometric differences persist?
- Or can local sensitivity be amplified, even when global distinguishability is conserved?

---

## 🧠 Why Geometric Quantum States?

Traditional reduced-state measures (e.g. entropy or trace distance) collapse the state to a single operator-level object.

In contrast, **geometric quantum states**:
- retain information about **where probability mass lies** on the quantum state manifold,
- distinguish between ensembles that are equally mixed but geometrically distinct,
- naturally connect to **optimal transport and Wasserstein geometry**.

This makes them ideal for studying **ensemble-level sensitivity** in open and many-body quantum systems.

---

## 📐 Distance Measures Used

- **Fubini–Study distance**  
  Used for pure global and local states at initial times.

- **Bures / fidelity-based distances**  
  Used for operator-level comparison of reduced density matrices.

- **Wasserstein distance (primary focus)**  
  Used to quantify geometric transport between GQS ensembles:
  how far probability mass must move on projective Hilbert space to transform one ensemble into another.

---

## ⚙️ Algorithm Overview

1. Prepare a global spin-coherent product state.
2. Prepare a nearby global coherent state with a small perturbation.
3. Evolve both states under the same global Hamiltonian.
4. Decompose the global state into system–environment form.
5. Construct environment-conditioned system states.
6. Represent reduced states as geometric quantum state ensembles.
7. Compute Wasserstein distance between ensembles.
8. Repeat for multiple perturbations and average the distance growth.

The resulting distance dynamics quantify **geometric sensitivity of local quantum states**.

---

## Install

### Option A: editable install (recommended for development)
```bash
pip install -e .
```

### Option B: install pinned dependencies only
```bash
pip install -r requirements.txt
```

## Package layout
- `gqs/operators.py`: spin operators
- `gqs/states.py`: initial states + reduced/conditional states
- `gqs/dynamics.py`: kicked-top Hamiltonian + Floquet operator
- `gqs/gqs.py`: GQS / Bloch utilities + visualizations
- `gqs/distances.py`: Fubini–Study + Wasserstein (OT) distances
- `gqs/entropy.py`: entropy/purity utilities
- `gqs/perturbations.py`: (theta,phi) perturbation helpers
- `gqs/gamma.py`: Gamma / separation-rate computations
- `gqs/plotting.py`: plotting helpers

