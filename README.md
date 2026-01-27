# Geometric Transport of Quantum State Ensembles

This repository implements a **geometric, ensemble-based framework** for studying how small differences in global quantum preparations manifest at the level of **local, environment-conditioned quantum states** in interacting many-body systems.

The core goal is to understand **what survives reduction**:  
while global quantum states evolve unitarily and remain equally distinguishable, their **local subsystems do not**.  
This project tracks how local quantum geometry is **transported, contracted, or amplified** under interacting dynamics.

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

## 📁 Repository Structure

```text
.
├── notebooks/              # Jupyter notebooks (theory + simulations)
│   ├── theory/             # GQS formalism, distance definitions
│   ├── simulations/        # Kicked top / spin-chain dynamics
│   └── figures/            # Reproducible plots and schematics
│
├── src/                    # Core Python modules
│   ├── dynamics.py         # Hamiltonians and time evolution
│   ├── gqs.py              # Geometric quantum state construction
│   ├── distances.py        # Wasserstein, Bures, FS distances
│   └── utils.py            # Helpers and numerical tools
│
├── dat

