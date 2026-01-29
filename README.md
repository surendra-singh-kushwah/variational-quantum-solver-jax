# Variational Quantum Solver (JAX)

## Overview
This project explores variational methods to compute ground-state energies of quantum systems using automatic differentiation with JAX.

## Physics Background
According to the variational principle, the expectation value of the Hamiltonian over any normalized trial wavefunction provides an upper bound to the true ground-state energy. By optimizing parameters of a trial state, one can approximate the ground state of a quantum system.

## Methodology
- Parameterized trial wavefunctions
- Evaluation of energy expectation values
- Gradient-based optimization using JAX automatic differentiation

## Planned Systems
- 1D Harmonic Oscillator
- Anharmonic Oscillator
- Simple lattice models

## Tools & Libraries
- Python
- JAX
- NumPy
- SciPy
- Matplotlib

## Status
Work in progress. Initial implementations and benchmarks are under development.
