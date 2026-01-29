"""
Variational Quantum Solver using JAX
-----------------------------------
Compute ground-state energy of a 1D harmonic oscillator
using the variational principle.
"""

import jax
import jax.numpy as jnp

# Trial wavefunction: Gaussian
def psi(x, alpha):
    return jnp.exp(-alpha * x**2)

# Hamiltonian for 1D Harmonic Oscillator
def hamiltonian_psi(x, alpha):
    d2psi = jax.grad(jax.grad(psi, argnums=0), argnums=0)(x, alpha)
    kinetic = -0.5 * d2psi
    potential = 0.5 * x**2 * psi(x, alpha)
    return kinetic + potential

# Energy expectation value
def energy(alpha):
    x = jnp.linspace(-5, 5, 1000)
    dx = x[1] - x[0]
    psi_vals = psi(x, alpha)
    Hpsi_vals = hamiltonian_psi(x, alpha)
    numerator = jnp.sum(jnp.conj(psi_vals) * Hpsi_vals) * dx
    denominator = jnp.sum(jnp.conj(psi_vals) * psi_vals) * dx
    return numerator / denominator

# Optimize alpha
grad_energy = jax.grad(energy)

alpha = 0.5
lr = 0.1

for _ in range(100):
    alpha -= lr * grad_energy(alpha)

print("Optimized alpha:", alpha)
print("Ground-state energy:", energy(alpha))
