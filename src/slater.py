import jax
import jax.numpy as jnp

def logslaterdet(indices, x, L, phi):
    """
        Compute the logarithm of the slater determinant of several plane-wave
    orbitals ln det(phi_j(r_i)), where phi_j(r_i) = 1/sqrt(L^dim) e^(i 2pi/L n_j r_i).

    INPUT SHAPES:
        indices: (n, dim) (array of integers)
        x: (n, dim)
    """

    k = 2*jnp.pi/L * indices
    k_dot_x = (k * x[:, None, :]).sum(axis=-1)
    _, dim = x.shape
    D = 1 / L**(dim/2) * jnp.exp(1j * k_dot_x)
    phase, logabsdet = jnp.linalg.slogdet(D*phi)
    return logabsdet + jnp.log(phase)
