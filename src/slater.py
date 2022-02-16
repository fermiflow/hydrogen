import jax
import jax.numpy as jnp

def logslaterdet(s, x, L, rs, phi):
    """
        Compute the logarithm of the slater determinant 

    INPUT SHAPES:
        s: (n, dim) proton position
        x: (n, dim) electron position
    """
    
    n, dim = s.shape

    rij = jnp.reshape(s, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)) # rij[:, P] == riPj
    
    rij = rij - L*jnp.rint(rij/L)
    r = jnp.linalg.norm(rij, axis=-1)
    #r = jnp.linalg.norm(jnp.sin(2*jnp.pi*rij/L), axis=-1)/(L/(2*jnp.pi))

    D = jnp.exp(-r*rs) # e^(-r/a0) = e^(-r*rs) 

    _, logabsdet = jnp.linalg.slogdet(D*phi)
    return logabsdet
