from config import *

import numpy as np
from potential import kpoints, Madelung, psi, potential_energy

def test_kpoints():
    Gmax = 4
    for dim in (2, 3):
        G = kpoints(dim, Gmax)
        print("dim:", dim, "Gmax:", Gmax)
        print(G)
        print("G.shape:", G.shape)
        assert G.shape[1] == dim

def test_psi():
    n, dim = 38, 3
    L, rs = 2.0 , 1.0
    Gmax, kappa = 15, 7.0
    
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (n, dim), minval=0.0, maxval=L)

    G = kpoints(dim, Gmax)

    i, j = jnp.triu_indices(n, k=1)
    rij = ( (x[:, None, :] - x)[i, j] )/L
    rij -= jnp.rint(rij)

    v = jax.vmap(psi, (0, None, None, None), 0)(rij, kappa, G, False)
    v_forloop = jax.vmap(psi, (0, None, None, None), 0)(rij, kappa, G, True)

    assert (v.shape[0] == n*(n-1)/2)
    assert jnp.allclose(v, v_forloop)

def test_madelung():
    dim = 3 
    kappa = 10.0 
    Gmax = 15 
    G = kpoints(dim, Gmax)
    madelung = Madelung(dim, kappa, G)
    assert jnp.allclose(madelung, -2.837297)
