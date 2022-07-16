from config import *

import numpy as np
from potential import kpoints, Madelung, psi, potential_energy
from pyscf.pbc import gto, scf

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

def test_ewald():

    n, dim = 14, 3
    L, rs = 20.0, 1.0
    Gmax, kappa = 15, 7.0
    
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (n, dim), minval=0.0, maxval=L)

    G = kpoints(dim, Gmax)

    Vpp = potential_energy(jnp.tile(x,[2,1]), kappa, G, L, rs)[0]
    Vconst = Madelung(dim, kappa, G)*n*rs/L
    print('vpp:', Vpp)
    print('vconst:', Vconst)
    print('vpp+vconst:', Vpp+Vconst)

    cell = gto.Cell()
    cell.unit = 'B'
    cell.atom = []
    for p in range(n):
            cell.atom.append(['H', tuple(x[p])])
    cell.spin = 0
    cell.basis = 'gth-szv'
    cell.a = np.eye(3) * L
    cell.build()
    kmf = scf.khf.KRHF(cell, kpts=G)
    Vpp_pyscf = kmf.energy_nuc()*2
    print('pyscf vpp:', Vpp_pyscf)
    jnp.allclose(Vpp+Vconst, Vpp_pyscf)
