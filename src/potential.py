import jax
import jax.numpy as jnp
from jax.scipy.special import erfc 
from functools import partial
import numpy as np 

def kpoints(dim, Gmax):
    """
        Compute all the integer k-mesh indices (n_1, ..., n_dim) in spatial
    dimention `dim`, whose length do not exceed `Gmax`.
    """
    n = np.arange(-Gmax, Gmax+1)
    nis = np.meshgrid(*( [n]*dim ))
    G = np.array([ni.flatten() for ni in nis]).T
    G2 = (G**2).sum(axis=-1)
    G = G[(G2<=Gmax**2) * (G2>0)]
    return jnp.array(G)

def Madelung(dim, kappa, G):
    """
        The Madelung constant of a simple cubic lattice of lattice constant L=1
    in spatial dimension `dim`, namely the electrostatic potential experienced by
    the unit charge at a lattice site.
    """
    Gnorm = jnp.linalg.norm(G, axis=-1)

    if dim == 3:
        g_k = jnp.exp(-jnp.pi**2 * Gnorm**2 / kappa**2) / (jnp.pi * Gnorm**2)
        g_0 = -jnp.pi / kappa**2
    elif dim == 2:
        g_k = erfc(jnp.pi * Gnorm / kappa) / Gnorm
        g_0 = -2 * jnp.sqrt(jnp.pi) / kappa

    return g_k.sum() + g_0 - 2*kappa/jnp.sqrt(jnp.pi)

def psi(rij, kappa, G, forloop=True):
    """
        The electron coordinate-dependent part 1/2 \sum_{i}\sum_{j neq i} psi(r_i, r_j)
    of the electrostatic energy (per cell) for a periodic system of lattice constant L=1.
        NOTE: to account for the Madelung part `Vconst` returned by the function
    `Madelung`, add the term 0.5*n*Vconst.
    """
    dim = rij.shape[0]

    # Only the nearest neighbor is taken into account in the present implementation of real-space summation.
    dij = jnp.linalg.norm(rij, axis=-1)
    V_shortrange = (erfc(kappa * dij) / dij)

    Gnorm = jnp.linalg.norm(G, axis=-1)

    if dim == 3:
        g_k = jnp.exp(-jnp.pi**2 * Gnorm**2 / kappa**2) / (jnp.pi * Gnorm**2)
        g_0 = -jnp.pi / kappa**2
    elif dim == 2:
        g_k = erfc(jnp.pi * Gnorm / kappa) / Gnorm
        g_0 = -2 * jnp.sqrt(jnp.pi) / kappa

    if forloop:
        def _body_fun(i, val):
            return val + g_k[i] * jnp.cos(2*jnp.pi * jnp.sum(G[i]*rij))  
        V_longrange = jax.lax.fori_loop(0, G.shape[0], _body_fun, 0.0) + g_0
    
    else:
        V_longrange = ( g_k * jnp.cos(2*jnp.pi * jnp.sum(G*rij, axis=-1)) ).sum() \
                    + g_0 
     
    potential = V_shortrange + V_longrange
    return potential

@partial(jax.vmap, in_axes=(0, 0, None, None, None, None), out_axes=0)
@partial(jax.vmap, in_axes=(0, 0, None, None, None, None), out_axes=0)
@partial(jax.vmap, in_axes=(None, 0, None, None, None, None), out_axes=0)
def potential_energy(xp, xe, kappa, G, L, rs):
    """
        Potential energy for a periodic box of size L, only the nontrivial
    coordinate-dependent part. Unit: Ry/rs^2.
        To account for the Madelung part `Vconst` returned by the function `Madelung`,
    add the term n*rs/L*Vconst. See also the docstring for function `psi`.

    INPUTS: 
        xp: (T, W, n, dim) 
        xe: (T, W, B, n, dim)
    OUTPUTS:
        (T, W, B)
    """
    
    x = jnp.concatenate((xp, xe), axis=0)
    n, dim = x.shape

    x -= L * jnp.floor(x/L)
    i, j = jnp.triu_indices(n, k=1)
    rij = ( (x[:, None, :] - x)[i, j] )/L
    rij -= jnp.rint(rij)
    
    Z = jnp.concatenate([jnp.ones(n//2), -jnp.ones(n//2)])

    total_charge = (Z[:, None]+Z )[i, j]

    v = jax.vmap(psi, (0, None, None), 0)(rij, kappa, G)

    v_pp = jnp.sum(jnp.where(total_charge==2, v, jnp.zeros_like(v)))

    v_ep = -jnp.sum(jnp.where(total_charge==0, v, jnp.zeros_like(v)))

    v_ee = jnp.sum(jnp.where(total_charge==-2, v, jnp.zeros_like(v)))

    return 2*rs/L*v_pp , 2*rs/L * v_ep , 2*rs/L*v_ee


