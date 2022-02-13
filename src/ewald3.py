import numpy as np 
import jax.numpy as jnp
from jax.scipy.special import erfc 
from jax import lax

def psi(rij, kappa, L, NG):
    
    V = L**3 
    G2max = NG**2 

    #real space 
    rij = rij - L*jnp.rint(rij/L)
    r = jnp.linalg.norm(rij, axis=-1)
    vr = erfc(kappa*r)/r 
    
    #k space 
    #we first do this in numpy 
    x = np.arange(-NG, NG+1)
    X,Y,Z = np.meshgrid(x,x,x)
    G = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T 

    G2 = np.sum(G**2, axis=-1)
    G = G[(G2<=G2max) * (G2>0)]
    G = G/L
    
    #after selecting G we move on to jax
    G = jnp.array(G)
    #G2 = jnp.sum(G**2, axis=-1)
    #Gr = jnp.sum(G*rij, axis=-1)
    #vk = jnp.sum( jnp.exp(-np.pi**2*G2/kappa**2) * jnp.cos(2*np.pi * Gr)/(np.pi*G2*V) )
    
    #fori_loop to save memory 
    def _body_fun(i, val):
        G2 = jnp.sum(G[i]**2)
        Gr = jnp.sum(G[i]*rij)
        return val + jnp.exp(-np.pi**2*G2/kappa**2) * jnp.cos(2*np.pi * Gr)/(np.pi*G2*V)
    vk = lax.fori_loop(0, G.shape[0], _body_fun, 0.0)
    
    return vk + vr - np.pi/kappa**2/V
