import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

def make_base(n, dim, L, beta, rs):
    def _lj(r):
        epsilon = 2*0.18                   # h2 binding enegy = 0.18 au, here 2 for au to Ry
        sigma = 1.4/rs/(2**(1/6))          # h2 bond length = 1.4 au = 2**(1/6)*sigma
        return 4*epsilon*((sigma/r)**12 - (sigma/r)**6)

    def _soft(r):
        return jnp.exp(-rs*r)

    def _h2(r):
        p = jnp.array([-1.0, 2.867, -5.819, -9.935, 4.456])
        q = jnp.array([1.0, -3.005, 7.81, 2.104, 0.4839])
        return jnp.polyval(p, r*rs)/ jnp.polyval(q, r*rs) * 2 # ( Hr = 2 Ry, a_B = a/rs )
    
    def logprob(z):
        rc = L/2

        z = z - L * jnp.floor(z/L)
        i, j = jnp.triu_indices(n, k=1)
        rij = (jnp.reshape(z, (n, 1, dim)) - jnp.reshape(z, (1, n, dim)))[i, j]
        #r = jnp.linalg.norm(jnp.sin(2*jnp.pi*rij/L), axis=-1)*(L/(2*jnp.pi))
        r = jnp.linalg.norm(rij, axis=-1)
    
        _f = _soft
        f_vmap = jax.vmap(_f)
        return -beta * jnp.sum(f_vmap(r) + f_vmap(2*rc-r) - 2*_f(rc)) 
        
    return logprob

def make_flow(network, n, dim, L, beta, rs):
    
    base_logp = make_base(n, dim, L, beta, rs)
    @partial(jax.jit, static_argnums=2)
    def logprob(params, s, scan=False):
        flow_flatten = lambda x: network.apply(params, None, x.reshape(n, dim)).reshape(-1)
        
        s_flatten = s.reshape(-1)
        if scan:
            _, jvp = jax.linearize(flow_flatten, s_flatten)
            def _body_fun(carry, x):
                return carry, jvp(x)
            _, jac = jax.lax.scan(_body_fun, None, jnp.eye(n*dim)) # this is actually jacobian transposed 
        else:
            jac = jax.jacfwd(flow_flatten)(s_flatten)
        
        _, logdetjac = jnp.linalg.slogdet(jac)
        
        #return logdetjac - (n*dim*np.log(L) + jax.scipy.special.gammaln(n+1)) #uniform base

        z = network.apply(params, None, s)
        return logdetjac + base_logp(z)
       
    return logprob


"""
    Classical score function: params, sample -> score
    This function can be useful for stochastic reconfiguration, the second-order
optimization algorithm based on classical Fisher information matrix.

Relevant dimension: (after vmapped)

INPUT:
    params: a pytree    sample: (batch, n), with elements being integers in [0, num_states).
OUTPUT:
    a pytree of the same structure as `params`, in which each leaf node has
an additional leading batch dimension.
"""
make_classical_score = lambda log_prob: jax.vmap(jax.grad(log_prob), (None, 0), 0)
