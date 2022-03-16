import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

def make_base(n, dim, L, indices):
    '''
    ideal fermi gas in a periodic box 
    '''

    def logprob(x):
        k = 2*jnp.pi/L * indices
        k_dot_x = (k * x[:, None, :]).sum(axis=-1)
        D = 1 / L**(dim/2) * jnp.exp(1j * k_dot_x)
        _, logabsdet = jnp.linalg.slogdet(D)
        return logabsdet*2 - jax.scipy.special.gammaln(n+1)  
        
    return logprob

def make_flow(network, n, dim, L, indices):
    
    #base_logp = make_base(n, dim, L, indices)
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
        
        return logdetjac - (n*dim*np.log(L) + jax.scipy.special.gammaln(n+1)) #uniform base

        #z = network.apply(params, None, s)
        #return logdetjac + base_logp(z)
       
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
