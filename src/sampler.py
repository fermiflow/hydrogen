import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

def make_flow(network, n, dim, L):
    
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
        
        return logdetjac - n*dim*np.log(L) #uniform base

    return logprob


"""
    Classical score function: params, sample -> score
    This function can be useful for stochastic reconfiguration, the second-order
optimization algorithm based on classical Fisher information matrix.

Relevant dimension: (after vmapped)

INPUT:
    params: a pytree    sample: (T, W, n, dim)
OUTPUT:
    a pytree of the same structure as `params`, in which each leaf node has
an additional leading batch dimension.
"""
make_classical_score = lambda log_prob: jax.vmap(jax.vmap(jax.grad(log_prob), (None, 0), 0), (None, 0), 0)

make_classical_force = lambda log_prob: jax.vmap(jax.vmap(jax.grad(lambda params, s: log_prob(params, s, False), argnums=1), (None, 0), 0), (None, 0), 0)
