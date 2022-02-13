import numpy as np
import jax
import jax.numpy as jnp

def make_flow(network, n, dim, L):

    def logprob(params, s):
        flow_flatten = lambda x: network.apply(params, None, x.reshape(n, dim)).reshape(-1)
        jac = jax.jacfwd(flow_flatten)(s.reshape(-1))

        _, logdetjac = jnp.linalg.slogdet(jac)

        return logdetjac - (n*dim*np.log(L) + jax.scipy.special.gammaln(n+1))

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
