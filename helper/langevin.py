import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from utils import pmean_if_pmap

@partial(jax.jit, static_argnums=(0, 1))
def mcmc(logp_fn, force_fn, x_init, key, mc_steps, mc_width=0.02):
    """
        Markov Chain Monte Carlo sampling algorithm.

    INPUT:
        logp_fn: callable that evaluate log-probability of a batch of configuration x.
            The signature is logp_fn(x), where x has shape (..., n, dim).
        force_fn: grad(logp_fn)
        x_init: initial value of x, with shape (..., n, dim).
        key: initial PRNG key.
        mc_steps: total number of Monte Carlo steps.
        mc_width: size of the Monte Carlo proposal.

    OUTPUT:
        x: resulting batch samples, with the same shape as `x_init`.
    """
    def step_fn(i, state):
        x, key = state
        key, subkey = jax.random.split(key)
        return x + 0.5*force_fn(x) * mc_width**2 + mc_width * jax.random.normal(subkey, x.shape), key

    return jax.lax.fori_loop(0, mc_steps, step_fn, (x_init, key))
