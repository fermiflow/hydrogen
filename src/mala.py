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
    clipped_force_fn = lambda x: jnp.clip(force_fn(x), a_min=-5/mc_width**2, a_max=5/mc_width**2)

    def step(i, state):
        x, logp, f, key, num_accepts = state
        key, key_proposal, key_accept = jax.random.split(key, 3)
        
        x_proposal = x + 0.5*f * mc_width**2 + mc_width * jax.random.normal(key_proposal, x.shape)
        logp_proposal = logp_fn(x_proposal)

        if clipped_force_fn is not None:
            f_proposal = clipped_force_fn(x_proposal)
            diff = jnp.sum(0.5*(f + f_proposal)*((x - x_proposal) + mc_width**2/4*(f - f_proposal)), axis=(-1,-2))
            #p_proposal = (x-x_proposal-0.5*f_proposal * mc_width**2)/mc_width
            #p = (x_proposal - x -0.5*f * mc_width**2)/mc_width
            #diff = -jnp.sum(p_proposal**2, axis=(-1, -2))/2 + jnp.sum(p**2, axis=(-1, -2))/2
        else:
            f_proposal = jnp.zeros_like(x_proposal)
            diff = jnp.zeros_like(logp_proposal)
        
        ratio = jnp.exp(diff + (logp_proposal - logp))
        accept = jax.random.uniform(key_accept, ratio.shape) < ratio

        x_new = jnp.where(accept[..., None, None], x_proposal, x)
        logp_new = jnp.where(accept, logp_proposal, logp)
        f_new = jnp.where(accept[..., None, None], f_proposal, f)
        num_accepts += accept.sum()
        return x_new, logp_new, f_new, key, num_accepts
    
    logp_init = logp_fn(x_init)

    if clipped_force_fn is not None:
        f_init = clipped_force_fn(x_init)
    else:
        f_init = jnp.zeros_like(x_init)

    x, logp, f, key, num_accepts = jax.lax.fori_loop(0, mc_steps, step, (x_init, logp_init, f_init, key, 0.))
    batch = np.prod( x.shape[:-2] )
    accept_rate = pmean_if_pmap(num_accepts / (mc_steps * batch), axis_name="p")
    return x, accept_rate

