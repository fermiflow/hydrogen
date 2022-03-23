"""
    Second-order optimization algorithm using stochastic reconfiguration.
    The design of API signatures is in parallel with the package `optax`.
"""
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from optax._src import base

def hybrid_fisher_sr(classical_score_fn, quantum_score_fn, classical_lr, quantum_lr, decay, damping, max_norm):
    """
        Hybrid SR for both a classical probabilistic model and a set of
    quantum basis wavefunction ansatz.
    """

    def init_fn(params):
        return {'step': 0}

    def fishers_fn(params_van, params_flow, ks, s, x):

        classical_score = classical_score_fn(params_van, s)
        classical_score = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(classical_score)

        quantum_score = quantum_score_fn(x, params_flow, ks)
        quantum_score = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(quantum_score)

        walkersize, batchsize = classical_score.shape[0], quantum_score.shape[0]
        print("classical_score.shape:", classical_score.shape)
        print("quantum_score.shape:", quantum_score.shape)
        
        quantum_score_mean = quantum_score.reshape(walkersize, batchsize//walkersize, -1).mean(axis=1) # (W,Nparams)
        #quantum_score_mean = jax.lax.pmean(quantum_score.mean(axis=0), axis_name="p")

        classical_fisher = jax.lax.pmean(
                    classical_score.T.dot(classical_score) / walkersize,
                    axis_name="p")
        quantum_fisher = jax.lax.pmean(
                    quantum_score.conj().T.dot(quantum_score).real / batchsize,
                    axis_name="p")

        return classical_fisher, quantum_fisher, quantum_score_mean

    def update_fn(grads, state, params):
        """
            NOTE: as the computation of (classical and quantum) Fisher information
        metrics calls for the Monte-Carlo sample `s` and `x`, we manually
        place them within the `params` argument.
        """
        grad_params_van, grad_params_flow = grads
        classical_fisher, quantum_fisher, quantum_score_mean = params
        
        #quantum_fisher -= (quantum_score_mean.conj()[:, None] * quantum_score_mean).real
        walkersize = quantum_score_mean.shape[0]
        quantum_fisher -= jax.lax.pmean(
                          quantum_score_mean.conj().T.dot(quantum_score_mean).real/walkersize, 
                          axis_name="p")

        grad_params_van_raveled, params_van_unravel_fn = ravel_pytree(grad_params_van)
        grad_params_flow_raveled, params_flow_unravel_fn = ravel_pytree(grad_params_flow)
        print("grad_params_van.shape:", grad_params_van_raveled.shape)
        print("grad_params_flow.shape:", grad_params_flow_raveled.shape)

        classical_fisher += damping * jnp.eye(classical_fisher.shape[0])
        update_params_van_raveled = jax.scipy.linalg.solve(classical_fisher, grad_params_van_raveled)

        lr = classical_lr/(1+decay*state['step'])
        #scale gradient according to gradnorm
        gnorm_van = jnp.sum(grad_params_van_raveled * update_params_van_raveled) 
        gnorm_van = jax.lax.pmean(gnorm_van, axis_name="p")

        scale = jnp.minimum(jnp.sqrt(max_norm/gnorm_van), lr)
        update_params_van_raveled *= -scale
        update_params_van = params_van_unravel_fn(update_params_van_raveled)

        lr = quantum_lr/(1+decay*state['step'])
        quantum_fisher += damping * jnp.eye(quantum_fisher.shape[0])
        update_params_flow_raveled = jax.scipy.linalg.solve(quantum_fisher, grad_params_flow_raveled)
        #scale gradient according to gradnorm
        gnorm_flow = jnp.sum(grad_params_flow_raveled * update_params_flow_raveled) 
        gnorm_flow = jax.lax.pmean(gnorm_flow, axis_name="p")
        scale = jnp.minimum(jnp.sqrt(max_norm/gnorm_flow), lr)
        update_params_flow_raveled *= -scale
        update_params_flow = params_flow_unravel_fn(update_params_flow_raveled)

        state["step"] += 1

        return (update_params_van, update_params_flow), state

    return fishers_fn, base.GradientTransformation(init_fn, update_fn)
