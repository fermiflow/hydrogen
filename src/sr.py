"""
    Second-order optimization algorithm using stochastic reconfiguration.
    The design of API signatures is in parallel with the package `optax`.
"""
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from optax._src import base

def hybrid_fisher_sr(classical_score_fn, quantum_score_fn, classical_lr, quantum_lr, decay, classical_damping, quantum_damping, classical_maxnorm, quantum_maxnorm):
    """
        Hybrid SR for both a classical probabilistic model and a set of
    quantum basis wavefunction ansatz.
    """

    def init_fn(params):
        return {'step': 0,
                'gnorm': jnp.zeros(2)
                }

    def fishers_fn(params_van, params_flow, k, s, x, state):

        classical_score = classical_score_fn(params_van, s)
        classical_score = jax.vmap(jax.vmap(lambda pytree: ravel_pytree(pytree)[0]))(classical_score)

        quantum_score = quantum_score_fn(x, params_flow, s, k)
        quantum_score = jax.vmap(jax.vmap(jax.vmap(lambda pytree: ravel_pytree(pytree)[0])))(quantum_score)

        T, W, B = quantum_score.shape[0], quantum_score.shape[1] , quantum_score.shape[2]
        print("classical_score.shape:", classical_score.shape)
        print("quantum_score.shape:", quantum_score.shape)
        
        classical_fisher = jax.lax.pmean(
                    jnp.einsum("twi,twj->ij", classical_score, classical_score)/(T*W), 
                    axis_name="p")
        
        quantum_fisher = jax.lax.pmean(
                    jnp.einsum("twbi,twbj->ij", quantum_score.conj(), quantum_score).real /(T*W*B), 
                    axis_name="p")

        quantum_score_mean = jnp.mean(quantum_score, axis=2) # (T,W,nparams) 
        factor = 1.- 1./(1+decay*state['step'])
        quantum_fisher -= factor*jax.lax.pmean(
                    jnp.einsum("twi,twj->ij", quantum_score_mean.conj(), quantum_score_mean).real/(T*W), 
                    axis_name="p")

        return classical_fisher, quantum_fisher

    def update_fn(grads, state, params):
        """
            NOTE: as the computation of (classical and quantum) Fisher information
        metrics calls for the Monte-Carlo sample `s` and `x`, we manually
        place them within the `params` argument.
        """
        grad_params_van, grad_params_flow = grads
        classical_fisher, quantum_fisher = params

        grad_params_van_raveled, params_van_unravel_fn = ravel_pytree(grad_params_van)
        grad_params_flow_raveled, params_flow_unravel_fn = ravel_pytree(grad_params_flow)
        print("grad_params_van.shape:", grad_params_van_raveled.shape)
        print("grad_params_flow.shape:", grad_params_flow_raveled.shape)

        classical_fisher += classical_damping * jnp.eye(classical_fisher.shape[0])
        update_params_van_raveled = jax.scipy.linalg.solve(classical_fisher, grad_params_van_raveled)

        lr = classical_lr/(1+decay*state['step'])
        #scale gradient according to gradnorm
        gnorm_van = jnp.sum(grad_params_van_raveled * update_params_van_raveled) 
        gnorm_van = jax.lax.pmean(gnorm_van, axis_name="p")

        scale = jnp.minimum(jnp.sqrt(classical_maxnorm/gnorm_van), lr)
        update_params_van_raveled *= -scale
        update_params_van = params_van_unravel_fn(update_params_van_raveled)

        lr = quantum_lr/(1+decay*state['step'])
        quantum_fisher += quantum_damping * jnp.eye(quantum_fisher.shape[0])
        update_params_flow_raveled = jax.scipy.linalg.solve(quantum_fisher, grad_params_flow_raveled)
        #scale gradient according to gradnorm
        gnorm_flow = jnp.sum(grad_params_flow_raveled * update_params_flow_raveled) 
        gnorm_flow = jax.lax.pmean(gnorm_flow, axis_name="p")
        scale = jnp.minimum(jnp.sqrt(quantum_maxnorm/gnorm_flow), lr)
        update_params_flow_raveled *= -scale
        update_params_flow = params_flow_unravel_fn(update_params_flow_raveled)

        state["step"] += 1
        state["gnorm"] = jnp.array([gnorm_van, gnorm_flow])

        return (update_params_van, update_params_flow), state

    return fishers_fn, base.GradientTransformation(init_fn, update_fn)
