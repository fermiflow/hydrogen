import jax
import jax.numpy as jnp

from functools import partial

from MCMC import mcmc

@partial(jax.pmap, axis_name="p",
                   in_axes=(0, 
                            None, 0, 0, 
                            None, 0, 0, 
                            None, None, None),
                   static_broadcasted_argnums=(1, 4))
def sample_s_and_x(key,
                       logprob, s, params_flow,
                       logpsi2, x, params_wfn,
                       mc_steps, mc_stddev, L):
    """
        Generate new state_indices of shape (batch, n), as well as coordinate sample
    of shape (batch, n, dim), from the sample of last optimization step.
    """
    key, key_proton, key_electron = jax.random.split(key, 3)
    
    # proton move
    s, proton_acc_rate = mcmc(lambda s: logprob(params_flow, s), s, key_proton, mc_steps, mc_stddev)
    s -= L * jnp.floor(s/L)
    
    # electron move
    x, electron_acc_rate = mcmc(lambda x: logpsi2(x, params_wfn, s), x, key_electron, mc_steps, mc_stddev)
    x -= L * jnp.floor(x/L)

    return key, s, x, proton_acc_rate, electron_acc_rate

####################################################################################

from potential import potential_energy

def make_loss(logprob, logpsi, logpsi_grad_laplacian, kappa, G, L, rs, Vconst, beta, clip_factor):

    def observable_and_lossfn(params_flow, params_wfn, s, x, key):
        logp_states = logprob(params_flow, s)
        grad, laplacian = logpsi_grad_laplacian(x, params_wfn, s, key)
        print("grad.shape:", grad.shape)
        print("laplacian.shape:", laplacian.shape)

        kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))
        potential = potential_energy(jnp.concatenate([s, x], axis=1), kappa, G, L, rs) + Vconst
        Eloc = kinetic + potential
        Floc = logp_states / beta + Eloc.real

        K_mean, K2_mean, V_mean, V2_mean, \
        E_mean, E2_mean, F_mean, F2_mean, S_mean, S2_mean = \
        jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
                     (kinetic.real.mean(), (kinetic.real**2).mean(),
                      potential.mean(), (potential**2).mean(),
                      Eloc.real.mean(), (Eloc.real**2).mean(),
                      Floc.mean(), (Floc**2).mean(),
                      -logp_states.mean(), (logp_states**2).mean()
                     )
                    )
        observable = {"K_mean": K_mean, "K2_mean": K2_mean,
                      "V_mean": V_mean, "V2_mean": V2_mean,
                      "E_mean": E_mean, "E2_mean": E2_mean,
                      "F_mean": F_mean, "F2_mean": F2_mean,
                      "S_mean": S_mean, "S2_mean": S2_mean}

        def classical_lossfn(params_flow):
            logp_states = logprob(params_flow, s)

            tv = jax.lax.pmean(jnp.abs(Floc - F_mean).mean(), axis_name="p")
            Floc_clipped = jnp.clip(Floc, F_mean - clip_factor*tv, F_mean + clip_factor*tv)
            gradF_phi = (logp_states * (Floc_clipped - F_mean)).mean()
            return gradF_phi

        def quantum_lossfn(params_wfn):
            logpsix = logpsi(x, params_wfn, s)

            tv = jax.lax.pmean(jnp.abs(Eloc - E_mean).mean(), axis_name="p")
            Eloc_clipped = jnp.clip(Eloc, E_mean - clip_factor*tv, E_mean + clip_factor*tv)
            gradF_theta = 2 * (logpsix * (Eloc_clipped - E_mean).conj()).real.mean()
            return gradF_theta

        return observable, classical_lossfn, quantum_lossfn

    return observable_and_lossfn
