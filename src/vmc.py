import jax
import jax.numpy as jnp

from functools import partial

from mcmc import mcmc

@partial(jax.pmap, axis_name="p",
                   in_axes=(0, 
                            None, 0, 0, 
                            None, 0, 0, 
                            None, None, None, None, None, None),
                   static_broadcasted_argnums=(1, 4))
def sample_s_and_x(key,
                       logprob, s, params_flow,
                       logpsi2, x, params_wfn,
                       mc_proton_steps, mc_electron_steps,
                       mc_proton_width, mc_electron_width, L, sp_indices):
    """
        Generate new proton samples of shape (batch, n, dim), as well as coordinate sample
    of shape (batch, n, dim), from the sample of last optimization step.
    """
    key, key_momenta, key_proton, key_electron = jax.random.split(key, 4)

    # sample momenta 
    batchsize, dim = s.shape[0], s.shape[2]
    twist = jax.random.uniform(key_momenta, (batchsize, dim), minval=-0.5, maxval=0.5)
    k = 2*jnp.pi/L * (sp_indices[None, :, :] + twist[:, None, :]) # (batchsize, n//2, dim)

    # proton move
    s, proton_acc_rate = mcmc(lambda s: logprob(params_flow, s), 
                              s, key_proton, mc_proton_steps, mc_proton_width)
    s -= L * jnp.floor(s/L)
    
    # electron move
    ks = jnp.concatenate([k,s], axis=1)
    x, electron_acc_rate = mcmc(lambda x: logpsi2(x, params_wfn, ks), 
                                x, key_electron, mc_electron_steps, mc_electron_width)
    x -= L * jnp.floor(x/L)

    return key, ks, s, x, proton_acc_rate, electron_acc_rate

####################################################################################

from potential import potential_energy

def make_loss(logprob, logpsi, logpsi_grad_laplacian, kappa, G, L, rs, Vconst, beta, clip_factor):

    def observable_and_lossfn(params_flow, params_wfn, ks, s, x, key):

        logp_states = logprob(params_flow, s)
        grad, laplacian = logpsi_grad_laplacian(x, params_wfn, ks, key)
        print("grad.shape:", grad.shape)
        print("laplacian.shape:", laplacian.shape)

        kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))
        v_pp, v_ep, v_ee = potential_energy(jnp.concatenate([s, x], axis=1), kappa, G, L, rs) 
        v_pp += Vconst
        v_ee += Vconst
        
        Eloc = kinetic + (v_ep + v_ee)
        Floc = logp_states *rs**2/ beta + Eloc.real + v_pp
        
        #pressure in Gpa using viral theorem 
        # 1 Ry/Bohr^3 = 14710.513242194795 GPa 
        #http://greif.geo.berkeley.edu/~driver/conversions.html
        P = (2*kinetic.real + (v_pp + v_ep + v_ee) )/(3*(L*rs)**3)* 14710.513242194795

        K, K2, Vpp, Vpp2, Vep, Vep2, Vee, Vee2, \
        P, P2, E, E2, F, F2, S, S2 = \
        jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="p"), 
                     (kinetic.real.mean(), (kinetic.real**2).mean(),
                      v_pp.mean(), (v_pp**2).mean(), 
                      v_ep.mean(), (v_ep**2).mean(), 
                      v_ee.mean(), (v_ee**2).mean(), 
                      P.mean(), (P**2).mean(), 
                      Eloc.real.mean(), (Eloc.real**2).mean(),
                      Floc.mean(), (Floc**2).mean(),
                      -logp_states.mean(), (logp_states**2).mean()
                     )
                    )
        observable = {"K": K, "K2": K2,
                      "Vpp": Vpp, "Vpp2": Vpp2,
                      "Vep": Vep, "Vep2": Vep2,
                      "Vee": Vee, "Vee2": Vee2,
                      "P": P, "P2": P2,
                      "E": E, "E2": E2,
                      "F": F, "F2": F2,
                      "S": S, "S2": S2}

        def classical_lossfn(params_flow):
            logp_states = logprob(params_flow, s)

            tv = jax.lax.pmean(jnp.abs(Floc - F).mean(), axis_name="p")
            Floc_clipped = jnp.clip(Floc, F - clip_factor*tv, F + clip_factor*tv)
            gradF_phi = (logp_states * (Floc_clipped - F)).mean()
            return gradF_phi

        def quantum_lossfn(params_wfn):
            logpsix = logpsi(x, params_wfn, ks)

            tv = jax.lax.pmean(jnp.abs(Eloc - E).mean(), axis_name="p")
            Eloc_clipped = jnp.clip(Eloc, E - clip_factor*tv, E + clip_factor*tv)
            gradF_theta = 2 * (logpsix * (Eloc_clipped - E).conj()).real.mean()
            return gradF_theta

        return observable, classical_lossfn, quantum_lossfn

    return observable_and_lossfn