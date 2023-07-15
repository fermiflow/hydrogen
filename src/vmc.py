import jax
import jax.numpy as jnp

from functools import partial

from mcmc import mcmc

@partial(jax.pmap, axis_name="p",
                   in_axes=(0, 
                            None, 0, 0, 
                            None, 0, 0, 
                            None, None, 
                            None, None, 
                            None, 0),
                   static_broadcasted_argnums=(1, 4))
def sample_s_and_x(key,
                   logprob, s, params_flow,
                   logpsi2, x, params_wfn,
                   mc_proton_steps, mc_electron_steps,
                   mc_proton_width, mc_electron_width, 
                   L, k):
    """
        s: (T, W, n, dim)
        x: (T, W, B, n, dim)
        k: (T, nk, dim)
    """
    key, key_proton, key_electron = jax.random.split(key, 3)

    walkersize, dim = s.shape[0], s.shape[2]
    batchsize = x.shape[0]

    # proton move
    s, proton_acc_rate = mcmc(lambda s: logprob(params_flow, s), 
                              s, key_proton, mc_proton_steps, mc_proton_width)
    s -= L * jnp.floor(s/L)
    
    # electron move
    x, electron_acc_rate = mcmc(lambda x: logpsi2(x, params_wfn, s, k), 
                                x, key_electron, mc_electron_steps, mc_electron_width)
    x -= L * jnp.floor(x/L)

    return key, s, x, proton_acc_rate, electron_acc_rate

####################################################################################

from potential import potential_energy

def make_loss(logprob, logpsi, logpsi_grad_laplacian, kappa, G, L, rs, Vconst, beta, clip_factor):

    def observable_and_lossfn(params_flow, params_wfn, k, s, x, key):
        '''
        k: (T, nk, dim)
        s: (T, W, n, dim)
        x: (T, W, B, n, dim)
        '''

        twistsize, walkersize, batchsize = x.shape[0], x.shape[1], x.shape[2]

        logp_states = logprob(params_flow, s) # (T, W)
        print("logp.shape", logp_states.shape)
        grad, laplacian = logpsi_grad_laplacian(x, params_wfn, s, k)
        print("grad.shape:", grad.shape)
        print("laplacian.shape:", laplacian.shape)

        kinetic = -laplacian - (grad**2).sum(axis=(-2, -1)) # (T, W, B)

        v_pp, v_ep, v_ee = potential_energy(s, x, kappa, G, L, rs) # (T, W, B)
        v_pp += Vconst
        v_ee += Vconst
        
        Eloc = kinetic + (v_ep + v_ee) # (T, W, B) 
        print("Eloc.shape", Eloc.shape)
        Floc = logp_states[..., None]*rs**2/ beta + Eloc.real + v_pp # (T, W, B)
        
        # average over electron and proton for each twist
        Fs = jnp.mean(Floc, axis=(1, 2), keepdims=True)  # (T, 1, 1)
        
        # average over electron position for each proton walker 
        Es = jnp.mean(Eloc, axis=2, keepdims=True) # (T, W, 1)
        
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
            logp_states = logprob(params_flow, s) # (T,W)

            tv = jax.lax.pmean(jnp.abs(Floc - Fs).mean(), axis_name="p")
            Floc_clipped = jnp.clip(Floc, Fs - clip_factor*tv, Fs + clip_factor*tv)
            gradF_phi = (logp_states[..., None] * (Floc_clipped - Fs)).mean()
            return gradF_phi

        def quantum_lossfn(params_wfn):
            logpsix = logpsi(x, params_wfn, s, k) # (T, W, B)

            tv = jax.lax.pmean(jnp.abs(Eloc - Es).mean(), axis_name="p")
            Eloc_clipped = jnp.clip(Eloc, Es - clip_factor*tv, Es + clip_factor*tv)
            gradF_theta = 2 * (logpsix * (Eloc_clipped - Es).conj()).real.mean()
            return gradF_theta

        return observable, classical_lossfn, quantum_lossfn

    return observable_and_lossfn
