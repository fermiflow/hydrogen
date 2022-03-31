from config import * 

from logpsi import make_logpsi, make_quantum_score 
from orbitals import sp_orbitals

from jax.flatten_util import ravel_pytree

def fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, nk):
    from ferminet import FermiNet
    def flow_fn(x, k):
        model = FermiNet(depth, spsize, tpsize, Nf, L, K, init_stddev=0.001)
        return model(x, k)
    flow = hk.transform(flow_fn)

    s = jnp.array( np.random.uniform(0., L, (n, dim)) )
    x = jnp.array( np.random.uniform(0., L, (n, dim)) )
    k = jnp.array( np.random.uniform(0., 2*jnp.pi/L, (2*nk, dim)) )
    
    key = jax.random.PRNGKey(42)
    params = flow.init(key, jnp.concatenate([s, x], axis=0), k)
    return flow, params

def test_quantum_sr():
    depth, spsize, tpsize, Nf, L, K, nk = 2, 2, 2, 5, 1.234, 1, 7
    n, dim = 14, 3
    W = 1
    B = 128

    assert (nk >= n//2)

    sp_indices, _ = sp_orbitals(dim)
    sp_indices = jnp.array(sp_indices)[:nk]
    k = 2*jnp.pi/L * (sp_indices)
    k = jnp.concatenate([k, k])
    k = k.reshape(1, 2*nk, dim)

    flow, params = fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, nk)
    params_flatten, unravel = ravel_pytree(params)
    nparams = params_flatten.size

    logpsi = make_logpsi(flow, L, nk)
    quantum_score_fn = make_quantum_score(logpsi)
    
    key = jax.random.PRNGKey(42)
    key, key_s, key_x = jax.random.split(key, 3)

    s = jax.random.uniform(key_s, (W, n, dim), minval=0., maxval=L)
    x = jax.random.uniform(key_x, (B, n, dim), minval=0., maxval=L)
    ks = jnp.concatenate([jnp.repeat(k, B, axis=0),
                          jnp.repeat(s, B//W, axis=0)],   
                          axis=1)
    
    quantum_score = quantum_score_fn(x, params, ks)
    quantum_score = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(quantum_score)
   
    print (quantum_score)
    print ('quantum_score', quantum_score.shape)
    assert (quantum_score.shape == (B, nparams))

    quantum_fisher = quantum_score.conj().T.dot(quantum_score).real /B 

    quantum_score_mean = quantum_score.reshape(W, B//W, -1).mean(axis=1) # (W,Nparams)
    quantum_fisher -= quantum_score_mean.conj().T.dot(quantum_score_mean).real / W

    w, _ = jnp.linalg.eigh(quantum_fisher)
    print ('fisher eigh', w, w.min(), w.max())
    
    diag = jnp.diag(quantum_fisher)
    print ('fisher diag', diag, diag.min(), diag.max())
    print (unravel(diag))

test_quantum_sr()
