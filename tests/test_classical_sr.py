from config import * 

from ferminet import FermiNet
from sampler import make_flow, make_classical_score
from jax.flatten_util import ravel_pytree

def test_classical_sr():
    depth = 2
    Nf = 5
    spsize, tpsize = 4, 4
    L = 1.234
    
    def flow_fn(x):
        net = FermiNet(depth, spsize, tpsize, Nf, L, 0)
        return net(x)
    model = hk.transform(flow_fn)

    n, dim = 14, 3
    B = 128 

    key = jax.random.PRNGKey(42)
    s = jnp.array( np.random.uniform(0., L, (n, dim)) )
    params = model.init(key, s)

    params_flatten, unravel = ravel_pytree(params)
    nparams = params_flatten.size

    logprob_novmap = make_flow(model, n, dim, L)

    classical_score_fn = make_classical_score(logprob_novmap)
    
    s = jax.random.uniform(key, (B, n, dim), minval=0, maxval=L)
    classical_score = classical_score_fn(params, s)
    classical_score = jax.vmap(lambda pytree: ravel_pytree(pytree)[0])(classical_score)

    print ('classical_score', classical_score.shape)
    assert (classical_score.shape == (B, nparams))

    classical_fisher = classical_score.T.dot(classical_score).real /B 

    classical_score_mean = classical_score.mean(axis=0) 
    classical_fisher -= classical_score_mean[:, None] * classical_score_mean

    w, _ = jnp.linalg.eigh(classical_fisher)
    print ('fisher eigh', w, w.min(), w.max())
    
    diag = jnp.diag(classical_fisher)
    print ('fisher diag', diag, diag.min(), diag.max())
    print (unravel(diag))

