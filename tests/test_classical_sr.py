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
    T = 4 
    W = 128 

    key = jax.random.PRNGKey(42)
    s = jnp.array( np.random.uniform(0., L, (n, dim)) )
    params = model.init(key, s)

    params_flatten, unravel = ravel_pytree(params)
    nparams = params_flatten.size

    logprob_novmap = make_flow(model, n, dim, L)

    classical_score_fn = make_classical_score(logprob_novmap)
    
    s = jax.random.uniform(key, (T, W, n, dim), minval=0, maxval=L)
    classical_score = classical_score_fn(params, s)
    classical_score = jax.vmap(jax.vmap(lambda pytree: ravel_pytree(pytree)[0]))(classical_score)

    print ('classical_score', classical_score.shape)
    assert (classical_score.shape == (T, W, nparams))

