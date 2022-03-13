from config import * 

from ferminet import FermiNet
from sampler import make_flow

def test_sampler():
    depth = 3
    Nf = 5
    spsize, tpsize = 16, 16
    L = 1.234
    beta = 10.0 
    rs = 1.4 
    
    def flow_fn(x):
        net = FermiNet(depth, spsize, tpsize, Nf, L, 0)
        return net(x)
    model = hk.transform(flow_fn)

    n, dim = 14, 3
    key = jax.random.PRNGKey(42)
    x = jnp.array( np.random.uniform(0., L, (n, dim)) )
    params = model.init(key, x)

    from orbitals import sp_orbitals
    sp_indices, Es = sp_orbitals(dim)
    sp_indices, Es = jnp.array(sp_indices), jnp.array(Es)

    logprob = make_flow(model, n, dim, L, sp_indices[:n])

    logp_scan = logprob(params, x, True)
    logp_no_scan = logprob(params, x, False)

    assert jnp.allclose(logp_scan, logp_no_scan)
