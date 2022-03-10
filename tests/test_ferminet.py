from config import * 

from ferminet import FermiNet
from jax.flatten_util import ravel_pytree

def test_flow():
    depth = 3
    Nf = 5
    spsize, tpsize = 16, 16
    L = 1.234
    
    def flow_fn(x):
        net = FermiNet(depth, spsize, tpsize, Nf, L, 0)
        return net(x)
    model = hk.transform(flow_fn)

    n, dim = 14, 3
    key = jax.random.PRNGKey(42)
    x = jnp.array( np.random.uniform(0., L, (n, dim)) )
    params = model.init(key, x)
    z = model.apply(params, None, x)

    # Test that flow results of two "equivalent" (under lattice translations of PBC)
    # particle configurations are equivalent.
    print("---- Test the flow is well-defined under lattice translations of PBC ----")
    image = np.random.randint(-5, 6, size=(n, dim)) * L
    #print("image:", image / L)
    imagez = model.apply(params, None, x + image)
    assert jnp.allclose(imagez, z + image)

    # Test the translation equivariance.
    print("---- Test translation equivariance ----")
    shift = jnp.array( np.random.randn(dim) )
    #print("shift:", shift)
    shiftz = model.apply(params, None, x + shift)
    assert jnp.allclose(shiftz, z + shift)

    # Test of permutation equivariance.
    print("---- Test permutation equivariance ----")
    P = np.random.permutation(n)
    Pz = model.apply(params, None, x[P, :])
    assert jnp.allclose(Pz, z[P, :])

def test_flow_params_size():
    depth = 3
    Nf = 5
    spsize, tpsize = 16, 16
    L = 1.234
    
    def init_flow(key, x):
        def flow_fn(x):
            net = FermiNet(depth, spsize, tpsize, Nf, L, 0)
            return net(x)
        model = hk.transform(flow_fn)
        params = model.init(key, x)
        return params 

    key = jax.random.PRNGKey(42)

    n, dim = 14, 3
    x = jnp.array( np.random.uniform(0., L, (n, dim)) )
    params = init_flow(key, x)
    raveled_params, _ = ravel_pytree(params)
    nparams_14 = raveled_params.size

    n, dim = 38, 3
    x = jnp.array( np.random.uniform(0., L, (n, dim)) )
    params = init_flow(key, x)
    raveled_params, _ = ravel_pytree(params)
    nparams_38 = raveled_params.size

    print ('number of params:', nparams_14)

    assert nparams_14 == nparams_38

def test_wfn_params_size():
    depth = 3
    Nf = 5
    spsize, tpsize = 16, 16
    L = 1.234
    K = 8
    nk = 19
    dim = 3
    k = jnp.array( np.random.uniform(0., 2*jnp.pi/L, (2*nk, dim)) )
     
    def init_flow(key, x, k):
        def flow_fn(x, k):
            net = FermiNet(depth, spsize, tpsize, Nf, L, K)
            return net(x, k)
        model = hk.transform(flow_fn)
        params = model.init(key, x, k)
        return params 

    key = jax.random.PRNGKey(42)

    n = 14
    x = jnp.array( np.random.uniform(0., L, (2*n, dim)) )
    params = init_flow(key, x, k)
    raveled_params, _ = ravel_pytree(params)
    nparams_14 = raveled_params.size

    print ('number of params:', nparams_14)

    n = 38
    x = jnp.array( np.random.uniform(0., L, (2*n, dim)) )
    params = init_flow(key, x, k)
    raveled_params, _ = ravel_pytree(params)
    nparams_38 = raveled_params.size

    print ('number of params:', nparams_38)

    assert nparams_14 == nparams_38
