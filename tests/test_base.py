from config import * 

def test_base():
    from sampler import make_base
    beta = 157888.088922572/10000
    n = 4
    dim = 3
    batchsize = 1024
    
    mc_therm = 50
    mc_steps = 1000
    mc_width = 0.003
    rs = 1.2
    L = (4/3*jnp.pi*n)**(1/3)

    logprob = make_base(n, dim, L, beta, rs)
    
    key = jax.random.PRNGKey(42)
    s = jax.random.uniform(key, (n, dim), minval=0, maxval=10)
    logp = logprob(s)
    
    # Test that flow results of two "equivalent" (under lattice translations of PBC)
    # particle configurations are equivalent.
    print("---- Test the flow is well-defined under lattice translations of PBC ----")
    image = np.random.randint(-5, 6, size=(n, dim)) * L
    #print("image:", image / L)
    logp_image = logprob(s+image)
    assert jnp.allclose(logp, logp_image)

    # Test the translation equivariance.
    print("---- Test translation equivariance ----")
    shift = jnp.array( np.random.randn(dim) )
    #print("shift:", shift)
    logp_shift = logprob(s+shift)
    assert jnp.allclose(logp, logp_shift)

    # Test of permutation equivariance.
    print("---- Test permutation equivariance ----")
    P = np.random.permutation(n)
    logp_P = logprob(s[P, :])
    assert jnp.allclose(logp, logp_P)

