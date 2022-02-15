from config import * 

from ferminet import FermiNet

def test_flow():
    depth = 3
    Nf = 5
    spsize, tpsize = 16, 16
    L = 1.234
    
    def flow_fn(x):
        net = FermiNet(depth, spsize, tpsize, Nf, L, False)
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

def test_wfn():
    depth = 3
    Nf = 5
    spsize, tpsize = 16, 16
    L = 1.234
    np.random.seed(43)
    
    def flow_fn(x):
        net = FermiNet(depth, spsize, tpsize, Nf, L, True)
        return net(x)
    model = hk.transform(flow_fn)

    n, dim = 10, 3
    key = jax.random.PRNGKey(42)
    sx = jax.random.uniform(key, (2*n, dim), minval=0., maxval=L )

    params = model.init(key, sx)
    z, (phi_up, phi_dn) = model.apply(params, None, sx)

    print("---- Test output shape ----")
    assert (z.shape == (n, dim))
    assert (phi_up.shape == (n//2,n//2))
    assert (phi_dn.shape == (n//2,n//2))

    # Test that flow results of two "equivalent" (under lattice translations of PBC)
    # particle configurations are equivalent.
    print("---- Test the flow is well-defined under lattice translations of PBC ----")
    image = np.random.randint(-5, 6, size=(2*n, dim)) * L
    #print("image:", image / L)

    image_z, (image_phi_up, image_phi_dn) = model.apply(params, None, sx+image)
    assert jnp.allclose(image_z, z + image[n:])
    assert jnp.allclose(image_phi_up, phi_up)
    assert jnp.allclose(image_phi_dn, phi_dn)

    # Test the translation equivariance.
    print("---- Test translation equivariance ----")
    shift = jnp.array( np.random.randn(dim) )
    #print("shift:", shift)
    shift_z, (shift_phi_up, shift_phi_dn) = model.apply(params, None, sx + shift)
    assert jnp.allclose(shift_z, z + shift)
    assert jnp.allclose(shift_phi_up, phi_up)
    assert jnp.allclose(shift_phi_dn, phi_dn)

    # Test of permutation equivariance.
    print("---- Test permutation equivariance ----")
    P1 = np.random.permutation(n)  #proton
    P2 = np.random.permutation(n//2) #alpha electron
    P3 = np.random.permutation(n//2) #beta electron

    P = np.concatenate([P1, P2 +n, P3 + (n+n//2)])
    P23 = np.concatenate([P2, P3 + n//2])

    Pz, (Pphi_up, Pphi_dn) = model.apply(params, None, sx[P, :])

    assert jnp.allclose(Pz, z[P23, :])
    assert jnp.allclose(Pphi_up, phi_up[:, P2])
    assert jnp.allclose(Pphi_dn, phi_dn[:, P3])
