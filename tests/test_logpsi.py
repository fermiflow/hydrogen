from config import * 

from logpsi import make_logpsi, make_logpsi_grad_laplacian, make_logpsi2
from orbitals import sp_orbitals

key = jax.random.PRNGKey(42)

def fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, rs):
    from ferminet import FermiNet

    sp_indices, _ = sp_orbitals(dim)
    sp_indices = jnp.array(sp_indices)[:n//2]

    def flow_fn(x):
        model = FermiNet(depth, spsize, tpsize, Nf, L, K, rs=rs, indices=sp_indices)
        return model(x)
    flow = hk.transform(flow_fn)

    s = jnp.array( np.random.uniform(0., L, (n, dim)) )
    x = jnp.array( np.random.uniform(0., L, (n, dim)) )
    params = flow.init(key, jnp.concatenate([s, x], axis=0))
    return flow, s, x, params

"""
    Below two tests are meant to check the various transformation properties of 
the functions logpsi and logp.
"""
def test_logpsi():
    depth, spsize, tpsize, Nf, L, K = 3, 16, 16, 5, 1.234, 8
    rs = 1.0
    n, dim = 14, 3
    flow, s, x, params = fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, rs)

    logpsi = make_logpsi(flow, L, rs)
    logpsix = logpsi(x, params, s)

    print("---- Test ln Psi_n(x + R) = ln Psi_n(x) under any lattice translation `R` of PBC ----")
    image = np.random.randint(-5, 6, size=(n, dim)) * L
    logpsix_image = logpsi(x + image, params, s)

    print("logpsix:", logpsix)
    print("logpsix_image:", logpsix_image)
    assert jnp.allclose(logpsix_image, logpsix)

    print("---- Test permutation invariance: Psi_n(Px) = +/- Psi_n(x) ----")
    Ps = np.random.permutation(n)
    Pup = np.random.permutation(n//2)
    Pdn = np.random.permutation(n//2)
    P = np.concatenate([Pup, Pdn+n//2])
    logpsix_P = logpsi(x[P, :], params, s[Ps, :])
    print("logpsix:", logpsix)
    print("logpsix_P:", logpsix_P)
    assert jnp.allclose(logpsix_P[0], logpsix[0]) and jnp.allclose(logpsix_P[1], logpsix[1])


def test_logpsi2():
    depth, spsize, tpsize, Nf, L, K = 3, 16, 16, 5, 1.234, 8
    rs = 1.0
    n, dim = 14, 3
    flow, s, x, params = fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, rs)

    logpsi = make_logpsi(flow, L, rs)
    logp = make_logpsi2(logpsi)
    logpx = logp(x[None, ...], params, s[None, ...])

    print("---- Test ln p_n(x + R) = ln p_n(x) under any lattice translation `R` of PBC ----")
    image = np.random.randint(-5, 6, size=(n, dim)) * L
    logpx_image = logp(x[None, ...] + image, params, s[None, ...])
    print (logpx_image)
    print (logpx)
    assert jnp.allclose(logpx_image, logpx)

    print("---- Test translation invariance: p_n(x + a) = p_n(x), where `a` is a common translation of all electrons ----")
    shift = jnp.array( np.random.randn(dim) )
    logpx_shift = logp(x[None, ...] + shift, params, s[None, ...] + shift)
    assert jnp.allclose(logpx_shift, logpx)

def test_kinetic_energy():
    """
        Test the present kinetic energy (i.e., laplacian) implementation, where
    the real and imaginary part are separated, yield correct result in the special
    case of identity flow.
    """
    depth, spsize, tpsize, Nf, L, K = 3, 16, 16, 5, 1.234, 8
    rs = 1.0
    n, dim = 14, 3
    flow, s, x, params = fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, rs)

    logpsi = make_logpsi(flow, L, rs)
    _, logpsi_grad_laplacian = make_logpsi_grad_laplacian(logpsi, forloop=True)
    grad, laplacian = logpsi_grad_laplacian(x[None, ...], params, s[None, ...], key)
    assert grad.shape == (1, n, dim)
    assert laplacian.shape == (1,)
    kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))

    def kinetic_energy(x, s):
        psi = lambda x: jnp.exp(logpsi(x, params, s)[0])
        h = jax.hessian(psi)(x)
        h = jnp.reshape(h, (n*dim, n*dim))
        return -jnp.trace(h)/psi(x)
    k = kinetic_energy(x, s)

    assert jnp.allclose(kinetic, k)

def test_laplacian():
    """ Check the two implementations of logpsi laplacian are equivalent. """
    depth, spsize, tpsize, Nf, L, K = 2, 4, 4, 5, 1.234, 4 
    rs = 1.0 
    n, dim = 14, 3
    flow, s, x, params = fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, rs)

    logpsi = make_logpsi(flow, L, rs)
    _, logpsi_grad_laplacian1 = make_logpsi_grad_laplacian(logpsi)
    _, logpsi_grad_laplacian2 = make_logpsi_grad_laplacian(logpsi, forloop=False)
    grad1, laplacian1 = logpsi_grad_laplacian1(x[None, ...], params, s[None, ...], key)
    grad2, laplacian2 = logpsi_grad_laplacian2(x[None, ...], params, s[None, ...], key)
    assert jnp.allclose(grad1, grad2)
    assert jnp.allclose(laplacian1, laplacian2)

def test_laplacian_hutchinson():
    """
        Use a large batch sample to (qualitatively) check the Hutchinson estimator
    of the laplacian of logpsi.
    """
    depth, spsize, tpsize, Nf, L, K = 2, 4, 4, 5, 1.234, 4 
    rs = 1.0 
    n, dim = 14, 3
    flow, s, x, params = fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, rs)

    batch = 4000

    logpsi = make_logpsi(flow, L, rs)
    logpsi_grad_laplacian = jax.jit(make_logpsi_grad_laplacian(logpsi)[1])
    grad, laplacian = logpsi_grad_laplacian(x[None, ...], params, s[None, ...], key)

    logpsi_grad_laplacian2 = jax.jit(make_logpsi_grad_laplacian(logpsi, hutchinson=True)[1])
    grad2, random_laplacian2 = logpsi_grad_laplacian2(
            jnp.tile(x, (batch, 1, 1)), params, jnp.tile(s, (batch, 1, 1)), key)
    laplacian2_mean = random_laplacian2.mean()
    laplacian2_std = random_laplacian2.std() / jnp.sqrt(batch)

    assert jnp.allclose(grad2, grad)
    print("batch:", batch)
    print("laplacian:", laplacian)
    print("laplacian hutchinson mean:", laplacian2_mean, "\tstd:", laplacian2_std)

