from config import * 

from logpsi import make_logpsi, make_logpsi_grad_laplacian, make_logpsi2
from orbitals import sp_orbitals
import pytest

key = jax.random.PRNGKey(42)

def fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, nk):
    from ferminet import FermiNet
    def flow_fn(x, k):
        model = FermiNet(depth, spsize, tpsize, Nf, L, K)
        return model(x, k)
    flow = hk.transform(flow_fn)

    s = jnp.array( np.random.uniform(0., L, (n, dim)) )
    x = jnp.array( np.random.uniform(0., L, (n, dim)) )
    k = jnp.array( np.random.uniform(0., 2*jnp.pi/L, (nk, dim)) )

    params = flow.init(key, jnp.concatenate([s, x], axis=0), k)
    return flow, s, x, params

"""
    Below two tests are meant to check the various transformation properties of 
the functions logpsi and logp.
"""
def test_logpsi():
    depth, spsize, tpsize, Nf, L, K, nk = 3, 16, 16, 5, 1.234, 4, 19
    n, dim = 14, 3

    assert (nk >= n//2)

    sp_indices, _ = sp_orbitals(dim)
    sp_indices = jnp.array(sp_indices)[:nk]
    k = 2*jnp.pi/L * (sp_indices)

    flow, s, x, params = fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, nk)

    logpsi = make_logpsi(flow)
    logpsix = logpsi(x, params, s, k)

    print("---- Test ln Psi_n(x + R) = ln Psi_n(x) under any lattice translation `R` of PBC ----")
    image = np.random.randint(-5, 6, size=(n, dim)) * L
    logpsix_image = logpsi(x + image, params, s, k)

    print("logpsix:", logpsix)
    print("logpsix_image:", logpsix_image)
    assert jnp.allclose(logpsix[0], logpsix_image[0]) 
    assert jnp.allclose(jnp.exp(2J*(logpsix[1] - logpsix_image[1])), 1.0)

    print("---- Test permutation invariance: Psi_n(Px) = +/- Psi_n(x) ----")
    Ps = np.random.permutation(n)
    Pup = np.random.permutation(n//2)
    Pdn = np.random.permutation(n//2)
    P = np.concatenate([Pup, Pdn+n//2])
    logpsix_P = logpsi(x[P, :], params, s[Ps, :], k)

    print("logpsix:", logpsix)
    print("logpsix_P:", logpsix_P)
    assert jnp.allclose(logpsix[0], logpsix_P[0]) 
    assert jnp.allclose(jnp.exp(2J*(logpsix[1] - logpsix_P[1])), 1.0)

@pytest.mark.skip(reason="not ensured at the moment")
def test_spin_symmetry():
    depth, spsize, tpsize, Nf, L, K, nk = 3, 16, 16, 5, 1.234, 4, 7
    rs = 1.0
    n, dim = 14, 3

    sp_indices, _ = sp_orbitals(dim)
    sp_indices = jnp.array(sp_indices)[:nk]
    k = 2*jnp.pi/L * (sp_indices)

    flow, s, x, params = fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, nk)

    logpsi = make_logpsi(flow)
    logpsix = logpsi(x, params, s, k)

    print("---- Test spin symmetry ----")
    xP = jnp.concatenate([x[n//2:, :], x[:n//2, :]])
    logpsix_P = logpsi(xP, params, s, k)

    print("logpsix:", logpsix)
    print("logpsix_P:", logpsix_P)
    assert jnp.allclose(logpsix[0], logpsix_P[0]) 
    assert jnp.allclose(jnp.exp(2J*(logpsix[1] - logpsix_P[1])), 1.0)

def test_twist():
    depth, spsize, tpsize, Nf, L, K, nk = 3, 16, 16, 5, 1.234, 8, 19
    rs = 1.0
    n, dim = 14, 3

    np.random.seed(42)

    sp_indices, _ = sp_orbitals(dim)
    sp_indices = jnp.array(sp_indices)[:nk]

    twist = jnp.array( np.random.uniform(-0.5, 0.5, (dim,)) )
    k = 2*jnp.pi/L * (sp_indices + twist[None, ...])

    flow, s, x, params = fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, nk)

    logpsi = make_logpsi(flow)
    logpsix = logpsi(x, params, s, k)

    print("---- Test Psi_n(x + R) = Psi_n(x) exp^{i*theta*R} under any lattice translation `R` with twisted BC----")
    image = np.random.randint(-5, 6, size=(n, dim)) * L
    logpsix_image = logpsi(x + image, params, s, k)
    print("logpsix:", logpsix[0] + 1J*logpsix[1])
    print("logpsix_image:", logpsix_image[0] + 1J*logpsix_image[1]- 1J* jnp.sum(2*jnp.pi*twist/L*image[:n//2]) +  1J* jnp.sum(2*jnp.pi*twist/L*image[n//2:]) )

    assert jnp.allclose(logpsix[0], logpsix_image[0])
    assert jnp.allclose(jnp.exp(1J*(logpsix[1]+ jnp.sum(2*jnp.pi*twist/L*image[:n//2])-jnp.sum(2*jnp.pi*twist/L*image[n//2:]) - logpsix_image[1])), 1.0)

def test_logpsi2():
    depth, spsize, tpsize, Nf, L, K, nk = 3, 16, 16, 5, 1.234, 8, 19
    rs = 1.0
    n, dim = 14, 3

    sp_indices, _ = sp_orbitals(dim)
    sp_indices = jnp.array(sp_indices)[:nk]
    k = 2*jnp.pi/L * (sp_indices)

    flow, s, x, params = fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, nk)

    logpsi = make_logpsi(flow)
    logp = make_logpsi2(logpsi)
    logpx = logp(x[None, None, None, ...], params, s[None, None, ...], k[None, ...])

    print("---- Test ln p_n(x + R) = ln p_n(x) under any lattice translation `R` of PBC ----")
    image = np.random.randint(-5, 6, size=(n, dim)) * L
    logpx_image = logp(x[None, None, None, ...] + image, params, s[None, None, ...], k[None, ...])
    print (logpx_image)
    print (logpx)
    assert jnp.allclose(logpx_image, logpx)

    print("---- Test translation invariance: p_n(x + a) = p_n(x), where `a` is a common translation of all electrons ----")
    shift = jnp.array( np.random.randn(dim) )
    logpx_shift = logp(x[None, None, None, ...] + shift, params, (s+shift)[None, None, ...], k[None, ...])
    print (logpx_shift)
    assert jnp.allclose(logpx_shift, logpx)

def test_kinetic_energy():
    """
        Test the present kinetic energy (i.e., laplacian) implementation, where
    the real and imaginary part are separated, yield correct result in the special
    case of identity flow.
    """
    depth, spsize, tpsize, Nf, L, K, nk = 3, 16, 16, 5, 1.234, 8, 19
    rs = 1.0
    n, dim = 14, 3

    sp_indices, _ = sp_orbitals(dim)
    sp_indices = jnp.array(sp_indices)[:nk]
    k = 2*jnp.pi/L * (sp_indices)
 
    flow, s, x, params = fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, nk)

    logpsi = make_logpsi(flow)
    _, logpsi_grad_laplacian = make_logpsi_grad_laplacian(logpsi, forloop=True)
    grad, laplacian = logpsi_grad_laplacian(x[None, None, None, ...], params, s[None, None, ...], k[None])
    assert grad.shape == (1, 1, 1, n, dim)
    assert laplacian.shape == (1, 1, 1)
    kinetic = -laplacian - (grad**2).sum(axis=(-2, -1))

    def kinetic_energy(x, s, k):
        def psi_real(x):
            log_real, log_imag = logpsi(x, params, s, k)
            return jnp.real(jnp.exp(log_real + 1J * log_imag))
        def psi_imag(x):
            log_real, log_imag = logpsi(x, params, s, k)
            return jnp.imag(jnp.exp(log_real + 1J * log_imag))

        h = jax.hessian(psi_real)(x) + 1J * jax.hessian(psi_imag)(x)
        h = jnp.reshape(h, (n*dim, n*dim))
        return -jnp.trace(h)/(psi_real(x) + 1J * psi_imag(x))
    kin = kinetic_energy(x, s, k)
    
    print (kin, (2*np.pi/L)**2*(0+6)*2)
    assert jnp.allclose(kinetic, kin)

def test_laplacian():
    """ Check the two implementations of logpsi laplacian are equivalent. """
    depth, spsize, tpsize, Nf, L, K, nk = 2, 4, 4, 5, 1.234, 4, 19
    rs = 1.0 
    n, dim = 14, 3

    sp_indices, _ = sp_orbitals(dim)
    sp_indices = jnp.array(sp_indices)[:nk]
    k = 2*jnp.pi/L * (sp_indices)

    flow, s, x, params = fermiflow(depth, spsize, tpsize, Nf, L, n, dim, K, nk)

    logpsi = make_logpsi(flow)
    _, logpsi_grad_laplacian1 = make_logpsi_grad_laplacian(logpsi)
    _, logpsi_grad_laplacian2 = make_logpsi_grad_laplacian(logpsi, forloop=False)
    grad1, laplacian1 = logpsi_grad_laplacian1(x[None, None, None, ...], params, s[None, None, ...], k[None, ...])
    grad2, laplacian2 = logpsi_grad_laplacian2(x[None, None, None, ...], params, s[None, None, ...], k[None, ...])
    assert jnp.allclose(grad1, grad2)
    assert jnp.allclose(laplacian1, laplacian2)

