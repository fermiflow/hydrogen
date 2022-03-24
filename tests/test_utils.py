from config import * 
from utils import logdet_matmul

def test_logdet_matmul():
    
    key = jax.random.PRNGKey(42)
    K = 8 
    N = 20 

    A = jax.random.normal(key, (K, N, N)) + 1J*jax.random.normal(key, (K, N, N))
    logw = jax.random.normal(key, (K, ))

    phase, logabsdet = logdet_matmul([A], logw)

    res = 0.0 
    for k in range(K):
        res += jnp.exp( logw[k]) * jnp.linalg.det(A[k])

    print (res, phase*jnp.exp(logabsdet))
    assert ( jnp.abs(res == phase * jnp.exp(logabsdet)) < 1e-10)
