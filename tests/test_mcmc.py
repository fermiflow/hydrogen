from config import * 

from sampler import make_base
from mcmc import mcmc 
from utils import shard 

def test_mcmc():
    beta = 157888.088922572/10000
    print ('beta', beta)
    n = 54
    dim = 3
    batchsize = 1024
    
    mc_therm = 50
    mc_steps = 1000
    mc_width = 0.003
    rs = 1.2
    L = (4/3*jnp.pi*n)**(1/3)

    def mean_dist(z):
        z = z - L * jnp.floor(z/L)
        i, j = jnp.triu_indices(n, k=1)
        rij = (jnp.reshape(z, (n, 1, dim)) - jnp.reshape(z, (1, n, dim)))[i, j]
        r = jnp.linalg.norm(rij, axis=-1)
        return jnp.mean(r)

    logprob = make_base(n, dim, L, beta, rs)
    logprob = jax.vmap(logprob)

    key = jax.random.PRNGKey(42)
        
    num_devices = jax.device_count()
    if batchsize % num_devices != 0:
        raise ValueError("Batch size must be divisible by the number of GPU devices. "
                         "Got batch = %d for %d devices now." % (args.batch, num_devices))
    batch_per_device = batchsize // num_devices

    s = jax.random.uniform(key, (num_devices, batch_per_device, n, dim), minval=0, maxval=10)

    keys = jax.random.split(key, num_devices)
    s, keys = shard(s), shard(keys)
    
    mcmc_p = jax.pmap(mcmc, axis_name="p", in_axes=(None, 0, 0, None, None), static_broadcasted_argnums=(0,))
    
    for i in range(mc_therm):
        s, acc_rate = mcmc_p(logprob, s, keys, mc_steps, mc_width) 
        s -= L * jnp.floor(s/L)
        r_mean = jnp.mean(jax.vmap(mean_dist)(s.reshape(batchsize, n, dim)))
        print (i, acc_rate, r_mean, -logprob(s.reshape(batchsize, n, dim)).mean()/n)

test_mcmc()
