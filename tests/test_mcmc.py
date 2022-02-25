from config import * 

from MCMC import mcmc 
from utils import shard 

def test_mcmc():
    beta = 157888.088922572/1500
    n = 14 
    dim = 3
    batchsize = 1024

    mc_steps = 100
    mc_stddev = 1e-4
    rs = 1.44
    L = (4/3*jnp.pi*n)**(1/3)

    def _lj(r):
        epsilon = 2*0.18                   # h2 binding enegy = 0.18 au, here 2 for au to Ry
        sigma = 1.4/rs/(2**(1/6))          # h2 bond length = 1.4 au = 2**(1/6)*sigma
        return 4*epsilon*((sigma/r)**12 - (sigma/r)**6)

    def _soft(r):
        return jnp.exp(-rs*r)
   
    def logprob(z):
        rc = L/2
        z = z - L * jnp.floor(z/L)
        i, j = jnp.triu_indices(n, k=1)
        rij = (jnp.reshape(z, (n, 1, dim)) - jnp.reshape(z, (1, n, dim)))[i, j]
        r = jnp.linalg.norm(rij, axis=-1)
    
        _f = _soft
        f_vmap = jax.vmap(_f)
        return -beta * jnp.sum(f_vmap(r) + f_vmap(2*rc-r) - 2*_f(rc)) 
    
    def mean_dist(z):
        z = z - L * jnp.floor(z/L)
        i, j = jnp.triu_indices(n, k=1)
        rij = (jnp.reshape(z, (n, 1, dim)) - jnp.reshape(z, (1, n, dim)))[i, j]
        r = jnp.linalg.norm(rij, axis=-1)
        return jnp.mean(r)

    force = jax.vmap(jax.grad(logprob))
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
    
    mcmc_p = jax.pmap(mcmc, axis_name="p", in_axes=(None, None, 0, 0, None, None), static_broadcasted_argnums=(0, 1))
    s, acc_rate = mcmc_p(logprob, force, s, keys, mc_steps, mc_stddev) 
    s -= L * jnp.floor(s/L)

    r_mean = jnp.mean(jax.vmap(mean_dist)(s.reshape(batchsize, n, dim)))
    print (acc_rate, r_mean)

    s, acc_rate = mcmc_p(logprob, None, s, keys, mc_steps, mc_stddev) 

    r_mean = jnp.mean(jax.vmap(mean_dist)(s.reshape(batchsize, n, dim)))
    print (acc_rate, r_mean)

test_mcmc()
