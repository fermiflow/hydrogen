import jax
import jax.numpy as jnp
import numpy as np
from jax import core
import functools
from typing import Sequence, Optional
import math
import itertools

p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))

shard = jax.pmap(lambda x: x)

def replicate(pytree, num_devices):
    dummy_input = jnp.empty(num_devices)
    return jax.pmap(lambda _: pytree)(dummy_input)

def make_different_rng_key_on_all_devices(rng):
    """Makes a different PRNG for all Jax devices and processes."""
    rng = jax.random.fold_in(rng, jax.process_index())
    rng = jax.random.split(rng, jax.local_device_count())
    return shard(rng)

def logdet_matmul(xs: Sequence[jnp.ndarray],
                  logw: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Combines determinants and takes dot product with weights in log-domain.
  We use the log-sum-exp trick to reduce numerical instabilities.
  Args:
    xs: FermiNet orbitals in each determinant. Either of length 1 with shape
      (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
      (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
      determinants are factorised into block-diagonals for each spin channel).
    w: weight of each determinant. If none, a uniform weight is assumed.
  Returns:
    sum_i exp(logw_i) D_i in the log domain, where logw_i is the log-weight of D_i, the i-th
    determinant (or product of the i-th determinant in each spin channel, if
    full_det is not used).
  """
  # Special case to avoid taking log(0) if any matrix is of size 1x1.
  # We can avoid this by not going into the log domain and skipping the
  # log-sum-exp trick.
  det1 = functools.reduce(
      lambda a, b: a * b,
      [x.reshape(-1) for x in xs if x.shape[-1] == 1],
      1
  )

  # Compute the logdet for all matrices larger than 1x1
  sign_in, logdet = functools.reduce(
      lambda a, b: (a[0] * b[0], a[1] + b[1]),
      [jnp.linalg.slogdet(x) for x in xs if x.shape[-1] > 1],
      (1, 0)
  )

  if logw is not None:
    logdet = logw + logdet

  # log-sum-exp trick
  maxlogdet = jnp.max(logdet)
  det = sign_in * det1 * jnp.exp(logdet - maxlogdet)
  result = jnp.sum(det)

  sign_out = result/jnp.abs(result)
  log_out = jnp.log(jnp.abs(result)) + maxlogdet
  return sign_out, log_out

def monkhorstpack(size):
    """Generates a Monkhorst-Pack grid of points of order (n1,n2,n3).
    The points are generated in the open cube
    ]-1/2,1/2[ x ]-1/2,1/2[ x ]-1/2,1/2[/
    same as cell.make_kpts([n1, n2, n3], with_gamma_point=False)/(2*np.pi) in p
    """
    kpts = jnp.swapaxes(jnp.indices(size, jnp.float64), 0, 3)
    kpts = jnp.reshape(kpts, (-1, 3))
    half = jnp.array([0.5, 0.5, 0.5])
    return (kpts + half) / jnp.array(size) - half

def in_pmap(axis_name: Optional[str]) -> bool:
  """Returns whether we are in a pmap with the given axis name."""

  if axis_name is None:
    return False

  try:
    # The only way to know if we are under `jax.pmap` is to check if the
    # function call below raises a `NameError` or not.
    core.axis_frame(axis_name)

    return True

  except NameError:
    return False

def wrap_if_pmap(p_func):
  """Wraps `p_func` to be executed only when inside a `jax.pmap` context."""

  @functools.wraps(p_func)
  def p_func_if_pmap(obj, axis_name):

    return p_func(obj, axis_name) if in_pmap(axis_name) else obj

  return p_func_if_pmap

pmean_if_pmap = wrap_if_pmap(jax.lax.pmean)
psum_if_pmap = wrap_if_pmap(jax.lax.psum)

def get_gr(x, L, bins=100): 
    batchsize, n, dim = x.shape[0], x.shape[1], x.shape[2]
    
    i,j = np.triu_indices(n, k=1)
    rij = (np.reshape(x, (-1, n, 1, dim)) - np.reshape(x, (-1, 1, n, dim)))[:,i,j]
    rij = rij - L*np.rint(rij/L)
    dist = np.linalg.norm(rij, axis=-1) # (batchsize, n*(n-1)/2)
   
    hist, bin_edges = np.histogram(dist.reshape(-1,), range=[0, L/2], bins=bins)
    dr = bin_edges[1] - bin_edges[0]
    hist = hist*2/(n * batchsize)

    rmesh = np.arange(hist.shape[0])*dr
    
    h_id = 4/3*np.pi*n/(L**3)* ((rmesh+dr)**3 - rmesh**3 )
    return rmesh, hist/h_id


