import jax
import jax.numpy as jnp
import functools
from typing import Sequence, Optional 
import math
import itertools

shard = jax.pmap(lambda x: x)

def replicate(pytree, num_devices):
    stacked_pytree = jax.tree_map(lambda x: jax.lax.broadcast(x, (num_devices,)), pytree)
    return shard(stacked_pytree)

def logdet_matmul(xs: Sequence[jnp.ndarray],
                  w: Optional[jnp.ndarray] = None) -> jnp.ndarray:
  """Combines determinants and takes dot product with weights in log-domain.
  We use the log-sum-exp trick to reduce numerical instabilities.
  Args:
    xs: FermiNet orbitals in each determinant. Either of length 1 with shape
      (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
      (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
      determinants are factorised into block-diagonals for each spin channel).
    w: weight of each determinant. If none, a uniform weight is assumed.
  Returns:
    sum_i w_i D_i in the log domain, where w_i is the weight of D_i, the i-th
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
  # log-sum-exp trick
  maxlogdet = jnp.max(logdet)
  det = sign_in * det1 * jnp.exp(logdet - maxlogdet)

  if w is None:
    result = jnp.sum(det)
  else:
    result = jnp.dot(det, w)

  sign_out = result/jnp.abs(result)
  log_out = jnp.log(jnp.abs(result)) + maxlogdet
  return sign_out, log_out

def cubic_init(n,spacing):
    K = math.ceil(n**(1 / 3))
    x = jnp.linspace(0, K*spacing, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return jnp.array(position[:n])
