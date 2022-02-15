import sys, os
testdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(testdir+"/../src/")

import jax
from jax.config import config   
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np 
import haiku as hk
