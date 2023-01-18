import jax 
import jax.numpy as jnp
import optax
import haiku as hk
import os
from typing import NamedTuple
import itertools

import checkpoint

class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState

def train(key, value_and_grad, num_epochs, batchsize, params, data, lr, path):
    
    assert (len(data)%batchsize==0)

    @jax.jit
    def step(i, state, x):
        value, grad = value_and_grad(state.params, x)

        updates, opt_state = optimizer.update(grad, state.opt_state)
        params = optax.apply_updates(state.params, updates)

        return TrainingState(params, opt_state), value
    
    optimizer = optax.adam(lr)
    init_opt_state = optimizer.init(params)
    state = TrainingState(params, init_opt_state)

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w", buffering=1, newline="\n")
    itercount = itertools.count()
    for epoch in range(num_epochs+1):
        key, subkey = jax.random.split(key)
        data = jax.random.permutation(subkey, data)

        total_loss = 0.0
        counter = 0 
        for batch_index in range(0, len(data), batchsize):
            x = data[batch_index:batch_index+batchsize]

            state, loss = step(next(itercount), 
                               state, 
                               x)
            total_loss += loss
            counter += 1
    
        f.write( ("%6d" + "  %.6f" + "\n") % (epoch, total_loss/counter) )

        if epoch % 100 == 0:
            ckpt = {"params": state.params,
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return state.params
