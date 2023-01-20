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

def train(key, loss_fn, num_epochs, batchsize, params, data, lr, path):

    train_set = data[:900]
    test_set = data[900:]
    
    assert (len(train_set)%batchsize==0)
    assert (len(test_set)%batchsize==0)

    value_and_grad = jax.value_and_grad(loss_fn)

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
        train_set = jax.random.permutation(subkey, train_set)

        train_loss = 0.0
        counter = 0 
        for batch_index in range(0, len(train_set), batchsize):
            x = train_set[batch_index:batch_index+batchsize]
            state, loss = step(next(itercount), 
                               state, 
                               x)
            train_loss += loss
            counter += 1
        train_loss = train_loss/counter

        test_loss = 0.0
        counter = 0
        for batch_index in range(0, len(test_set), batchsize):
            x = test_set[batch_index:batch_index+batchsize]
            test_loss += loss_fn(state.params, x)
            counter += 1
        test_loss = test_loss/counter

        f.write( ("%6d" + "  %.6f"*2 + "\n") % (epoch, train_loss, test_loss) )

        if epoch % 100 == 0:
            ckpt = {"params": state.params,
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return state.params
