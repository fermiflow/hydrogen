import sys, os
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(here+"/../src/")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import haiku as hk
import numpy as np

from ferminet import FermiNet
from sampler import make_flow
from train import train
from mcmc import mcmc 
import checkpoint
import utils 

import argparse
parser = argparse.ArgumentParser(description="Density estimation for dense hydrogen")

parser.add_argument("--folder", default="../data/", help="the folder to save data")
parser.add_argument("--dataset", default="../data/position.dat",help="The path to training dataset")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate of adam")
parser.add_argument("--batchsize", type=int, default=100, help="batch size")
parser.add_argument("--epoch", type=int, default=10000, help="final epoch")

parser.add_argument("--depth", type=int, default=2, help="FermiNet: network depth")
parser.add_argument("--h1size", type=int, default=16, help="FermiNet: single-particle feature size")
parser.add_argument("--h2size", type=int, default=16, help="FermiNet: two-particle feature size")
parser.add_argument("--Nf", type=int, default=5, help="FermiNet: number of fequencies")

parser.add_argument("--restore_path", default=None, help="checkpoint path or file")
parser.add_argument("--mc_therm", type=int, default=10, help="mcmc therm steps for inference")
parser.add_argument("--mc_steps", type=int, default=100, help="mcmc steps for inference")
parser.add_argument("--mc_width", type=float, default=0.02, help="mcmc width for inference")

args = parser.parse_args()

data = np.loadtxt(args.dataset)
datasize, n, dim = 1000, 64, 3
L = data[-1, -1]
data = data[:, :-3].reshape(datasize, n, dim)
print (data.shape, L)
assert (datasize % args.batchsize == 0)
data -= L * jnp.floor(data/L)
print("Load dataset: %s" % args.dataset)

def make_ferminet(key, depth, h1size, h2size, Nf, L):
    @hk.transform
    def ferminet(x):
        model = FermiNet(depth, h1size, h2size, Nf, L, 0)
        x = model(x)
        return x

    x_dummy = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
    params = ferminet.init(key, x_dummy)
    return params, ferminet

def make_loss(logp_fn):
    def loss_fn(params, x):
        logp = logp_fn(params, x)
        return -jnp.mean(logp)
    return loss_fn

key = jax.random.PRNGKey(42)
params, network = make_ferminet(key, args.depth, args.h1size, args.h2size, args.Nf, L)
logp_fn= make_flow(network, n, dim, L)
logp_fn = jax.vmap(logp_fn, (None, 0), 0)
loss_fn = make_loss(logp_fn)
value_and_grad = jax.value_and_grad(loss_fn)

path = args.folder + "ds_n_%d_dim_%d_lr_%g" % (n, dim, args.lr) 
os.makedirs(path, exist_ok=True)
print("Create directory: %s" % path)

if args.restore_path:
    folder = os.path.dirname(args.restore_path)
    ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path)

    print ('folder:', folder)
    if ckpt_filename is not None:
        print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
        ckpt = checkpoint.load_data(ckpt_filename)
        params = ckpt["params"]
    else:
        raise ValueError("no checkpoint found")
    
    #key, subkey = jax.random.split(key)
    #x = jax.random.uniform(subkey, (args.batchsize, n, dim), minval=0, maxval=L)
    x = data[:args.batchsize] # start from data so we do not suffer from thermaliation issue
    for i in range(args.mc_therm):
        key, subkey = jax.random.split(key)
        x, acc_rate = mcmc(lambda x: logp_fn(params, x), x, subkey, args.mc_steps, args.mc_width) 
        x -= L * jnp.floor(x/L)
        print (i, acc_rate, logp_fn(params,x).mean()/n)
    
    import matplotlib.pyplot as plt
    rdf_data = utils.get_gr(data.reshape(-1, n, dim), L)
    plt.plot(rdf_data[0], rdf_data[1], linestyle='-', c='blue', label='data')

    rdf_model = utils.get_gr(x.reshape(-1, n, dim), L)
    plt.plot(rdf_model[0], rdf_model[1], linestyle='-', c='red', label='red')

    plt.legend()
    plt.show()

else:
    params = train(key, value_and_grad, args.epoch, args.batchsize, params, data, args.lr, path)
