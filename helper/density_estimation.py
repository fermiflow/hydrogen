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
from mala import mcmc 
import checkpoint
import utils 

import argparse
parser = argparse.ArgumentParser(description="Density estimation for dense hydrogen")

parser.add_argument("--folder", default="../data/", help="the folder to save data")
parser.add_argument("--dataset", default="../data/position.dat",help="The path to training dataset")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate of adam")
parser.add_argument("--batchsize", type=int, default=10, help="batch size")
parser.add_argument("--epoch", type=int, default=10000, help="final epoch")

parser.add_argument("--steps", type=int, default=1, help="FermiNet: network steps")
parser.add_argument("--depth", type=int, default=4, help="FermiNet: network depth")
parser.add_argument("--h1size", type=int, default=32, help="FermiNet: single-particle feature size")
parser.add_argument("--h2size", type=int, default=16, help="FermiNet: two-particle feature size")
parser.add_argument("--Nf", type=int, default=5, help="FermiNet: number of fequencies")

parser.add_argument("--restore_path", default=None, help="checkpoint path or file")
parser.add_argument("--mc_therm", type=int, default=10, help="mcmc therm steps for inference")
parser.add_argument("--mc_steps", type=int, default=100, help="mcmc steps for inference")
parser.add_argument("--mc_width", type=float, default=0.02, help="mcmc width for inference")
parser.add_argument("--hotinit", action='store_true',  help="hot initilization")

args = parser.parse_args()

data = np.loadtxt(args.dataset)
datasize, n, dim = 1000, 64, 3
L = data[-1, -1]
data = data[:, :-3].reshape(datasize, n, dim)
print (data.shape, L)
assert (datasize % args.batchsize == 0)
data -= L * jnp.floor(data/L)
print("Load dataset: %s" % args.dataset)

def make_ferminet(key, steps, depth, h1size, h2size, Nf, L):
    @hk.transform
    def ferminet(x):
        for _ in range(steps):
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
params, network = make_ferminet(key, args.steps, args.depth, args.h1size, args.h2size, args.Nf, L)
logp_fn= make_flow(network, n, dim, L)
force_fn = jax.vmap(jax.grad(logp_fn, argnums=1), (None, 0), 0)
logp_fn = jax.vmap(logp_fn, (None, 0), 0)

loss_fn = make_loss(logp_fn)

path = args.folder + "ds_n_%d_dim_%d_s_%g_d_%g_h1_%g_h2_%g_lr_%g" % (n, dim, args.steps, args.depth, args.h1size, args.h2size, args.lr) 
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
    
    if args.hotinit:
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, (args.batchsize, n, dim), minval=0, maxval=L)
    else:
        x = data[:args.batchsize] # start from data so we do not suffer from thermaliation issue
     
    print (logp_fn(params, x))
    update = 0.5*force_fn(params, x) * args.mc_width**2 + args.mc_width * jax.random.normal(key, x.shape)
    print (update, update.min(), update.max())
    
    rdf_data = utils.get_gr(data.reshape(-1, n, dim), L)
    import h5py 
    h5_filename = ckpt_filename.replace('.pkl', '.h5')
    with h5py.File(h5_filename, "w") as f:
        f.create_dataset("data", data=rdf_data)

    for i in range(args.mc_therm):
        key, subkey = jax.random.split(key)
        x, acc_rate = mcmc(lambda x: logp_fn(params, x), 
                           lambda x: force_fn(params, x), 
                           x, subkey, args.mc_steps, args.mc_width) 
        x -= L * jnp.floor(x/L)
        print (i, acc_rate, logp_fn(params,x).mean()/n)

        rdf_model = utils.get_gr(x.reshape(-1, n, dim), L)
        with h5py.File(h5_filename, "a") as f:
            f.create_dataset("mcstep_%g"%i, data=rdf_model)
            f.create_dataset("acrate_%g"%i, data=acc_rate)
else:
    params = train(key, loss_fn, args.epoch, args.batchsize, params, data, args.lr, path)
