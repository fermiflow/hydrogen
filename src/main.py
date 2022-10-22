import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import os
import time

import checkpoint
from vmc import sample_s_and_x, make_loss
from utils import shard, replicate, p_split
from utils import monkhorstpack
from utils import make_different_rng_key_on_all_devices

print("jax.__version__:", jax.__version__)

import argparse
parser = argparse.ArgumentParser(description="Variational free-energy for dense hydrogen")

# multihost setup
parser.add_argument("--num_hosts", type=int, default=1, help="num of hosts")
parser.add_argument("--server_addr", type=str, default="10.1.1.3:9999", help="server ip addr, will be used only when num_hosts>1")
parser.add_argument("--host_idx", type=int, default=0, help="index of current host, will be used onlly when num_hosts>1")

parser.add_argument("--folder", default="../data/", help="the folder to save data")
parser.add_argument("--restore_path", default=None, help="checkpoint path or file")

# physical parameters.
parser.add_argument("--n", type=int, default=14, help="total number of electrons == # of protons")
parser.add_argument("--dim", type=int, default=3, help="spatial dimension")
parser.add_argument("--rs", type=float, default=1.4, help="rs")
parser.add_argument("--T", type=float, default=1000.0, help="temperature in  Kelvin")

# normalizing flow.
parser.add_argument("--flow_steps", type=int, default=2, help="FermiNet: transformation steps")
parser.add_argument("--flow_depth", type=int, default=2, help="FermiNet: network depth")
parser.add_argument("--flow_h1size", type=int, default=16, help="FermiNet: single-particle feature size")
parser.add_argument("--flow_h2size", type=int, default=16, help="FermiNet: two-particle feature size")

parser.add_argument("--wfn_depth", type=int, default=2, help="FermiNet: network depth")
parser.add_argument("--wfn_h1size", type=int, default=16, help="FermiNet: single-particle feature size")
parser.add_argument("--wfn_h2size", type=int, default=16, help="FermiNet: two-particle feature size")

parser.add_argument("--Nf", type=int, default=5, help="FermiNet: number of fequencies")
parser.add_argument("--K", type=int, default=4, help="FermiNet: number of dets")
parser.add_argument("--nk", type=int, default=None, help="FermiNet: number of plane wave basis")

# parameters relevant to th Ewald summation of Coulomb interaction.
parser.add_argument("--Gmax", type=int, default=15, help="k-space cutoff in the Ewald summation of Coulomb potential")
parser.add_argument("--kappa", type=int, default=10, help="screening parameter (in unit of 1/L) in Ewald summation")

# MCMC.
parser.add_argument("--mc_therm", type=int, default=10, help="MCMC thermalization steps")
parser.add_argument("--mc_proton_steps", type=int, default=50, help="MCMC update steps")

parser.add_argument("--mc_electron_steps", type=int, default=50, help="MCMC update steps")

parser.add_argument("--mc_proton_width", type=float, default=0.01, help="standard deviation of the Gaussian proposal in MCMC update")
parser.add_argument("--mc_electron_width", type=float, default=0.05, help="standard deviation of the Gaussian proposal in MCMC update")

# optimizer parameters.
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--sr", action='store_true',  help="use the second-order stochastic reconfiguration optimizer")
group.add_argument("--lr_adam", type=float, default=1e-2, help="initial learning rate of adam")

parser.add_argument("--lr_proton", type=float, default=1e-2, help="initial learning rate")
parser.add_argument("--lr_electron", type=float, default=1e-2, help="initial learning rate")
parser.add_argument("--decay", type=float, default=1e-2, help="learning rate decay")
parser.add_argument("--damping_proton", type=float, default=1e-3, help="damping")
parser.add_argument("--damping_electron", type=float, default=1e-3, help="damping")
parser.add_argument("--maxnorm_proton", type=float, default=1e-3, help="gradnorm maximum")
parser.add_argument("--maxnorm_electron", type=float, default=1e-3, help="gradnorm maximum")
parser.add_argument("--clip_factor", type=float, default=5.0, help="clip factor for gradient")
parser.add_argument("--alpha", type=float, default=0.1, help="mixing of new fisher matrix")

# training parameters.
parser.add_argument("--twists", type=int, default=4, help="twist in each direction")
parser.add_argument("--walkersize", type=int, default=16, help="walker per twist")
parser.add_argument("--batchsize", type=int, default=8, help="batch size per walker")
parser.add_argument("--acc_steps", type=int, default=4, help="gradient accumulation steps")
parser.add_argument("--epoch", type=int, default=100000, help="final epoch")

args = parser.parse_args()

if args.num_hosts > 1:
    jax.distributed.initialize(args.server_addr, args.num_hosts, args.host_idx)

print("Cluster connected with totally %d GPUs" % jax.device_count())
print("This is process %d with %d local GPUs." % (jax.process_index(), jax.local_device_count()))

num_devices = jax.local_device_count()
num_hosts = jax.device_count() // num_devices

key = jax.random.PRNGKey(42)

n, dim = args.n, args.dim
assert (n%2==0)
assert (dim==3)
if args.nk is None:
    nk = n//2 # number of plane wave basis in the envelope function
else:
    nk = args.nk
    assert(nk >= n//2)

beta = 157887.66/args.T # inverse temperature in unit of 1/Ry
print ("temperature in Rydberg unit:", 1.0/beta)

L = (4/3*jnp.pi*n)**(1/3)

print("n = %d, dim = %d, L = %f, rs = %f" % (n, dim, L, args.rs))

####################################################################################

print("\n========== Initialize single-particle orbitals ==========")

from orbitals import sp_orbitals
sp_indices, Es = sp_orbitals(dim)
sp_indices, Es = jnp.array(sp_indices), jnp.array(Es)
print("beta = %f, Ef = %d"% (beta, Es[n//2-1]))

####################################################################################

print("\n========== Setup twists ==========")

if (args.twists**dim) % (num_devices * num_hosts) != 0:
    raise ValueError("Total number of twists size must be divisible by the number of GPU devices. "
                     "Got twists = %d for %d devices now." % (args.twists**dim, num_devices*num_hosts))
device_twist_size = (args.twists**dim) // (num_devices * num_hosts)

twist = monkhorstpack([args.twists]*dim) # (args.twists**dim, dim) = (T, dim)
k = 2*jnp.pi/L * (sp_indices[None, :nk, :] + twist[:, None, :])  #(T, nk, dim)
k = jnp.reshape(k, (num_hosts, num_devices, device_twist_size, nk, dim)) # (H, D, T/(H*D), nk, dim)
k = k[jax.process_index()]
k = shard(k)

print ("total number of twists", (args.twists**dim))

####################################################################################
print("\n========== Initialize relevant quantities for Ewald summation ==========")

from potential import kpoints, Madelung
G = kpoints(dim, args.Gmax)
Vconst = n * args.rs/L * Madelung(dim, args.kappa, G) 
print("(scaled) Vconst:", Vconst/(n*args.rs/L)) 

####################################################################################

print("\n========== Initialize normalizing flow ==========")

import haiku as hk
from ferminet import FermiNet
def forward_fn(x):
    for _ in range(args.flow_steps):
        model = FermiNet(args.flow_depth, args.flow_h1size, args.flow_h2size, args.Nf, L, 0)
        x = model(x)
    return x
network_flow = hk.transform(forward_fn)
x_dummy = jax.random.uniform(key, (n, dim), minval=0., maxval=L)
params_flow = network_flow.init(key, x_dummy)

raveled_params_flow, _ = ravel_pytree(params_flow)
print("#parameters in the flow model: %d" % raveled_params_flow.size)

from sampler import make_flow, make_classical_score
logprob_novmap = make_flow(network_flow, n, dim, L)
logprob = jax.vmap(jax.vmap(logprob_novmap, (None, 0), 0), (None, 0), 0)

####################################################################################

print("\n========== Initialize wavefunction ==========")
def forward_fn(x, k):
    model = FermiNet(args.wfn_depth, args.wfn_h1size, args.wfn_h2size, args.Nf, L, args.K)
    return model(x, k)
network_wfn = hk.transform(forward_fn)
sx_dummy = jax.random.uniform(key, (2*n, dim), minval=0., maxval=L)
k_dummy = jax.random.uniform(key, (nk, dim), minval=0, maxval=2*jnp.pi/L)
params_wfn = network_wfn.init(key, sx_dummy, k_dummy)

raveled_params_wfn, _ = ravel_pytree(params_wfn)
print("#parameters in the wavefunction model: %d" % raveled_params_wfn.size)

from logpsi import make_logpsi, make_logpsi_grad_laplacian, \
                   make_logpsi2, make_quantum_score
logpsi_novmap = make_logpsi(network_wfn)
logpsi2 = make_logpsi2(logpsi_novmap)

####################################################################################

print("\n========== Initialize optimizer ==========")

import optax
if args.sr:
    classical_score_fn = make_classical_score(logprob_novmap)
    quantum_score_fn = make_quantum_score(logpsi_novmap)
    from sr import hybrid_fisher_sr
    fishers_fn, optimizer = hybrid_fisher_sr(classical_score_fn, quantum_score_fn,
            args.lr_proton, args.lr_electron, args.decay, args.damping_proton, args.damping_electron, args.maxnorm_proton, args.maxnorm_electron)
    print("Optimizer hybrid_fisher_sr: lr = %g, %g, decay = %g, damping = %g %g, maxnorm = %g %g." %
            (args.lr_proton, args.lr_electron, args.decay, args.damping_proton, args.damping_electron, args.maxnorm_proton, args.maxnorm_electron))
else:
    optimizer = optax.chain(
                optax.scale_by_adam(),
                optax.scale_by_schedule(lambda t: args.lr_adam/(1+args.decay*t)),
                optax.scale(-1.))

    print("Optimizer adam: lr = %g." % args.lr_adam)

####################################################################################

print("\n========== Checkpointing ==========")

path = args.folder + "n_%d_dim_%d_t_%d_rs_%g_T_%g" % (n, dim, args.twists, args.rs, args.T) \
                   + "_fs_%d_fd_%d_fh1_%d_fh2_%d" % \
                      (args.flow_steps, args.flow_depth, args.flow_h1size, args.flow_h2size) \
                   + "_wd_%d_wh1_%d_wh2_%d_Nf_%d_K_%d_nk_%d" % \
                      (args.wfn_depth, args.wfn_h1size, args.wfn_h2size, args.Nf, args.K, nk) \
                   + "_Gmax_%d_kappa_%d" % (args.Gmax, args.kappa) \
                   + "_mctherm_%d_mcsteps_%d_%d_mcwidth_%g_%g" % (args.mc_therm, args.mc_proton_steps, args.mc_electron_steps, args.mc_proton_width, args.mc_electron_width) \
                   + ("_lr_%g_%g_decay_%g_damping_%g_%g_norm_%g_%g" % (args.lr_proton, args.lr_electron, args.decay, args.damping_proton, args.damping_electron, args.maxnorm_proton, args.maxnorm_electron) \
                        if args.sr else "_lr_%g_decay_%g" % (args.lr_adam, args.decay)) \
                   + "_clip_%g_alpha_%g"%(args.clip_factor, args.alpha) \
                   + "_ws_%d_bs_%d_accsteps_%d" % (args.walkersize, args.batchsize, args.acc_steps)

os.makedirs(path, exist_ok=True)
print("Create directory: %s" % path)

ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path or path)

if ckpt_filename is not None:
    print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
    ckpt = checkpoint.load_data(ckpt_filename)
    s, x, params_flow, params_wfn, opt_state = \
        ckpt["s"], ckpt["x"], ckpt["params_flow"], ckpt["params_wfn"], ckpt["opt_state"]
   
    s = jnp.reshape(s, (num_devices, device_twist_size, args.walkersize, n, dim))
    x = jnp.reshape(x, (num_devices, device_twist_size, args.walkersize, args.batchsize, n, dim))

else:
    print("No checkpoint file found. Start from scratch.")
    opt_state = optimizer.init((params_flow, params_wfn))

if epoch_finished == 0:
    print("Initialize coordinate samples...")

    key, subkey = jax.random.split(key)
    subkey = jax.random.fold_in(subkey, jax.process_index())

    key, key_proton, key_electron = jax.random.split(subkey, 3)

    s = jax.random.uniform(key_proton, (num_devices, device_twist_size, args.walkersize, n, dim), minval=0., maxval=L)
    x = jax.random.uniform(key_electron, (num_devices, device_twist_size, args.walkersize, args.batchsize, n, dim), minval=0., maxval=L)

s, x = shard(s), shard(x)
params_flow, params_wfn = replicate((params_flow, params_wfn), num_devices)
keys = make_different_rng_key_on_all_devices(key)

#rerun thermalization steps since we regenerated s and x samples
if epoch_finished == 0:
    for i in range(args.mc_therm):
        print("---- thermal step %d ----" % (i+1))
        keys, s, x, ar_s, ar_x = sample_s_and_x(keys,
                                   logprob, s, params_flow,
                                   logpsi2, x, params_wfn,
                                   args.mc_proton_steps, args.mc_electron_steps, args.mc_proton_width, args.mc_electron_width, L, k)
        print ('acc, entropy:', jnp.mean(ar_s), jnp.mean(ar_x), -jax.pmap(logprob)(params_flow, s).mean()/n)
    print("keys shape:", keys.shape, "\t\ttype:", type(keys))
    print("x shape:", x.shape, "\t\ttype:", type(x))

####################################################################################

print("\n========== Training ==========")

logpsi, logpsi_grad_laplacian = make_logpsi_grad_laplacian(logpsi_novmap)

observable_and_lossfn = make_loss(logprob, logpsi, logpsi_grad_laplacian,
                                  args.kappa, G, L, args.rs, Vconst, beta, args.clip_factor)

from functools import partial

@partial(jax.pmap, axis_name="p",
        in_axes=(0, 0, None, 0, 0, 0, 0, 0, 0, 0) +
                ((0, 0, None, None) if args.sr else (None, None, None, None)),
        out_axes=(0, 0, None, 0, 0, 0) +
                ((0, 0) if args.sr else (None, None)),
        static_broadcasted_argnums=(12,13) if args.sr else (10, 11, 12, 13),
        donate_argnums=(3, 4, 5))
def update(params_flow, params_wfn, opt_state, k, s, x, key, data_acc, grads_acc, classical_score_acc,
        classical_fisher_acc, quantum_fisher_acc, final_step, mix_fisher):

    data, classical_lossfn, quantum_lossfn = observable_and_lossfn(
            params_flow, params_wfn, k, s, x, key)

    grad_params_flow, classical_score = jax.jacrev(classical_lossfn)(params_flow)
    grad_params_wfn = jax.grad(quantum_lossfn)(params_wfn)
    grads = grad_params_flow, grad_params_wfn
    grads, classical_score = jax.lax.pmean((grads, classical_score), axis_name="p")
    
    data_acc, grads_acc, classical_score_acc = jax.tree_util.tree_map(lambda acc, i: acc + i, 
                                                       (data_acc, grads_acc, classical_score_acc),  
                                                       (data, grads, classical_score))

    if args.sr:
        classical_fisher, quantum_fisher = fishers_fn(params_flow, params_wfn, k, s, x, opt_state)

        if mix_fisher:
            i= opt_state["acc"]
            #account for the same mixing factor for all acc steps
            classical_fisher_acc = (1-args.alpha)**(1/args.acc_steps)*classical_fisher_acc \
                + args.alpha/(1-args.alpha)**((args.acc_steps-1-i)/args.acc_steps)*classical_fisher/args.acc_steps
            quantum_fisher_acc = (1-args.alpha)*quantum_fisher_acc + args.alpha/(1-args.alpha)**(args.acc_steps-1-i)*quantum_fisher/args.acc_steps
        else:
            classical_fisher_acc += classical_fisher/args.acc_steps
            quantum_fisher_acc += quantum_fisher/args.acc_steps

    if final_step:
        data_acc, grads_acc, classical_score_acc = \
                jax.tree_map(lambda acc: acc / args.acc_steps,
                             (data_acc, grads_acc, classical_score_acc))

        updates, opt_state = optimizer.update(grads_acc, opt_state,
                                params=(classical_fisher_acc, quantum_fisher_acc) if args.sr else None)
        params_flow, params_wfn = optax.apply_updates((params_flow, params_wfn), updates)
    
    return params_flow, params_wfn, opt_state, data_acc, grads_acc, classical_score_acc,\
            classical_fisher_acc, quantum_fisher_acc

if args.sr:
    classical_fisher_acc = replicate(jnp.zeros((raveled_params_flow.size, raveled_params_flow.size)), num_devices)
    quantum_fisher_acc = replicate(jnp.zeros((raveled_params_wfn.size, raveled_params_wfn.size)), num_devices)
else:
    classical_fisher_acc = quantum_fisher_acc = None
mix_fisher = False

time_of_last_ckpt = time.time()
log_filename = os.path.join(path, "data.txt")
if jax.process_index() == 0:
    f = open(log_filename, "w" if epoch_finished == 0 else "a",
            buffering=1, newline="\n")
    if os.path.getsize(log_filename)==0:
        f.write("epoch f f_err e e_err k k_err vpp vpp_err vep vep_err vee vee_err p p_err s s_err acc_s acc_x\n")

for i in range(epoch_finished + 1, args.epoch + 1):

    data_acc = replicate({"K": 0., "K2": 0.,
                          "Vpp": 0., "Vpp2": 0.,
                          "Vep": 0., "Vep2": 0.,
                          "Vee": 0., "Vee2": 0.,
                          "P": 0., "P2": 0.,
                          "E": 0., "E2": 0.,
                          "F": 0., "F2": 0.,
                          "S": 0., "S2": 0.}, num_devices)
    grads_acc = shard(jax.tree_map(jnp.zeros_like, (params_flow, params_wfn)))
    classical_score_acc = shard(jax.tree_map(jnp.zeros_like, params_flow)) 
    ar_s_acc = shard(jnp.zeros(num_devices))
    ar_x_acc = shard(jnp.zeros(num_devices))

    for acc in range(args.acc_steps):
        keys, s, x, ar_s, ar_x = sample_s_and_x(keys,
                                               logprob, s, params_flow,
                                               logpsi2, x, params_wfn,
                                               args.mc_proton_steps, args.mc_electron_steps, args.mc_proton_width, args.mc_electron_width, L, k)
        ar_s_acc += ar_s/args.acc_steps
        ar_x_acc += ar_x/args.acc_steps

        final_step = (acc == args.acc_steps - 1)
        if args.sr:
            opt_state["acc"] = acc

        params_flow, params_wfn, opt_state, data_acc, grads_acc, classical_score_acc,\
        classical_fisher_acc, quantum_fisher_acc\
            = update(params_flow, params_wfn, opt_state, k, s, x, keys, 
                     data_acc, grads_acc, classical_score_acc, classical_fisher_acc, quantum_fisher_acc, final_step, mix_fisher)
        
        # if we have finished a step, we can mix fisher from now on
        if final_step: mix_fisher = True

    data = jax.tree_map(lambda x: x[0], data_acc)
    ar_s = ar_s_acc[0] 
    ar_x = ar_x_acc[0] 

    K, K2, Vpp, Vpp2, Vep, Vep2, Vee, Vee2, P, P2, E, E2, F, F2, S, S2 = \
            data["K"], data["K2"], \
            data["Vpp"], data["Vpp2"],\
            data["Vep"], data["Vep2"],\
            data["Vee"], data["Vee2"],\
            data["P"], data["P2"], \
            data["E"], data["E2"], \
            data["F"], data["F2"], \
            data["S"], data["S2"]

    T = args.twists**dim
    W = T *args.walkersize
    B = W *args.batchsize

    K_std = jnp.sqrt((K2- K**2) / (B*args.acc_steps))
    Vpp_std = jnp.sqrt((Vpp2- Vpp**2) / (W*args.acc_steps))
    Vep_std = jnp.sqrt((Vep2- Vep**2) / (B*args.acc_steps))
    Vee_std = jnp.sqrt((Vee2- Vee**2) / (B*args.acc_steps))
    P_std = jnp.sqrt((P2- P**2) / (B*args.acc_steps))
    E_std = jnp.sqrt((E2- E**2) / (B*args.acc_steps))
    F_std = jnp.sqrt((F2- F**2) / (B*args.acc_steps))
    S_std = jnp.sqrt((S2- S**2) / (W*args.acc_steps))
    
    if jax.process_index() == 0:
        # Note the quantities with energy dimension has a prefactor 1/rs^2
        print("iter: %04d" % i,
                "F:", F/args.rs**2, "F_std:", F_std/args.rs**2,
                "E:", E/args.rs**2, "E_std:", E_std/args.rs**2,
                "K:", K/args.rs**2, "K_std:", K_std/args.rs**2,
                "S:", S, "S_std:", S_std,
                "accept_rate:", ar_s, ar_x
                )
        f.write( ("%6d" + "  %.6f"*16 + "  %.4f"*2 + "\n") % (i,
                                                    F/n/args.rs**2, F_std/n/args.rs**2,
                                                    E/n/args.rs**2, E_std/n/args.rs**2,
                                                    K/n/args.rs**2, K_std/n/args.rs**2,
                                                    Vpp/n/args.rs**2, Vpp_std/n/args.rs**2,
                                                    Vep/n/args.rs**2, Vep_std/n/args.rs**2,
                                                    Vee/n/args.rs**2, Vee_std/n/args.rs**2, # Ry
                                                    P/args.rs**2, P_std/args.rs**2, # GPa 
                                                    S/n, S_std/n, 
                                                    ar_s, ar_x) )

        if time.time() - time_of_last_ckpt > 3600:
            ckpt = {"s": s, 
                    "x": x,
                    "params_flow": jax.tree_map(lambda x: x[0], params_flow),
                    "params_wfn": jax.tree_map(lambda x: x[0], params_wfn),
                    "opt_state": opt_state
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(i))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)
            time_of_last_ckpt = time.time()

if jax.process_index() == 0:
    f.close()
