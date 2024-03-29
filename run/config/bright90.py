import subprocess 
import time 
import re
import numpy as np 

nickname = 'ff35520-r-fixk0-backflow-tabc-w-feature-learnf-corrects-real-twist-mala'

###############################
nlist = [14]
rslist = [4.0]
Tlist = [5000]

dim = 3
Gmax = 15

twists = 1
flow_steps, flow_depth, flow_h1size, flow_h2size = 1, 4, 32, 16
wfn_depth, wfn_h1size, wfn_h2size = 4, 32, 16
Nf, K = 1, 1
nk = 57

lr_proton, lr_electron = 0.01, 0.01
damping_proton, damping_electron = 1e-3, 1e-3
maxnorm_proton, maxnorm_electron = 1e-3, 1e-3

decay = 1e-2
clip_factor = 4.0
alpha = 0.05

mc_proton_steps = 100
mc_electron_steps = 500

mc_proton_width = 0.01
mc_electron_width = 0.05

walkersize, batchsize, acc_steps = 1024, 8, 1
###############################
prog = '../src/main.py'
resfolder = '/data/wanglei/hydrogen/' + nickname  + '/' 

def submitJob(bin,args,jobname,logname,run=False,wait=None):

    #prepare the job file 
    job='''#!/bin/bash -l
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --error=%s'''%(jobname,logname,logname)

    if wait is not None:
        dependency ='''
#SBATCH --dependency=afterany:%d\n'''%(wait)
        job += dependency 


    job += '''
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes:"
echo $SLURM_JOB_NODELIST
echo "Using $SLURM_NTASKS_PER_NODE tasks per node"
echo "A total of $SLURM_NTASKS tasks is used"\n'''

    job +='''
num_hosts=$SLURM_JOB_NUM_NODES

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
master=${nodes_array[0]}

ip=$(srun --nodes=1 --ntasks=1 -w $master hostname --ip-address)
port='6379'
ip_address=$ip:$port
'''

    job += '''
echo Job started at `date`\n
for ((i=0; i<$num_hosts;i++))
do 
    node=${nodes_array[$i]}
    echo "host $i node $node CUDA devices $CUDA_VISIBLE_DEVICES"
    srun --nodes=1 --ntasks=1 -w $node '''
    job +='taskset -c 8 python '+ str(bin) + ' '
    for key, val in args.items():
        job += '--'+str(key) + ' '+ str(val) + ' '
    job += '--server_addr=$ip_address --num_hosts=$num_hosts --host_idx=$i &'
    job +='''
done 
wait 

echo Job finished at `date`\n
'''

    #print job
    jobfile = open("jobfile", "w")
    jobfile.write("%s"%job)
    jobfile.close()

    #submit the job 
    if run:
        cmd = ['sbatch', 'jobfile']
        time.sleep(0.1)
    else:
        cmd = ['cat','jobfile']

    subprocess.check_call(cmd)
    return None
