import subprocess 
import time 
import re

nickname = 'walker-uniform-geminal'

###############################
nlist = [38]
rslist = [1.44]
Tlist = [1200]

dim = 3
Gmax = 15

flow_steps, flow_depth, flow_h1size, flow_h2size = 1, 3, 64, 16
wfn_depth, wfn_h1size, wfn_h2size = 3, 32, 16
Nf, K = 5, 4

lr_proton, lr_electron = 1.0, 0.05
decay = 1e-2
damping = 1e-3
max_norm = 1e-3
clip_factor = 5.0

mc_proton_steps = 100
mc_electron_steps = 400

mc_proton_width = 0.02
mc_electron_width = 0.04

walkersize = 256 
batchsize, acc_steps = 1024, 1
###############################
prog = '../src/main.py'
resfolder = '/data/wanglei/hydrogen/' + nickname  + '/' 

def submitJob(bin,args,jobname,logname,run=False,wait=None):

    #prepare the job file 
    job='''#!/bin/bash -l
#SBATCH --partition=v100
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
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
echo "A total of $SLURM_NTASKS tasks is used"
echo "CUDA devices $CUDA_VISIBLE_DEVICES"\n'''

    job += '''
echo Job started at `date`\n'''
    job +='python '+ str(bin) + ' '
    for key, val in args.items():
        job += '--'+str(key) + ' '+ str(val) + ' '
    job += '--sr' 
    job += '''
echo Job finished at `date`\n'''

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
