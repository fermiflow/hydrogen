import subprocess 
import time 
import re

nickname = 'ones-eloc'

###############################
nlist = [14]
rslist = [1.4]
Tlist = [1500]

dim = 3

Gmax = 15
steps, depth = 1, 2
h1size, h2size = 32, 16
Nf = 5 

lr = 0.01
decay = 1e-2
damping = 1e-3
max_norm = 1e-3
clip_factor = 5.0

mc_steps = 50
mc_stddev = 0.05

batchsize, acc_steps = 1024, 1 
###############################
prog = '../src/main.py'
resfolder = '/data/wanglei/hydrogen/' + nickname  + '/' 

def submitJob(bin,args,jobname,run=False,wait=None):

    #prepare the job file 
    job='''#!/bin/bash -l
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=%s
#SBATCH --output=%s
#SBATCH --error=%s'''%(jobname,jobname+'.log',jobname+'.log')

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
