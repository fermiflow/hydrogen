#!/usr/bin/env python
import sys 

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action='store_true', help="Run or not")
    parser.add_argument("--waitfor", type=int, help="wait for this job for finish")
    input = parser.parse_args()


    #this import might overwrite the above default parameters 
    #########################################################
    import socket, getpass
    machinename = socket.gethostname()
    username = getpass.getuser()
    print ('\n', username, 'on', machinename, '\n')
    if 'ip-10-0-0-26' in machinename:
        from config.aws import * 
    elif 'ln01' in machinename:
        from config.ln01 import * 
    elif 'bright90' in machinename:
        from config.bright90 import * 
    else:
        print ('where am I ?', machinename)
        sys.exit(1)
    #########################################################

    jobdir='../jobs/' + nickname + '/'
    cmd = ['mkdir', '-p', jobdir]
    subprocess.check_call(cmd)

    cmd = ['mkdir', '-p', resfolder]
    subprocess.check_call(cmd)
    
    for rs in rslist:
        for n in nlist:
            for T in Tlist:

                args = {'T':T,
                        'rs':rs,
                        'n':n, 
                        'Gmax':Gmax, 
                        'dim': dim, 
                        'steps': steps,
                        'depth': depth, 
                        'spsize': h1size,
                        'tpsize': h2size,
                        'Nf': Nf, 
                        'K': K,
                        'folder':resfolder,
                        'batch':batchsize,
                        'mc_proton_steps':mc_proton_steps,
                        'mc_electron_steps':mc_electron_steps,
                        'mc_proton_width': mc_proton_width, 
                        'mc_electron_width': mc_electron_width, 
                        'lr': lr, 
                        'decay': decay, 
                        'damping': damping,
                        'max_norm': max_norm,
                        'clip_factor': clip_factor,
                        'acc_steps': acc_steps,
                        }
                jobname = jobdir
                for key, val in args.items():
                    if key != 'folder':
                        jobname +=  str(key) + str(val) + '_'
                jobname = jobname[:-1] 

                jobid = submitJob(prog,args,jobname,run=input.run, wait=input.waitfor)
