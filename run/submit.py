#!/usr/bin/env python
import sys , os 

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
                        'flow_steps': flow_steps,
                        'flow_depth': flow_depth, 
                        'flow_h1size': flow_h1size,
                        'flow_h2size': flow_h2size,
                        'wfn_depth': wfn_depth, 
                        'wfn_h1size': wfn_h1size,
                        'wfn_h2size': wfn_h2size,
                        'Nf': Nf, 
                        'K': K,
                        'nk': nk,
                        'folder':resfolder,
                        'walkersize': walkersize,
                        'batchsize':batchsize,
                        'mc_proton_steps':mc_proton_steps,
                        'mc_electron_steps':mc_electron_steps,
                        'mc_proton_width': mc_proton_width, 
                        'mc_electron_width': mc_electron_width, 
                        'lr_proton': lr_proton, 
                        'lr_electron': lr_electron, 
                        'decay': decay, 
                        'damping_proton': damping_proton,
                        'damping_electron': damping_electron,
                        'maxnorm_proton': maxnorm_proton,
                        'maxnorm_electron': maxnorm_electron,
                        'clip_factor': clip_factor,
                        'alpha': alpha, 
                        'acc_steps': acc_steps,
                        }
                
                try:
                    args['lr_adam'] = lr_adam
                except NameError:
                    args['sr'] = ' '
 
                logname = jobdir
                for key, val in args.items():
                    if key != 'folder':
                        k = str(key)
                        if '_' in k:
                            k = ''.join([s[0] for s in k.split('_')])
                        logname +=  k + str(val) + '_'
                logname = logname[:-1] + '.log'

                jobname = os.path.basename(os.path.dirname(logname))

                jobid = submitJob(prog,args,jobname,logname,run=input.run, wait=input.waitfor)
