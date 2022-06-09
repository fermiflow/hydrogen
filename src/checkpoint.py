import pickle
import os
import re

def find_ckpt_filename(path_or_file):
    if os.path.isfile(path_or_file):
        epoch = int(re.search('epoch_([0-9]*).pkl', path_or_file).group(1))
        return path_or_file, epoch
    files = [f for f in os.listdir(path_or_file) if ('pkl' in f)]
    for f in sorted(files, reverse=True):
        fname = os.path.join(path_or_file, f)
        try:
            with open(fname, "rb") as f:
                pickle.load(f)
            epoch = int(re.search('epoch_([0-9]*).pkl', fname).group(1))
            return fname, epoch
        except (OSError, EOFError):
            print('Error loading checkpoint. Trying next checkpoint...', fname)
    return None, 0

def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


if __name__=='__main__':
    import re 
    f = '/data/wanglei/hydrogen/ff35520-r/n_16_dim_3_rs_1.44_T_1200_fs_1_fd_3_fh1_32_fh2_16_wd_3_wh1_32_wh2_16_Nf_5_K_1_nk_19_Gmax_15_kappa_10_mctherm_10_mcsteps_100_400_mcwidth_0.02_0.04_lr_1_0.05_decay_0.01_damping_0.001_0.001_norm_0.001_0.001_clip_5_alpha_0.1_ws_512_bs_4096_accsteps_1/data.txt'

    path = os.path.dirname(f)
    ckpt_files = [os.path.join(path, f) for f in os.listdir(path) if ('pkl' in f)]
    for c in sorted(ckpt_files): 
        epoch = re.search('epoch_([0-9]*).pkl', c).group(1)
        print (epoch, load_data(c)['params_wfn']['fermi_net']['f'])



