import re 


def _parse(cast, pattern, f):
    res = re.search(pattern, f)
    return None if (res is None) else cast(res.group(1))
    

def parse_filename(f):
    n = _parse(int, 'n_([0-9]*)', f)

    dim = _parse(int, '_dim_([0-9]*)', f)
    rs = _parse(float, '_rs_([0-9]*\.?[0-9]*)', f)
    T = _parse(float, '_T_([0-9]*\.?[0-9]*)', f)

    s =  _parse(int, '_fs_([0-9]*)', f)
    fd =  _parse(int, '_fd_([0-9]*)', f)
    fh1 = _parse(int, '_fh1_([0-9]*)', f)
    fh2 = _parse(int, '_fh2_([0-9]*)', f)

    wd =  _parse(int, '_wd_([0-9]*)', f)
    wh1 = _parse(int, '_wh1_([0-9]*)', f)
    wh2 = _parse(int, '_wh2_([0-9]*)', f)

    clip = _parse(float, '_clip_([0-9]*\.?[0-9]*)', f)
    lr = _parse(float, '_lr_([0-9]*\.?[0-9]*)', f)
    decay = _parse(float, '_decay_([0-9]*\.?[0-9]*)', f)
    eta = _parse(float, '_damping_([0-9]*\.?[0-9]*)', f)
    maxnorm = _parse(float, 'norm_([0-9]*\.?[0-9]*)', f)
    b = _parse(int, '_bs_([0-9]*)', f)

    K = _parse(int, '_K_([0-9]*)', f)
    Nf = _parse(int, '_Nf_([0-9]*)', f)

    acc = _parse(int, '_accsteps_([0-9]*)', f)

    return n, dim, rs, T, s, fd, fh1, fh2, wd, wh1, wh2, b, acc, lr, decay, eta, maxnorm, Nf, clip, K

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


if __name__=='__main__':

    fname = '/data/wanglei/hydrogen/jastrow-tabc-rint-h2/n_14_dim_3_rs_1.2_T_10000_fs_1_fd_3_fh1_64_fh2_16_wd_3_wh1_32_wh2_16_Nf_5_K_4_Gmax_15_kappa_10_mctherm_10_mcsteps_100_100_mcwidth_0.03_0.03_lr_0.05_decay_0.01_damping_0.001_norm_0.001_clip_5_bs_1024_devices_1_accsteps_1/data.txt'
    r = parse_filename(fname)

    print (r)
