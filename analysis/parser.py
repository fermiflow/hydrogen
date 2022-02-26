import re 


def _parse(cast, pattern, f):
    res = re.search(pattern, f)
    return None if (res is None) else cast(res.group(1))
    

def parse_filename(f):
    n = _parse(int, 'n_([0-9]*)', f)

    dim = _parse(int, '_dim_([0-9]*)', f)
    rs = _parse(float, '_rs_([0-9]*\.?[0-9]*)', f)
    T = _parse(float, '_T_([0-9]*\.?[0-9]*)', f)

    s =  _parse(int, '_steps_([0-9]*)', f)
    d =  _parse(int, '_depth_([0-9]*)', f)
    h1 = _parse(int, '_spsize_([0-9]*)', f)
    h2 = _parse(int, '_tpsize_([0-9]*)', f)

    clip = _parse(float, '_clip_([0-9]*\.?[0-9]*)', f)
    lr = _parse(float, '_lr_([0-9]*\.?[0-9]*)', f)
    decay = _parse(float, '_decay_([0-9]*\.?[0-9]*)', f)
    eta = _parse(float, '_damping_([0-9]*\.?[0-9]*)', f)
    maxnorm = _parse(float, 'norm_([0-9]*\.?[0-9]*)', f)
    b = _parse(int, '_bs_([0-9]*)', f)

    K = _parse(int, '_K_([0-9]*)', f)
    Nf = _parse(int, '_Nf_([0-9]*)', f)

    acc = _parse(int, '_accsteps_([0-9]*)', f)

    return n, dim, rs, T, s, d, h1, h2, b, acc, lr, decay, eta, maxnorm, Nf, clip, K

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


if __name__=='__main__':

    fname = '/data/wanglei/hydrogen/jastrow-tabc-soft/n_14_dim_3_rs_1.44_T_1200_steps_1_depth_3_spsize_32_tpsize_16_Nf_5_K_4_Gmax_15_kappa_10_mctherm_10_mcsteps_100_100_mcwidth_0.01_0.05_lr_0.01_decay_0.01_damping_0.001_norm_0.001_clip_5_bs_1024_devices_1_accsteps_1/data.txt'
    r = parse_filename(fname)

    print (r)
