

## Requirements

- [JAX](https://github.com/google/jax) with Nvidia GPU support
- [haiku](https://github.com/deepmind/dm-haiku)
- [optax](https://github.com/deepmind/optax)

## Demo run
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python ../src/main.py --T 1200 --rs 1.44 --n 16 --Gmax 15 --dim 3 --flow_steps 1 --flow_depth 3 --flow_h1size 32 --flow_h2size 16 --wfn_depth 3 --wfn_h1size 32 --wfn_h2size 16 --Nf 5 --K 4 --nk 33 --folder /data/wanglei/hydrogen/ff35520-r-fixk0-backflow/ --walkersize 512 --batchsize 4096 --mc_proton_steps 50 --mc_electron_steps 500 --mc_proton_width 0.02 --mc_electron_width 0.04 --lr_proton 1.0 --lr_electron 1.0 --decay 0.01 --damping_proton 0.001 --damping_electron 0.001 --maxnorm_proton 0.001 --maxnorm_electron 0.001 --clip_factor 5.0 --alpha 0.1 --acc_steps 1 --sr
```
