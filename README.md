[![Paper](https://img.shields.io/badge/paper-arXiv:2209.06095-B31B1B.svg)](https://arxiv.org/abs/2209.06095)

<div align="center">
<img align="middle" src="_assets/Depositphotos_6284414_XL.jpg" width="500" alt="logo"/>
<h1> Deep variational free energy approach to dense hydrogen</h1>
</div>



## Problem setup

Dense hydrogen consists of an equal number of protons and electrons in a box. We use a normalizing flow to model the proton Boltzmann distribution, and a fermionic neural network to model the electronic ground state. Then, we jointly optimize them to minimize the variational free energy of the whole system. The calculation provides equations of state of dense hydrogen (energy, pressure, entropy, etc at various densities). 



<img align="middle" src="_assets/concept.jpg" width="500" alt="logo"/>



## Requirements

- [JAX](https://github.com/google/jax) 
- [haiku](https://github.com/deepmind/dm-haiku)
- [optax](https://github.com/deepmind/optax)


## Demo run
```bash
python ../src/main.py --T 6000 --rs 1.25 --n 14 --Gmax 15 --flow_steps 1 --flow_depth 3 --flow_h1size 32 --flow_h2size 16 --wfn_depth 3 --wfn_h1size 32 --wfn_h2size 16 --Nf 5 --K 1 --nk 33 --folder ../data/ --walkersize 512 --batchsize 4096 --mc_proton_steps 50 --mc_electron_steps 500 --mc_proton_width 0.02 --mc_electron_width 0.04 --lr_proton 1.0 --lr_electron 1.0 --decay 0.01 --damping_proton 0.001 --damping_electron 0.001 --maxnorm_proton 0.001 --maxnorm_electron 0.001 --clip_factor 5.0 --alpha 0.1 --acc_steps 1 --sr
```


## To cite

```
@article{xie2022hydrogen,
      title={A deep variational free energy approach to dense hydrogen}, 
      author={Hao Xie and Zi-Hang Li and Han Wang and Linfeng Zhang and Lei Wang},
      year={2022},
      eprint={2209.06095},
      archivePrefix={arXiv},
      primaryClass={cond-mat.str-el}
}
```
