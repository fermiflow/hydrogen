run with
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python ../src/main.py --T 1200 --rs 1.44 --n 38 --Gmax 15 --dim 3 --flow_steps 1 --flow_depth 3 --flow_h1size 64 --flow_h2size 16 --wfn_depth 3 --wfn_h1size 32 --wfn_h2size 16 --Nf 5 --K 4 --folder /data/wanglei/hydrogen/walker-uniform-geminal/ --walkersize 256 --batchsiz 1024 --mc_proton_steps 100 --mc_electron_steps 400 --mc_proton_width 0.02 --mc_electron_width 0.04 --lr_proton 1.0 --lr_electron 0.05 --decay 0.01 --damping 0.001 --max_norm 0.001 --clip_factor 5.0 --acc_steps 1 --sr
```
