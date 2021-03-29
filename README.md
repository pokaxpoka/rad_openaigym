# Reinforcement Learning with Augmented Data (RAD): State augmentation on Mujoco Envs

Official codebase for [Reinforcement Learning with Augmented Data](https://mishalaskin.github.io/rad). This codebase was originally forked from [rlkit](https://github.com/vitchyr/rlkit). Official codebases for DM control and Procgen are available at [RAD: DM control](https://github.com/MishaLaskin/rad) and [RAD: Procgen](https://github.com/pokaxpoka/rad_procgen).

## BibTex

```
@article{laskin2020reinforcement,
  title={Reinforcement learning with augmented data},
  author={Laskin, Michael and Lee, Kimin and Stooke, Adam and Pinto, Lerrel and Abbeel, Pieter and Srinivas, Aravind},
  journal={arXiv preprint arXiv:2004.14990},
  year={2020}
}
```

## install

1. Install and use the included Ananconda environment
```
$ conda env create -f environment/linux-gpu-env.yml
$ source activate rlkit
```
You'll need to [get your own MuJoCo key](https://www.roboti.us/license.html) if you want to use MuJoCo.

2. Add this repo directory to your `PYTHONPATH` environment variable or simply
run:
```
pip install -e .
```

3. Install ["benchmarking MBRL"](https://arxiv.org/abs/1907.02057),
```
pip uninstall gym
pip install gym==0.9.4 mujoco-py==0.5.7 termcolor
cd mbbl_envs
pip install --user -e .
```

## Try it out

SAC
```
./scripts/run_sac.sh [env_name]
```

SAC + BN
```
./scripts/run_sac_bn.sh [env_name]
```

SAC + Gaussian
```
./scripts/run_sac_gaussian.sh [env_name] [prob] [std]
```

SAC + random amplitude scaling
```
./scripts/run_rand_ampl.sh [env_name] [single_flag: 0 or 1] [equal_flag: 0 or 1] [lower] [upper]
```