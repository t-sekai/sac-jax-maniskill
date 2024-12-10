# SAC-JAX

Soft Actor-Critic implemented in JAX, adapted from [RFCL: Reverse Forward Curriculum Learning (ICLR 2024)](https://github.com/StoneT2000/rfcl.git)'s SAC implementation (a partial fork of the original RLPD and JaxRL repos that has been optimized to run faster and support vectorized environments). 

The code currently only supports Maniskill tasks. It has been extended to run on cpu/gpu vectorized environment and supports state/rgb observation. SAC-JAX can solve PushCube-v1 in 2 minutes, which is lightning fast for SAC. 

## Installation

We recommend using conda/mamba and you can install the dependencies as so:

```bash
conda create -n "sac-jax" "python==3.9"
conda activate sac-jax
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```

Then you can install ManiSkill and its dependencies

```bash
pip install mani_skill torch
```

## Train

To train with environment vectorization run

```bash
env_id=PushCube-v1
seed=42
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_sac.py configs/base_sac_ms3.yml \
  seed=${seed} env.env_id=${env_id} \
  logger.exp_name="sac-${env_id}-state-${seed}-walltime_efficient" \
  logger.wandb=True logger.project_name="sac_jax" logger.wandb_cfg.group=${env_id}
```

**Some useful hyperparameters to tune:** 

- train.steps=1_000_000
- sac.discount : 0.95
- sac.tau : 0.005
- sac.replay_buffer_capacity : 1_000_000
- env.num_envs : 8
- env.env_type : "gym:cpu" ("gym:gpu" for gpu vectorized env)
- env.env_kwargs.obs_mode : "state" ("rgb" for pixel observation)
- env.render_size : 64 (rgb obs render size)
- env.num_frames : 3 (frame stacking for rgb obs)
- env.max_episode_steps : 50

See configs/base_sac_ms3.yml for all hyperparameters.

**evaluation videos are saved to `exps/<exp_name>/videos` and wandb.**

## Citation

If you find this work useful, consider citing:
```
@article{tao2024rfcl,
  title={Reverse Forward Curriculum Learning for Extreme Sample and Demonstration Efficiency in RL},
  author={Tao, Stone and Shukla, Arth and Chan, Tse-kai and Su, Hao},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year={2024}
}
```