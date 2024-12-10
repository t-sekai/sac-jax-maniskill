"""
Code to run Reverse Forward Curriculum Learning.
Configs can be a bit complicated, we recommend directly looking at configs/ms2/base_sac_ms2_sample_efficient.yml for what options are available.
Alternatively, go to the file defining each of the nested configurations and see the comments.
"""
import copy
import os
import os.path as osp
import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import jax
import numpy as np
import optax
from omegaconf import OmegaConf
import time
import multiprocessing

from rfcl.agents.sac import SAC, ActorCritic, SACConfig
from rfcl.agents.sac.networks import DiagGaussianActor, VisualEncoder
from rfcl.data.dataset import ReplayDataset
from rfcl.envs.make_env import EnvConfig, make_env_from_cfg
from rfcl.logger import LoggerConfig, Logger
from rfcl.models import NetworkConfig, build_network_from_cfg
from rfcl.utils.parse import parse_cfg
from rfcl.utils.spaces import get_action_dim


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@dataclass
class TrainConfig:
    steps: int
    actor_ve_lr: float
    critic_ve_lr: float
    actor_lr: float
    critic_lr: float
    dataset_path: str
    shuffle_demos: bool
    num_demos: int

    data_action_scale: Optional[float]

@dataclass
class SACNetworkConfig:
    rgb_encoder: NetworkConfig
    pointcloud_encoder: NetworkConfig
    actor: NetworkConfig
    critic: NetworkConfig


@dataclass
class SACExperiment:
    seed: int
    sac: SACConfig
    env: EnvConfig
    eval_env: EnvConfig
    train: TrainConfig
    network: SACNetworkConfig
    logger: Optional[LoggerConfig]
    verbose: int
    algo: str = "sac"
    save_eval_video: bool = True  # whether to save eval videos
    demo_seed: int = None  # fix a seed to fix which demonstrations are sampled from a dataset


from dacite import from_dict


def main(cfg: SACExperiment):
    np.random.seed(cfg.seed)

    ### Setup the experiment parameters ###

    # Setup training and evaluation environment configs
    env_cfg = cfg.env
    if "env_kwargs" not in env_cfg:
        env_cfg["env_kwargs"] = dict()
    cfg.eval_env = {**env_cfg, **cfg.eval_env}
    cfg = from_dict(data_class=SACExperiment, data=OmegaConf.to_container(cfg))
    env_cfg = cfg.env
    eval_env_cfg = cfg.eval_env

    # change exp name if it exists
    orig_exp_name = cfg.logger.exp_name
    exp_path = osp.join(cfg.logger.workspace, orig_exp_name)
    if osp.exists(exp_path):
        i = 1
        prev_exp_path = exp_path
        while osp.exists(exp_path):
            prev_exp_path = exp_path
            cfg.logger.exp_name = f"{orig_exp_name}_{i}"
            exp_path = osp.join(cfg.logger.workspace, cfg.logger.exp_name)
            i += 1
        warnings.warn(f"{prev_exp_path} already exists. Changing exp_name to {cfg.logger.exp_name}")
    video_path = osp.join(cfg.logger.workspace, cfg.logger.exp_name, "videos")

    cfg.sac.num_envs = cfg.env.num_envs
    cfg.sac.num_eval_envs = cfg.eval_env.num_envs

    ### Set up Logger ###
    cfg.logger.cfg = asdict(cfg)
    logger_cfg = cfg.logger

    if logger_cfg is not None:
        if logger_cfg.exp_name is None:
            exp_name = f"{round(time.time_ns() / 1000)}"
            if hasattr(env, "name"):
                exp_name = f"{env.name}/{exp_name}"
            logger_cfg.exp_name = exp_name
        if not logger_cfg.best_stats_cfg:
            logger_cfg.best_stats_cfg = {"test/ep_ret_avg": 1, "train/ep_ret_avg": 1}
        manager = multiprocessing.Manager()
        logger = Logger.create_from_cfg(logger_cfg, manager)
    else:
        logger = None

    ### Create Environments ###
    
    np.random.seed(cfg.seed)

    env, env_meta = make_env_from_cfg(env_cfg, seed=cfg.seed)
    eval_env = None
    if cfg.sac.num_eval_envs > 0:
        eval_env, _ = make_env_from_cfg(
            eval_env_cfg,
            seed=cfg.seed + 1_000_000,
            video_path=video_path if cfg.save_eval_video else None,
            logger=logger,
        )

    sample_obs, sample_acts = env_meta.sample_obs, env_meta.sample_acts
    # create actor and critics models
    act_dims = get_action_dim(env_meta.act_space)

    visual_encoder, encoder_is_pcd = None, False
    if cfg.env.env_kwargs['obs_mode'] == 'rgb':
        visual_encoder = VisualEncoder(
            visual_encoder=build_network_from_cfg(cfg.network.rgb_encoder),
            obs_mode=cfg.env.env_kwargs['obs_mode']
        )
        #sample_obs, ve = ve.init_with_output(jax.random.key(0), sample_obs)
    elif cfg.env.env_kwargs['obs_mode'] == 'pointcloud':
        visual_encoder = VisualEncoder(
            visual_encoder=build_network_from_cfg(cfg.network.pointcloud_encoder), 
            obs_mode=cfg.env.env_kwargs['obs_mode']
        )
    def create_ac_model():
        actor = DiagGaussianActor(
            feature_extractor=build_network_from_cfg(cfg.network.actor),
            act_dims=act_dims,
            state_dependent_std=True,
        )
        ac = ActorCritic.create(
            cfg.env.env_kwargs['obs_mode'],
            jax.random.PRNGKey(cfg.seed),
            visual_encoder=visual_encoder,
            actor=actor,
            critic_feature_extractor=build_network_from_cfg(cfg.network.critic),
            sample_obs=sample_obs,
            sample_acts=sample_acts,
            initial_temperature=cfg.sac.initial_temperature,
            actor_ve_optim=optax.adam(learning_rate=cfg.train.actor_ve_lr),
            critic_ve_optim=optax.adam(learning_rate=cfg.train.critic_ve_lr),
            actor_optim=optax.adam(learning_rate=cfg.train.actor_lr),
            critic_optim=optax.adam(learning_rate=cfg.train.critic_lr),
            num_qs=cfg.sac.num_qs,
            num_min_qs=cfg.sac.num_min_qs,
            share_ve=cfg.sac.share_ve,
        )
        return ac

    # create our algorithm
    if cfg.sac.eval_steps == None:
        cfg.sac.eval_steps = cfg.eval_env.max_episode_steps * cfg.sac.eval_episodes
    else:
        assert cfg.sac.eval_steps % cfg.eval_env.max_episode_steps == 0, "sac.eval_stpes should be a multiple of eval_env.max_episode_steps!"
    ac = create_ac_model()
    def count_params(params):
        import jax.numpy as jnp
        return sum(jnp.size(p) for p in jax.tree_util.tree_leaves(params))
    if ac.actor_ve is not None:
        print(f'Visual encoder size: {count_params(ac.actor_ve.params)}')
    print(f'Actor size: {count_params(ac.actor.params)}')
    print(f'Critic size: {count_params(ac.critic.params)}')
    algo = SAC(
        env=env,
        eval_env=eval_env,
        env_type=cfg.env.env_type,
        obs_mode=cfg.env.env_kwargs["obs_mode"],
        ac=ac,
        logger=logger,
        cfg=cfg.sac,
    )
    if cfg.sac.load_ckpt:
        # Load checkpoint from path
        assert cfg.sac.ckpt_path is not None, "Please specify train.ckpt_path when loading checkpoint!"
        algo.load_from_path(cfg.sac.ckpt_path)
    #algo.offline_buffer = demo_replay_dataset  # create offline buffer to oversample from
    rng_key, train_rng_key = jax.random.split(jax.random.PRNGKey(cfg.seed), 2)
    algo.train(
        rng_key=train_rng_key,
        steps=cfg.train.steps,
        verbose=cfg.verbose,
    )
    algo.save(osp.join(algo.logger.model_path, "latest.jx"), with_buffer=False)
    # with_buffer=True means you can use the checkpoint to easily resume training with the same replay buffer data
    # algo.save(osp.join(algo.logger.model_path, "latest.jx"), with_buffer=True)
    env.close(), eval_env.close()

if __name__ == "__main__":
    cfg = parse_cfg(default_cfg_path=sys.argv[1])
    main(cfg)
