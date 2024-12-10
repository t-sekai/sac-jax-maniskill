from collections import deque
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces.dict import Dict
from typing import Union
import torch
import numpy as np
from rfcl.logger import Logger
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils import common, gym_utils
from mani_skill.utils.visualization.misc import (
    images_to_video,
    tile_images,
)
from sapien.core import Pose
from mani_skill.utils.common import flatten_state_dict

class SparseRewardWrapper(gym.Wrapper):
    def step(self, action):
        o, _, terminated, truncated, info = self.env.step(action)
        return o, int(info["success"]), terminated, truncated, info


class ContinuousTaskWrapper(gym.Wrapper):
    """
    Makes a task continuous by disabling any early terminations, allowing episode to only end
    when truncated=True (timelimit reached)
    """

    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.success_once = False
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        terminated = False
        return observation, reward, terminated, truncated, info


class EpisodeStatsWrapper(gym.Wrapper):
    """
    Adds additional info. Anything that goes in the stats wrapper is logged to tensorboard/wandb under train_stats and test_stats
    """

    def reset(self, *, seed=None, options=None):
        self.eps_seed = seed
        obs, info = super().reset(seed=seed, options=options)
        self.eps_ret = 0
        self.eps_len = 0
        self.success_once = False
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.eps_ret += reward
        self.eps_len += 1
        info["eps_ret"] = self.eps_ret
        info["eps_len"] = self.eps_len
        info["seed"] = self.eps_seed
        self.success_once = self.success_once | info["success"]
        info["stats"] = dict(
            success_at_end=info["success"],
            success=self.success_once,
        )
        return observation, reward, terminated, truncated, info

class ClipActionWrapper(gym.ActionWrapper, gym.utils.RecordConstructorArgs): # Torch GPU variation, taken from the gymnasium clip action wrapper
    """Clip the continuous action within the valid :class:`Box` observation space bound.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ClipAction
        >>> env = gym.make("Hopper-v4")
        >>> env = ClipAction(env)
        >>> env.action_space
        Box(-1.0, 1.0, (3,), float32)
        >>> _ = env.reset(seed=42)
        >>> _ = env.step(np.array([5.0, -2.0, 0.0]))
        ... # Executes the action np.array([1.0, -1.0, 0]) in the base environment
    """
    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)
        self.action_space_low = torch.from_numpy(self.action_space.low).cuda()
        self.action_space_high = torch.from_numpy(self.action_space.high).cuda()

    def action(self, action):
        """Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """
        return torch.clip(action, self.action_space_low, self.action_space_high)
    
class RescaleActionWrapper(gym.ActionWrapper, gym.utils.RecordConstructorArgs): # Torch GPU variation, taken from the gymnasium rescale action wrapper
    """Affinely rescales the continuous action space of the environment to the range [min_action, max_action].

    The base environment :attr:`env` must have an action space of type :class:`spaces.Box`. If :attr:`min_action`
    or :attr:`max_action` are numpy arrays, the shape must match the shape of the environment's action space.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import RescaleAction
        >>> import numpy as np
        >>> env = gym.make("Hopper-v4")
        >>> _ = env.reset(seed=42)
        >>> obs, _, _, _, _ = env.step(np.array([1,1,1]))
        >>> _ = env.reset(seed=42)
        >>> min_action = -0.5
        >>> max_action = np.array([0.0, 0.5, 0.75])
        >>> wrapped_env = RescaleAction(env, min_action=min_action, max_action=max_action)
        >>> wrapped_env_obs, _, _, _, _ = wrapped_env.step(max_action)
        >>> np.alltrue(obs == wrapped_env_obs)
        True
    """

    def __init__(
        self,
        env: gym.Env,
        min_action: Union[float, int, torch.tensor],
        max_action: Union[float, int, torch.tensor],
    ):
        """Initializes the :class:`RescaleAction` wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            min_action (float, int or np.ndarray): The min values for each action. This may be a numpy array or a scalar.
            max_action (float, int or np.ndarray): The max values for each action. This may be a numpy array or a scalar.
        """
        assert isinstance(
            env.action_space, Box
        ), f"expected Box action space, got {type(env.action_space)}"
        assert np.less_equal(min_action, max_action).all(), (min_action, max_action)

        gym.utils.RecordConstructorArgs.__init__(
            self, min_action=min_action, max_action=max_action
        )
        gym.ActionWrapper.__init__(self, env)

        self.action_space_low = torch.from_numpy(self.action_space.low).cuda()
        self.action_space_high = torch.from_numpy(self.action_space.high).cuda()

        dtype_mapping = {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float64): torch.float64,
            np.dtype(np.float16): torch.float16,
            np.dtype(np.int32): torch.int32,
            np.dtype(np.int64): torch.int64,
            np.dtype(np.int16): torch.int16,
            np.dtype(np.int8): torch.int8,
            np.dtype(np.uint8): torch.uint8,
            np.dtype(np.bool_): torch.bool
        }

        self.min_action = (
            torch.zeros(env.action_space.shape, dtype=dtype_mapping[env.action_space.dtype]) + min_action
        ).cuda()
        self.max_action = (
            torch.zeros(env.action_space.shape, dtype=dtype_mapping[env.action_space.dtype]) + max_action
        ).cuda()

        self.action_space = Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.

        Args:
            action: The action to rescale

        Returns:
            The rescaled action
        """
        assert torch.all(torch.greater_equal(action, self.min_action)), (
            action,
            self.min_action,
        )
        assert torch.all(torch.less_equal(action, self.max_action)), (action, self.max_action)
        low = self.action_space_low
        high = self.action_space_high
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = torch.clip(action, low, high)
        return action
    
class RecordEpisodeWrapper(RecordEpisode):
    def __init__(
        self,
        env,
        output_dir,
        save_trajectory=True,
        trajectory_name=None,
        save_video=True,
        info_on_video=False,
        save_on_reset=True,
        save_video_trigger=None,
        max_steps_per_video=None,
        clean_on_close=True,
        record_reward=True,
        video_fps=30,
        source_type=None,
        source_desc=None,
        logger:Logger=None,
    ):
        super().__init__(env, output_dir, save_trajectory, trajectory_name, save_video, info_on_video, save_on_reset, save_video_trigger, max_steps_per_video, clean_on_close, record_reward, video_fps, source_type, source_desc)
        self.logger = logger
        self.untiled_render_images = [] # render_images but not tiled by num_envs. for organized wandb video

    def capture_image(self):
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) > 3:
            tiled_img = tile_images(img, nrows=self.video_nrows)
        else:
            tiled_img = img
            img = np.expand_dims(img, axis=0)
        self.untiled_render_images.append(img) # (num_envs, h, w, 3)
        return tiled_img # (h, w, 3)

    def flush_video(
        self,
        name=None,
        suffix="",
        verbose=False,
        ignore_empty_transition=True,
        save: bool = True,
    ):
        """
        Flush a video of the recorded episode(s) anb by default saves it to disk

        Arguments:
            name (str): name of the video file. If None, it will be named with the episode id.
            suffix (str): suffix to add to the video file name
            verbose (bool): whether to print out information about the flushed video
            ignore_empty_transition (bool): whether to ignore trajectories that did not have any actions
            save (bool): whether to save the video to disk
        """
        if len(self.render_images) == 0:
            return
        if ignore_empty_transition and len(self.render_images) == 1:
            return
        if save:
            self._video_id += 1
            if name is None:
                video_name = "{}".format(self._video_id)
                if suffix:
                    video_name += "_" + suffix
            else:
                video_name = name                
            if self.logger is not None:
                if self.logger.save_video_local:
                    images_to_video(
                        self.render_images,
                        str(self.output_dir),
                        video_name=video_name,
                        fps=self.video_fps,
                        verbose=verbose,
                    )
                untiled_render_images = np.array(self.untiled_render_images)
                untiled_render_images = untiled_render_images.transpose(1, 0, 2, 3, 4)
                self.logger.add_wandb_video(untiled_render_images)
        self._video_steps = 0
        self.render_images = []
        self.untiled_render_images = []

class PixelWrapper(gym.Wrapper):
    """
    Wrapper for pixel observations. Works with Maniskill vectorized environments
    """

    def __init__(self, env, num_frames=1):
        super().__init__(env)
        self._vis_shape = self.unwrapped.single_observation_space['sensor_data']['base_camera']['rgb'].shape # (h, w, c)
        self._qpos_shape = self.unwrapped.single_observation_space['agent']['qpos'].shape # (n)
        if self.unwrapped.spec.id == 'PickCube-v1':
            _goal_pos_shape = self.unwrapped.single_observation_space['extra']['goal_pos'].shape # (3,)
            self._qpos_shape = (self._qpos_shape[0] + _goal_pos_shape[0],)
        self.observation_space = Dict({
                'vis': Box(low=0, high=255, shape=(self._vis_shape[0], self._vis_shape[1], self._vis_shape[2] * num_frames), dtype=np.uint8),
                'qpos': Box(low=-np.inf, high=np.inf, shape=(self._qpos_shape[0] * num_frames,), dtype=np.float32)
            })
        self.single_observation_space = Dict({
                'vis': Box(low=0, high=255, shape=(self._vis_shape[0], self._vis_shape[1], self._vis_shape[2] * num_frames), dtype=np.uint8),
                'qpos': Box(low=-np.inf, high=np.inf, shape=(self._qpos_shape[0] * num_frames,), dtype=np.float32)
            })
        self._num_frames = num_frames
        self._rgb_stack = torch.zeros((self.unwrapped.num_envs, self._vis_shape[0], self._vis_shape[1], self._vis_shape[2], num_frames)).to(self.unwrapped.device)
        self._qpos_stack = torch.zeros((self.unwrapped.num_envs, self._qpos_shape[0], num_frames)).to(self.unwrapped.device)
        self._stack_idx = 0

    def _get_obs(self, obs):
        self._rgb_stack[..., self._stack_idx] = obs['sensor_data']['base_camera']['rgb']
        qpos = obs['agent']['qpos']
        if self.unwrapped.spec.id == 'PickCube-v1': # concat goal_pos to qpos
            qpos = torch.cat((qpos, obs['extra']['goal_pos']), dim=1)
        self._qpos_stack[..., self._stack_idx] = qpos
        self._stack_idx = (self._stack_idx + 1) % self._num_frames

        vis = torch.cat((self._rgb_stack[..., self._stack_idx:],self._rgb_stack[..., :self._stack_idx]), dim=-1)
        vis = vis.reshape((self.unwrapped.num_envs, self._vis_shape[0], self._vis_shape[1], -1))
        qpos = torch.cat((self._qpos_stack[..., self._stack_idx:],self._qpos_stack[..., :self._stack_idx]), dim=-1)
        qpos = qpos.reshape((self.unwrapped.num_envs, -1))

        return {'vis': vis, 'qpos' : qpos}

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        for _ in range(self._num_frames):
            obs_frames = self._get_obs(obs)
        return obs_frames, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._get_obs(obs), reward, terminated, truncated, info

class PointCloudWrapper(gym.Wrapper):
    """
    Wrapper for pointcloud observations. Works with Maniskill vectorized environments
    {'agent': {'qpos': torch.Size([3, 9]), 'qvel': torch.Size([3, 9])}, 
    'extra': {'goal_pos': torch.Size([3, 3]), 'is_grasped': torch.Size([3]), 'tcp_pose': torch.Size([3, 7])}, 
    'pointcloud': {'rgb': torch.Size([3, 4096, 3]), 'segmentation': torch.Size([3, 4096, 1]), 'xyzw': torch.Size([3, 4096, 4])}, 
    'sensor_data': {}, 'sensor_param': {'base_camera': {'cam2world_gl': torch.Size([3, 4, 4]), 'extrinsic_cv': torch.Size([3, 3, 4]), 'intrinsic_cv': torch.Size([3, 3, 3])}}}
    """

    def __init__(self, env, num_frames=1):
        super().__init__(env)
        num_envs, num_points, num_dims = self.unwrapped.observation_space['pointcloud']['xyzw'].shape
        num_dims += 3 # rgb
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_envs, num_points, num_dims), dtype=np.float32)
        self.single_observation_space = Box(low=-np.inf, high=np.inf, shape=(num_points, num_dims), dtype=np.float32)
        # self._num_frames = num_frames
        # self._pcd_stack = torch.zeros((self.unwrapped.num_envs, self._vis_shape[0], self._vis_shape[1], self._vis_shape[2], num_frames)).to(self.env.unwrapped.device)

    def reset(self, *, seed=None, options=None):
        # num_envs, num_points (16384?), 4
        obs, info = super().reset(seed=seed, options=options)
        return torch.cat([obs['pointcloud']['xyzw'], obs['pointcloud']['rgb']], dim=-1), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return torch.cat([obs['pointcloud']['xyzw'], obs['pointcloud']['rgb']], dim=-1), reward, terminated, truncated, info

class PointCloudWrapper2(gym.Wrapper):
    """
    Wrapper for pointcloud observations. Works with Maniskill vectorized environments
    {'agent': {'qpos': torch.Size([3, 9]), 'qvel': torch.Size([3, 9])}, 
    'extra': {'goal_pos': torch.Size([3, 3]), 'is_grasped': torch.Size([3]), 'tcp_pose': torch.Size([3, 7])}, 
    'pointcloud': {'rgb': torch.Size([3, 4096, 3]), 'segmentation': torch.Size([3, 4096, 1]), 'xyzw': torch.Size([3, 4096, 4])}, 
    'sensor_data': {}, 'sensor_param': {'base_camera': {'cam2world_gl': torch.Size([3, 4, 4]), 'extrinsic_cv': torch.Size([3, 3, 4]), 'intrinsic_cv': torch.Size([3, 3, 3])}}}
    """
    def __init__(self, env, obs_frame="tcp", n_points=2048):
        super().__init__(env)
        self.n_points = n_points
        self.observation_space = self.build_obs_space(env, self.n_points)
        self.single_observation_space = self.build_single_obs_space(env, self.n_points)
        self.obs_frame = obs_frame

    def observation(self, obs):
        obs = self.convert_obs(obs, self.obs_frame, self.n_points)
        return obs
    
    def reset(self, *, seed=None, options=None):
        # num_envs, num_points (16384?), 4
        obs, info = super().reset(seed=seed, options=options)
        return self.observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self.observation(obs), reward, terminated, truncated, info

    @staticmethod
    def build_obs_space(env, n_points=2048):
        obs_space = env.observation_space
        # Dict('agent': Dict('qpos': Box(-inf, inf, (2, 9), float32), 
        #     'qvel': Box(-inf, inf, (2, 9), float32)), 
        # 'extra': Dict('is_grasped': Box(False, True, (2,), bool), 
        #     'tcp_pose': Box(-inf, inf, (2, 7), float32), 
        #     'goal_pos': Box(-inf, inf, (2, 3), float32)), 
        # 'sensor_param': Dict('base_camera': Dict('extrinsic_cv': Box(-inf, inf, (2, 3, 4), float32), 'cam2world_gl': Box(-inf, inf, (2, 4, 4), float32), 
        # 'intrinsic_cv': Box(-inf, inf, (2, 3, 3), float32))), 
        # 'sensor_data': Dict(), 
        # 'pointcloud': Dict('xyzw': Box(-inf, inf, (2, 4096, 4), float32), 'rgb': Box(0, 255, (2, 4096, 3), uint8), 
        # 'segmentation': Box(-32768, 32767, (2, 4096, 1), int16)))
        state_dim = sum([v.shape[1] if len(v.shape) > 1 else 1 for _, v in obs_space['agent'].items()])
        state_dim += sum([v.shape[1] if len(v.shape) > 1 else 1 for _, v in obs_space['extra'].items()])

        return Dict({
            'state': Box(-float("inf"), float("inf"), shape=(env.unwrapped.num_envs, state_dim,), dtype=np.float32),
            'rgb': Box(-float("inf"), float("inf"), shape=(env.unwrapped.num_envs, n_points, 3), dtype=np.float32),
            'xyz': Box(-float("inf"), float("inf"), shape=(env.unwrapped.num_envs, n_points, 3), dtype=np.float32),
        })
    
    @staticmethod
    def build_single_obs_space(env, n_points=2048):
        obs_space = env.observation_space
        state_dim = sum([v.shape[1] if len(v.shape) > 1 else 1 for _, v in obs_space['agent'].items()])
        state_dim += sum([v.shape[1] if len(v.shape) > 1 else 1 for _, v in obs_space['extra'].items()])

        return Dict({
            'state': Box(-float("inf"), float("inf"), shape=(state_dim,), dtype=np.float32),
            'rgb': Box(-float("inf"), float("inf"), shape=(n_points, 3), dtype=np.float32),
            'xyz': Box(-float("inf"), float("inf"), shape=(n_points, 3), dtype=np.float32),
        })

    @classmethod
    def process_pcd_loop(cls, in_xyz, in_rgb, agent, extra, obs_frame, n_points):
        xyzs, rgbs = [], []
        for index, (xyz, rgb) in enumerate(zip(in_xyz, in_rgb)):
            if obs_frame == "base":
                # Use base pose transformation, Yuan: buggy since it's not 7 dims
                base_pose = agent["qpos"][index]
                p, q = base_pose[..., :3], base_pose[..., 3:]
                to_origin = Pose(p=p.cpu().numpy(), q=q.cpu().numpy()).inv()
            elif obs_frame == "tcp":
                # TCP stands for Tool Center Point, which is the position and orientation of the end-effector
                # Use tcp pose transformation
                tcp_pose = extra["tcp_pose"][index]
                p, q = tcp_pose[:3], tcp_pose[3:]
                to_origin = Pose(p=p.cpu().numpy(), q=q.cpu().numpy()).inv()

            to_origin_matrix = torch.from_numpy(to_origin.to_transformation_matrix()).to(xyz.device)
            rotation = to_origin_matrix[:3, :3]
            translation = to_origin_matrix[:3, -1:]

            mask = (xyz[:, 2] > 1e-4) & (xyz[:, -1] == 1)
            out_rgb = rgb[mask]
            out_xyz = xyz[mask][:, :3]

            # Sampling points
            index = np.arange(out_xyz.shape[0])
            if index.shape[0] > n_points:
                np.random.shuffle(index)
                index = index[:n_points]
            elif index.shape[0] < n_points:
                index = np.concatenate([index] * (n_points // index.shape[0]))
                index = np.concatenate([index, index[:n_points - index.shape[0]]])

            xyzs.append(out_xyz[index] @ rotation.T + translation.T)
            rgbs.append(out_rgb[index])

        out_xyz = torch.stack(xyzs, dim=0)
        out_rgb = torch.stack(rgbs, dim=0)

        return {"xyz": out_xyz, "rgb": out_rgb}

    @classmethod
    def convert_obs(cls, obs, obs_frame='tcp', n_points=2048):
        in_xyz, in_rgb = obs["pointcloud"]['xyzw'], obs["pointcloud"]['rgb']
        agent, extra = obs["agent"], obs["extra"]
        new_pcd_dict = cls.process_pcd_loop(in_xyz, in_rgb, agent, extra, obs_frame, n_points)

        states = [flatten_state_dict(obs["agent"], use_torch=True)]
        if len(obs["extra"]) > 0:
            states.append(flatten_state_dict(obs["extra"], use_torch=True))
        new_pcd_dict['state'] = torch.cat(states, -1)

        return new_pcd_dict
