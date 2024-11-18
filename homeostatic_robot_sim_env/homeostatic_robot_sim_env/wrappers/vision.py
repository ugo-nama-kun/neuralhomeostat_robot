from collections import deque
from typing import Tuple

import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Box

from playroom_env.envs.playroom_env import BIG
from playroom_env.envs.realant import RealAntPlayroomEnv, RealAntNavigationEnv


class VisionEnvWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env: gymnasium.Env, n_frame: int, mode: str, noise=0.0):
        """

        :param env: environment to learn
        :param n_frame:  number of frames to stack
        :param mode: vision input mode. rgb_array or rgbd_array.
        :param noise: gaussian noise scale to raw image.
        """
        super().__init__(env)
        
        assert env.vision is True

        self.n_frame_stack = n_frame
        self.noise = noise

        assert mode in {"rgb_array", "rgbd_array"}
        self.mode = mode
        
        base = env
        while True:
            try:
                base = base.env
            except:
                break
        
        if isinstance(base, RealAntNavigationEnv):
            self.no_interoception = True
        elif isinstance(base, RealAntPlayroomEnv):
            self.no_interoception = False
        else:
            raise ValueError("env must be RealAntPlayroomEnv or RealAntNavigationEnv.")

        self.frame_stack = deque(maxlen=n_frame)

        self.obs_dims = env.multi_modal_dims

        dummy_obs, _ = self.reset()
        ub = BIG * np.ones(dummy_obs.shape, dtype=np.float32)
        self.obs_space = Box(ub * -1, ub)

    def reset(self, **kwargs):
        self.frame_stack.clear()
        observation, info = self.env.reset(**kwargs)
        return self.observation(observation), info

    @property
    def observation_space(self):
        return self.obs_space

    def observation(self, observation):
        # scale into (-1, 1)
        vision = 2. * (self.env.get_image(mode=self.mode).astype(np.float32) / 255. - 0.5)
        
        if 0.0 < self.noise:
            eps = np.random.randn(vision.shape[0], vision.shape[1], vision.shape[2])
            vision = np.clip(vision + self.noise * eps, a_min=-1, a_max=1)

        if len(self.frame_stack) < self.n_frame_stack:
            for i in range(self.n_frame_stack):
                self.frame_stack.append(vision.flatten())
        else:
            self.frame_stack.append(vision.flatten())

        # observation of low-dim two-resource is [proprioception(27 dim), exteroception (40 dim, default), interoception (2dim)]
        proprioception = observation[:self.obs_dims[0]]
        
        if self.no_interoception:
            fullvec = np.concatenate([proprioception, np.concatenate(self.frame_stack)])
        else:
            interoception = observation[-self.obs_dims[2]:]
            fullvec = np.concatenate([proprioception, interoception, np.concatenate(self.frame_stack)])
            
        return fullvec

    def decode_vision(self, im):
        return 0.5 * im + 0.5


def to_multi_modal(obs_tensor,
                   im_size: Tuple[int, int],
                   n_frame: int,
                   n_channel: int,
                   prop_size: int,
                   n_stack_motor=1,
                   action_as_obs=False,
                   no_interoception=False,
                   ):
    """
    function to convert the flatten multimodal observation to invididual modality
    :param obs_tensor: observation obtained from data: Size = [n_batch, channel, height, width]
    :param im_size: size of the image (height, width)
    :param n_frame: number of stacks of observation
    :param n_channel: number of channel of vision (rgb:3, rgbd:4)
    :param no_interoception: enable if interoception is omitted.
    :return:
    # TODO: remove torch dependency
    """

    dim_prop = prop_size * n_stack_motor
    if action_as_obs:
        dim_prop += 8

    outputs = []
    if no_interoception:
        sizes = (dim_prop, np.prod(im_size) * n_channel * n_frame)
        input_prop, input_vision = torch.split(obs_tensor, split_size_or_sections=sizes, dim=1)
        outputs.append(input_prop)
    else:
        sizes = (dim_prop, 2, np.prod(im_size) * n_channel * n_frame)
        input_prop, input_intero, input_vision = torch.split(obs_tensor, split_size_or_sections=sizes, dim=1)
        outputs.extend([input_prop, input_intero])

    input_vision = torch.split(input_vision, split_size_or_sections=[np.prod(im_size) * n_channel] * n_frame, dim=1)

    input_vision = [im.reshape(-1, im_size[0], im_size[1], n_channel) for im in input_vision]

    input_vision = torch.cat(input_vision, dim=3)

    input_vision = input_vision.view(-1, im_size[0], im_size[1], n_channel * n_frame).permute(0, 3, 1, 2)
    
    outputs.append(input_vision)

    return tuple(outputs)
