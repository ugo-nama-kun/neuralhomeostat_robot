import math
import os
import tempfile
import xml.etree.ElementTree as ET
import inspect
from collections import deque
from enum import Enum, auto
from copy import copy
from logging import warning

import glfw
import numpy as np

import mujoco


from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE, MujocoEnv
from gymnasium import utils
from scipy.spatial.transform import Rotation

BIG = 1e6
DEFAULT_CAMERA_CONFIG = {}


class OnTheBoardEnv(MujocoEnv, utils.EzPickle):
    MODEL_CLASS = None
    ORI_IND = None
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps"  : 20,
    }
    
    def __init__(self,
                 vision=False,
                 width=64,
                 height=64,
                 *args, **kwargs):
        
        utils.EzPickle.__init__(**locals())
        
        # for openai baseline
        self.reward_range = (-float('inf'), float('inf'))
        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise Exception("MODEL_CLASS unspecified!")
        import pathlib
        p = pathlib.Path(inspect.getfile(self.__class__))
        MODEL_DIR = os.path.join(p.parent, "models", model_cls.FILE)
        
        tree = ET.parse(MODEL_DIR)
        worldbody = tree.find(".//worldbody")

        asset = tree.find(".//asset")
        # ET.SubElement(
        #     asset, "texture", dict(
        #         name="floor_texture",
        #         type="2d",
        #         file=os.path.dirname(__file__) + "/models/texture/floor.png",
        #         width="100",
        #         height="100",
        #     )
        # )
        # ET.SubElement(
        #     asset, "material", dict(
        #         name="grass",
        #         texture="floor_texture",
        #         texrepeat="20 20"
        #     )
        # )

        attrs = dict(
            type="box", conaffinity="1", condim="3",
        )
        ET.SubElement(
            worldbody, "geom", dict(
                attrs,
                name="box",
                pos="0 0 0.1",
                size="0.06 0.06 0.06"))
        
        with tempfile.NamedTemporaryFile(mode='wt', suffix=".xml") as tmpfile:
            file_path = tmpfile.name
            tree.write(file_path)
            
            # build mujoco
            self.wrapped_env = model_cls(
                file_path,
                vision=vision,
                width=width,
                height=height,
                **kwargs
            )
        
        # optimization, caching obs spaces
        ub = BIG * np.ones(self.get_current_obs().shape, dtype=np.float32)
        self.obs_space = spaces.Box(ub * -1, ub)
        ub = BIG * np.ones(self.get_current_obs().shape, dtype=np.float32)
        self.robot_obs_space = spaces.Box(ub * -1, ub)

        # Augment the action space
        ub = np.ones(len(self.wrapped_env.action_space.high), dtype=np.float32)
        self.act_space = spaces.Box(ub * -1, ub)

        self.max_episode_length = np.inf
        
    def reset(self, seed=None, return_info=True, options=None):
        self.wrapped_env.reset(seed=seed)
        
        info = {}
        
        return (self.get_current_obs(), info) if return_info else self.get_current_obs()
    
    def step(self, action: np.ndarray):
        
        _, inner_rew, terminated, truncated, info = self.wrapped_env.step(action)
        
        return self.get_current_obs(), 0.0, False, False, {}
        
    def get_current_obs(self):
        return self.wrapped_env.get_current_obs()
        
    @property
    def observation_space(self):
        return self.obs_space
        
    @property
    def action_space(self):
        return self.act_space
    
    @property
    def dt(self):
        return self.wrapped_env.dt
    
    def close(self):
        if self.wrapped_env.mujoco_renderer is not None:
            self.wrapped_env.mujoco_renderer.close()
    
    def get_ori(self):
        """
        First it tries to use a get_ori from the wrapped env. If not successfull, falls
        back to the default based on the ORI_IND specified in Maze (not accurate for quaternions)
        """
        obj = self.wrapped_env
        while not hasattr(obj, 'get_ori') and hasattr(obj, 'wrapped_env'):
            obj = obj.wrapped_env
        try:
            return obj.get_ori()
        except (NotImplementedError, AttributeError) as e:
            pass
        return self.wrapped_env.data.qpos[self.__class__.ORI_IND]
    
    def render(
            self,
            mode='human',
            camera_id=None,
            camera_name=None
    ):
        return self.get_image(mode=mode, camera_id=camera_id, camera_name=camera_name)
    
    def get_image(
            self,
            mode='human',
            camera_id=None,
            camera_name=None
    ):
        
        if mode == "rgbd_array":
            viewers = [self.wrapped_env.mujoco_renderer._get_viewer(render_mode="rgb_array")]
            viewers.append(self.wrapped_env.mujoco_renderer._get_viewer(render_mode="depth_array"))
        else:
            viewers = [self.wrapped_env.mujoco_renderer._get_viewer(render_mode=mode)]
            
        if mode == "rgbd_array":
            im = self.wrapped_env.mujoco_renderer.render(
                "rgb_array",
                camera_id,
                camera_name, )
            im_d = self.wrapped_env.mujoco_renderer.render(
                "depth_array",
                camera_id,
                camera_name, )[:, :, np.newaxis]
            
            im = np.dstack((im, im_d))
        else:
            im = self.wrapped_env.mujoco_renderer.render(
                mode,
                camera_id,
                camera_name,
            )
        
        # delete unnecessary markers: https://github.com/openai/mujoco-py/issues/423#issuecomment-513846867
        for v in viewers:
            del v._markers[:]
        
        return im
