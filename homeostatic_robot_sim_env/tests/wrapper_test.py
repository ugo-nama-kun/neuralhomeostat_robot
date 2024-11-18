import pytest
import torch

from playroom_env.wrappers.vision import VisionEnvWrapper, to_multi_modal

from playroom_env.envs.realant import RealAntPlayroomEnv, RealAntNavigationEnv


@pytest.mark.parametrize("expected, env_class, im_setting, n_frame",
                         [
                             (8 * 3 + 32 * 32 * 3 * 2 + 2, RealAntPlayroomEnv, "rgb_array", 2),
                             (8 * 3 + 32 * 32 * 4 * 4 + 2, RealAntPlayroomEnv, "rgbd_array", 4),
                             (8 * 3 + 32 * 32 * 3 * 2, RealAntNavigationEnv, "rgb_array", 2),
                             (8 * 3 + 32 * 32 * 4 * 4, RealAntNavigationEnv, "rgbd_array", 4),
                         ])
def test_vision_wrapper(expected, env_class, im_setting, n_frame):
    env = env_class(vision=True, width=32, height=32, obs_stack=3)
    env = VisionEnvWrapper(env,
                           n_frame=n_frame,
                           mode=im_setting)

    obs, _ = env.reset()

    assert obs.shape[0] == expected
    assert env.observation_space.shape == (expected,)


@pytest.mark.parametrize("shape_prop, shape_vision, shape_intero, n_channel, env_class, im_setting, n_frame, action_as_obs",
                         [
                             ((1, 8 * 3), (1, 3 * 2, 32, 32), (1, 2), 3, RealAntPlayroomEnv, "rgb_array", 2, False),
                             ((1, 8 * 3), (1, 4 * 4, 32, 32), (1, 2), 4, RealAntPlayroomEnv, "rgbd_array", 4, False),
                             ((1, 8 * 3 + 8), (1, 3 * 4, 32, 32), (1, 2), 3, RealAntPlayroomEnv, "rgb_array", 4, True),
                         ])
def test_to_multi_model(shape_prop, shape_vision, shape_intero, n_channel, env_class, im_setting, n_frame, action_as_obs):
    env = env_class(vision=True, width=32, height=32, action_as_obs=action_as_obs, obs_stack=3)
    env = VisionEnvWrapper(env,
                           n_frame=n_frame,
                           mode=im_setting)

    obs, _ = env.reset()

    outputs = to_multi_modal(torch.tensor([obs]),
                                          im_size=(32, 32),
                                          n_frame=n_frame,
                                          n_channel=n_channel,
                                          action_as_obs=action_as_obs,
                                          prop_size=8*3)

    assert len(outputs) == 3
    
    prop, intero, vision = outputs
    assert prop.shape == shape_prop
    assert intero.shape == shape_intero
    assert vision.shape == shape_vision


@pytest.mark.parametrize("shape_prop, shape_vision, n_channel, env_class, im_setting, n_frame, action_as_obs",
                         [
                             ((1, 22), (1, 1 * 3, 32, 32), 3, RealAntNavigationEnv, "rgb_array", 1, False),
                             ((1, 22 + 8), (1, 2 * 4, 32, 32), 4, RealAntNavigationEnv, "rgbd_array", 2, True),
                             ((1, 22), (1, 3 * 3, 32, 32), 3, RealAntNavigationEnv, "rgb_array", 3, False),
                             ((1, 22 + 8), (1, 4 * 4, 32, 32), 4, RealAntNavigationEnv, "rgbd_array", 4, True),
                         ])
def test_to_multi_model_navigation(shape_prop, shape_vision, n_channel, env_class, im_setting, n_frame, action_as_obs):
    env = env_class(vision=True, width=32, height=32, action_as_obs=action_as_obs)
    env = VisionEnvWrapper(env,
                           n_frame=n_frame,
                           mode=im_setting)
    
    obs, _ = env.reset()
    
    outputs = to_multi_modal(torch.tensor([obs]),
                             im_size=(32, 32),
                             n_frame=n_frame,
                             n_channel=n_channel,
                             action_as_obs=action_as_obs,
                             no_interoception=True)
    
    assert len(outputs) == 2
    
    prop, vision = outputs
    assert prop.shape == shape_prop
    assert vision.shape == shape_vision


@pytest.mark.parametrize("expected, env_class, im_setting, n_frame",
                         [
                             (8 * 3 + 32 * 32 * 3 * 2 + 2, RealAntPlayroomEnv, "rgb_array", 2),
                             (8 * 3 + 32 * 32 * 4 * 4 + 2, RealAntPlayroomEnv, "rgbd_array", 4),
                         ])
def test_vision_wrapper_obs(expected, env_class, im_setting, n_frame):
    env = env_class(vision=True, width=32, height=32, obs_stack=3)

    env = VisionEnvWrapper(env,
                           n_frame=n_frame,
                           mode=im_setting)
    
    from gymnasium.wrappers import NormalizeReward

    env = NormalizeReward(env)

    obs, _ = env.reset()
    
    for _ in range(10):
        obs, rew, done, truncated, info = env.step(env.action_space.sample())

        assert obs.shape[0] == expected
        assert env.observation_space.shape == (expected,)


@pytest.mark.parametrize("n_channel, env_class, im_setting, n_frame, action_as_obs",
                         [
                             (3, RealAntNavigationEnv, "rgb_array", 1, False),
                         ])
def test_to_multi_model_navigation(n_channel, env_class, im_setting, n_frame, action_as_obs):
    env = env_class(vision=True, width=32, height=32, action_as_obs=action_as_obs)
    env = VisionEnvWrapper(env,
                           n_frame=n_frame,
                           mode=im_setting)

    obs, _ = env.reset()

    outputs = to_multi_modal(torch.tensor([obs]),
                             im_size=(32, 32),
                             n_frame=n_frame,
                             n_channel=n_channel,
                             action_as_obs=action_as_obs,
                             no_interoception=True,
                             n_stack_motor=3,
                             prop_size=8)

    _, vision = outputs

    import einops
    import matplotlib.pyplot as plt

    im = einops.rearrange(vision[0, :3], "c x y -> x y c").numpy()
    plt.clf()
    plt.imshow(im)
    plt.savefig(f"test_wrapper_im_{im_setting}.png")
