import pytest
import numpy as np

import numpy.testing as npt

from pytest import approx

from playroom_env.envs.realant import RealAntPlayroomEnv, RealAntEnv, RealAntNavigationEnv
from playroom_env.envs.realant_base import RealAntBasePlayroomEnv, RealAntBasePlayroomEnv2
from playroom_env.envs.playroom_env import InteroClass, ObjectClass

def variance_of_uniform(a, b):
    assert a < b
    return (b - a) ** 2 / 12.


def test_instance():
    env = RealAntPlayroomEnv()


def test_reset_env():
    env = RealAntPlayroomEnv()
    env.reset()


def test_reset_env_with_options():
    env = RealAntPlayroomEnv()
    env.reset(seed=0, options={})


def test_instance_not_ego_obs():
    env = RealAntPlayroomEnv(
        ego_obs=False,
        no_contact=False,
        sparse=False
    )
    env.reset()


def test_instance_no_contact():
    env = RealAntPlayroomEnv(
        ego_obs=True,
        no_contact=True,
        sparse=False
    )
    env.reset()


def test_reset_internal_state():
    env = RealAntPlayroomEnv(internal_reset="full")
    env.reset()
    env.internal_state = {
        InteroClass.ENERGY     : 1.0,
        InteroClass.TEMPERATURE: 1.0,
    }
    for key in InteroClass:
        assert env.internal_state[key] == approx(1.0)
    
    env.reset()
    initial_internal_state = {
        InteroClass.ENERGY     : env.full_battery,
        InteroClass.TEMPERATURE: 0.0,
    }
    
    for key in InteroClass:
        assert env.internal_state[key] == initial_internal_state[key]


def test_reset_if_resource_end():
    env = RealAntPlayroomEnv(internal_reset="full")
    env.default_metabolic_update = 0.1
    env.reset(seed=0)
    while True:
        ob, reward, terminated, truncated, info = env.step(0 * env.action_space.sample())
        if terminated:
            break
        else:
            intero = ob[-2:]
            assert intero[0] >= env.energy_lower_bound and intero[1] >= -1.0
    
    intero = ob[-2:]
    assert intero[0] < -env.energy_lower_bound or intero[1] < -1.0
    ob, _ = env.reset()
    intero = ob[-2:]
    assert intero[0] == approx(1.0) and intero[1] == approx(0)


def test_run_env():
    env = RealAntPlayroomEnv()
    env.reset()
    for i in range(10):
        env.step(env.action_space.sample())
    env.close()


def test_render_env():
    env = RealAntPlayroomEnv(show_sensor_range=True, n_bins=20, sensor_range=16.)
    for n in range(5):
        env.reset()
        for i in range(10):
            env.step(env.action_space.sample())
            env.render()
    env.close()


@pytest.mark.parametrize("setting,expected_mean, expected_var",
                         [
                             ("full", np.array([1.0, 38.0]), np.array([0.0, 0.0])),
                             ("random",
                              np.array([0.75, 38.]),
                              np.array([variance_of_uniform(0.5, 1.0),
                                        variance_of_uniform(38 - (42. - 38.) / 6.0, 38 + (42. - 38.) / 6.0)])),
                             ("error_case", None, None)
                         ])
def test_reset_internal(setting, expected_mean, expected_var):
    if setting != "error_case":
        env = RealAntPlayroomEnv(internal_reset=setting)
    else:
        with pytest.raises(ValueError):
            RealAntPlayroomEnv(internal_reset=setting)
        return
    
    obs_intero_list = []
    
    for i in range(2000):
        obs, _ = env.reset()
        
        obs_intero = obs[-2:]  # interoception
        obs_intero[1] = env.decode_temperature(obs_intero[1])
        obs_intero_list.append(obs_intero)
    
    obs_intero_mean = np.array(obs_intero_list).mean(axis=0)
    obs_intero_var = np.array(obs_intero_list).var(axis=0)
    
    # Test mean
    np.testing.assert_allclose(actual=obs_intero_mean,
                               desired=expected_mean,
                               atol=0.06)
    
    # Test var
    np.testing.assert_allclose(actual=obs_intero_var,
                               desired=expected_var,
                               atol=0.06)


@pytest.mark.parametrize("setting_e,setting_t,expected_mean, expected_var",
                         [
                             ([-1, 1], [34, 42], np.array([0.0, 0.0]), np.array([0.33, 0.33])),
                             ([-0.5, 0.5], [36, 40], np.array([0.0, 0.0]), np.array([1. / 12, 1. / 12])),
                             ([0, 1], [38, 42], np.array([0.5, 0.5]), np.array([1. / 12, 1. / 12])),
                         ])
def test_reset_internal_random_limit(setting_e, setting_t, expected_mean, expected_var):
    env = RealAntPlayroomEnv(internal_reset="random",
                             energy_random_range=setting_e,
                             temperature_random_range=setting_t)
    obs_intero_list = []
    
    for i in range(2000):
        obs, _ = env.reset()
        
        obs_intero = obs[-2:]  # interoception
        
        obs_intero_list.append(obs_intero)
    
    obs_intero_mean = np.array(obs_intero_list).mean(axis=0)
    obs_intero_var = np.array(obs_intero_list).var(axis=0)
    
    # Test mean
    np.testing.assert_allclose(actual=obs_intero_mean,
                               desired=expected_mean,
                               atol=0.03)
    
    # Test var
    np.testing.assert_allclose(actual=obs_intero_var,
                               desired=expected_var,
                               atol=0.02)


@pytest.mark.parametrize("reward_setting,expected,param",
                         [
                             ("homeostatic", -1.04, None),
                             ("homeostatic", -1.04, 0.5),  # Reward bias should be ignored
                             ("homeostatic_shaped", +1.04, None),
                             ("one", 0, None),
                             ("homeostatic_biased", 0.1 - 1.04, 0.1),
                             ("something_else", None, None),
                         ])
def test_reward_definition(reward_setting, expected, param):
    env = RealAntPlayroomEnv(reward_setting=reward_setting, coef_main_rew=1.0, reward_bias=param)
    
    action = np.array([0, -1, 0, 1, 0, 1, 0, -1], dtype=np.float32)
    
    env.prev_interoception = np.array([1, 1])
    env.internal_state = {
        InteroClass.ENERGY     : 0.8,
        InteroClass.TEMPERATURE: 0.0,
    }
    
    if reward_setting != "something_else":
        rew, info = env.get_reward(reward_setting, action, False)
        assert rew == approx(expected, abs=0.0001)
    else:
        with pytest.raises(ValueError):
            env.get_reward(reward_setting, action, False)
    
    if reward_setting == "one":
        env.internal_state = {
            InteroClass.ENERGY     : env.energy_lower_bound + 0.00001,
            InteroClass.TEMPERATURE: -0.999999,
        }
        _, reward, ternimated, truncated, _ = env.step(action)
        
        assert ternimated
        assert reward == approx(-1.0, abs=0.0001)


def test_object_num():
    env = RealAntPlayroomEnv(n_food=10, activity_range=10)
    env.reset()
    
    n_food = 0
    for obj in env.objects:
        if obj[3] is ObjectClass.FOOD:
            n_food += 1
    
    assert n_food == 10


def test_default_object_num():
    env = RealAntPlayroomEnv()
    env.reset()
    
    n_food = 0
    for obj in env.objects:
        if obj[3] is ObjectClass.FOOD:
            n_food += 1
    
    assert n_food == 1


def test_dt():
    env = RealAntPlayroomEnv()
    env.reset()
    
    assert env.dt == 0.01 * 5


def test_max_time_steps():
    env = RealAntPlayroomEnv()
    env._max_episode_steps = 10
    
    num_of_decisions = 0
    env.reset()
    
    while True:
        a = env.action_space.sample()
        num_of_decisions += 1
        
        _, _, ternimated, _, _ = env.step(a)
        
        if ternimated:
            break
    
    assert num_of_decisions == 10


def test_max_time_steps_init():
    env = RealAntPlayroomEnv(max_episode_steps=42)
    
    num_of_decisions = 0
    env.reset()
    
    while True:
        a = env.action_space.sample()
        num_of_decisions += 1
        
        _, _, terninated, _, _ = env.step(a)
        
        if terninated:
            break
    
    assert num_of_decisions == 42


def test_scale_temp():
    env = RealAntPlayroomEnv()
    
    assert env.scale_temperature(42) == approx(1.0)
    assert env.scale_temperature(34) == approx(-1.0)
    assert env.scale_temperature(38) == approx(0.0)


def test_decode_temp():
    env = RealAntPlayroomEnv()
    
    assert env.decode_temperature(1) == approx(42)
    assert env.decode_temperature(-1) == approx(34)
    assert env.decode_temperature(0.0) == approx(38)


@pytest.mark.parametrize("temp",
                         [
                             -0.3,
                             0.3,
                             -0.2,
                             0.2,
                             -0.1,
                             0.1,
                             -0.05,
                             0.05,
                             0,
                         ])
def test_render_temp_env(temp):
    env = RealAntPlayroomEnv(show_sensor_range=True, visualize_temp=True)
    env.reset()
    env.thermal_model.reset(temp_init=env.decode_temperature(temp))
    for i in range(10):
        env.step(env.action_space.sample())
        env.render()
    env.close()


def test_rgb():
    env = RealAntPlayroomEnv(vision=True, width=32, height=32)
    env.reset()
    
    im = None
    for i in range(10):
        env.step(0 * env.action_space.sample())
        im = env.get_image(mode='rgb_array')
    
    import matplotlib.pyplot as plt
    plt.clf()
    plt.imshow(im)
    plt.savefig("test_im.png")
    
    assert im.shape == (32, 32, 3)
    
    env.close()


def test_rgbd():
    env = RealAntPlayroomEnv(vision=True)
    env.reset()
    
    im = None
    for i in range(10):
        env.step(0. * env.action_space.sample())
        im = env.get_image(mode='rgbd_array')
    
    import matplotlib.pyplot as plt
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.imshow(im[:, :, :3] / 255.)
    plt.subplot(2, 2, 2)
    plt.hist(im[:, :, :3].flatten())
    plt.subplot(2, 2, 3)
    plt.imshow(im[:, :, 3], "gray")
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.hist(im[:, :, 3].flatten())
    plt.savefig("test_rgbd.png")
    
    assert im.shape == (32, 32, 4)
    assert im[:, :, :3].shape == (32, 32, 3)


def test_rgb_navigation():
    env = RealAntNavigationEnv(vision=True, width=32, height=32)
    env.reset()
    
    im = None
    for i in range(10):
        env.step(0 * env.action_space.sample())
        im = env.get_image(mode='rgb_array')
    
    import matplotlib.pyplot as plt
    plt.clf()
    plt.imshow(im)
    plt.savefig("test_nav_im.png")
    
    assert im.shape == (32, 32, 3)
    
    env.close()
    

def test_render_motion_line():
    env = RealAntPlayroomEnv(show_sensor_range=True, n_bins=20, sensor_range=16., show_move_line=True)
    env.reset()
    for i in range(10):
        env.step(0.1 * env.action_space.sample())
        env.render()
    env.close()


@pytest.mark.parametrize("reward_setting,expected,param",
                         [
                             ("homeostatic", -2, None),
                             ("homeostatic_shaped", +2, None),
                             ("one", 0, None),
                             ("homeostatic_biased", -1.5, 0.5),
                         ])
def test_modular_reward(reward_setting, expected, param):
    env = RealAntPlayroomEnv(reward_setting=reward_setting, coef_main_rew=1.0, reward_bias=param)
    
    action = env.action_space.sample() * 0
    
    env.prev_interoception = np.array([1, 1])
    env.internal_state = {
        InteroClass.ENERGY     : 0.0,
        InteroClass.TEMPERATURE: 0.0,
    }
    
    if reward_setting == "one":
        env.internal_state = {
            InteroClass.ENERGY     : env.energy_lower_bound + 0.00001,
            InteroClass.TEMPERATURE: -0.999999,
        }
        _, reward, ternimated, truncated, info = env.step(action)
        
        assert ternimated
        assert info["reward_module"] is None
    else:
        _, reward, ternimated, truncated, info = env.step(action)
        rm = info["reward_module"]
        assert rm.shape == (3,)


def test_intero_obs_position():
    env = RealAntPlayroomEnv(internal_reset="random")
    
    for _ in range(10):
        env.reset()
        
        obs, _, _, _, info = env.step(env.action_space.sample())
        
        assert np.all(obs[-2:] == info["interoception"])


def test_intero_dim():
    env = RealAntPlayroomEnv(internal_reset="random")
    assert env.dim_intero == 2


def test_update_n_food():
    env = RealAntPlayroomEnv(internal_reset="random", activity_range=10, n_food=6)
    
    env.reset()
    assert len(env.objects) == 6
    
    env.reset(n_food=2)
    assert len(env.objects) == 2
    
    env.reset(n_food=10)
    assert len(env.objects) == 10
    
    env.reset(n_food=0)
    assert len(env.objects) == 0


def test_max_episode_steps():
    env = RealAntPlayroomEnv()
    env.reset()
    
    assert env._max_episode_steps == 60_000


def test_set_max_episode_steps():
    env = RealAntPlayroomEnv(max_episode_steps=10)
    env.reset()
    
    assert env._max_episode_steps == 10


def test_info_at_reset():
    env = RealAntPlayroomEnv()
    obs, info = env.reset()
    
    assert "interoception" in info.keys()


@pytest.mark.parametrize("Env",
                         [
                             (RealAntPlayroomEnv),
                             (RealAntBasePlayroomEnv),
                             (RealAntBasePlayroomEnv2)
                         ])
def test_random_position_at_reset(Env):
    env = Env(random_position=True)
    
    env.reset()
    first_init_pos = env.wrapped_env.init_qpos.copy()
    
    for _ in range(10):
        env.step(env.action_space.sample())
        env.render()
    
    env.reset()
    second_init_pos = env.wrapped_env.init_qpos.copy()
    
    for _ in range(10):
        env.step(env.action_space.sample())
        env.render()
    
    assert 0.1 < ((first_init_pos - second_init_pos) ** 2).sum()
    env.close()


@pytest.mark.parametrize("Env",
                         [
                             (RealAntPlayroomEnv),
                         ])
def test_not_random_position_at_reset(Env):
    env = Env(random_position=False)
    
    env.reset()
    first_init_pos = env.wrapped_env.init_qpos.copy()
    
    env.reset()
    second_init_pos = env.wrapped_env.init_qpos.copy()
    
    assert 0.1 > ((first_init_pos - second_init_pos) ** 2).sum()
    env.close()


def test_obs_stack():
    env0 = RealAntEnv(obs_stack=1)
    env1 = RealAntEnv(obs_stack=3)

    obs0, _ = env0.reset()
    obs1, _ = env1.reset()
    
    assert obs0.shape[0] * 3 == obs1.shape[0]


def test_obs_delay():
    obs_delay = 3
    
    env = RealAntEnv(obs_delay=obs_delay, obs_stack=1)
    
    env.reset()
    
    for _ in range(obs_delay):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        
    npt.assert_allclose(env.past_measurements[-1], obs)


def test_obs_delay_stack():
    obs_delay = 3
    obs_stack = 4
    
    env = RealAntEnv(obs_delay=obs_delay, obs_stack=obs_stack)
    
    env.reset()
    
    for _ in range(obs_delay):
        obs, _, _, _, _ = env.step(env.action_space.sample())
    
    
    assert env.get_latest_measurement().size * obs_stack == obs.size
    
    npt.assert_allclose(
        np.concatenate(env.past_measurements[obs_delay:]),
        obs
    )
