import pytest
import numpy as np

from pytest import approx

from playroom_env.envs.command_env import Commands
from playroom_env.envs.realant_base import RealAntBasePlayroomEnv, RealAntBaseEnv, RealAntBaseCommandEnv, RealAntBaseNavigationEnv, RealAntBasePlayroomEnv2


def test_instance():
    env = RealAntBasePlayroomEnv()


def test_reset_env():
    env = RealAntBasePlayroomEnv()
    env.reset()


def test_run_env():
    env = RealAntBasePlayroomEnv()
    for _ in range(3):
        env.reset()
        for i in range(10):
            env.render()
            env.step(env.action_space.sample())
    env.close()


def test_run_env_no_wall():
    env = RealAntBasePlayroomEnv(no_wall=True)
    for _ in range(3):
        env.reset()
        for i in range(10):
            env.render()
            env.step(env.action_space.sample())
    env.close()
    
def test_run_navigation_env():
    env = RealAntBaseNavigationEnv()
    for _ in range(3):
        env.reset()
        for i in range(10):
            env.render()
            env.step(env.action_space.sample())
    env.close()

def test_dim():
    env = RealAntBasePlayroomEnv()
    obs, _ = env.reset()
    
    robo_obs = 3 + 3 + 8 + 8
    assert len(env.observation_space.high) == 2 + robo_obs + 20 + 2
    assert len(env.action_space.high) == 8
    assert len(obs) == 2 + robo_obs + 20 + 2
    assert len(env.action_space.sample()) == 8

def test_run_length_env():
    env = RealAntBaseEnv(max_episode_steps=50)
    env.reset()
    steps = 0
    done = False
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        steps += 1
        
    assert steps == 50


def test_run_length_playroom_env():
    env = RealAntBasePlayroomEnv(max_episode_steps=50)
    env.reset()
    steps = 0
    done = False
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        steps += 1
    
    assert steps == 50


def test_dim_action_not_included():
    env = RealAntBaseEnv(action_as_obs=False)
    obs, _ = env.reset()
    
    robo_obs = 3 + 3 + 8 + 8
    
    assert len(env.observation_space.high) == robo_obs
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs
    assert len(env.action_space.sample()) == 8
    
def test_dim_action_included():
    env = RealAntBaseEnv(action_as_obs=True)
    obs, _ = env.reset()
    
    robo_obs =  3 + 3 + 8 + 8

    assert len(env.observation_space.high) == robo_obs + 8
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs + 8
    assert len(env.action_space.sample()) == 8


def test_dim_action_included_playroom():
    env = RealAntBasePlayroomEnv(action_as_obs=True)
    obs, _ = env.reset()
    
    robo_obs = 3 + 3 + 8 + 8

    # action as observation is only about the agent's motor action
    assert len(env.observation_space.high) == 2 + robo_obs + 20 + 2 + 8
    assert len(env.action_space.high) == 8
    assert len(obs) == 2 + robo_obs + 20 + 2 + 8
    assert len(env.action_space.sample()) == 8


def test_dim_command_env():
    env = RealAntBaseCommandEnv()
    obs, _ = env.reset()
    
    robo_obs = 3 + 3 + 8 + 8
    
    # action as observation is only about the agent's motor action
    assert len(env.observation_space.high) == robo_obs + len(Commands)
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs + len(Commands)
    assert len(env.action_space.sample()) == 8


def test_command_env_fixed_command():
    env = RealAntBaseCommandEnv(fixed_command=1)
    
    assert env.is_command_fixed is True

    for _ in range(20):
        obs, _ = env.reset()
        assert env.command is Commands.FORWARD

def test_command_env_fix_command():
    env = RealAntBaseCommandEnv(fixed_command=None)

    assert env.is_command_fixed is False

    # Check if command is fixed
    initial_command = env.command
    checker = 0
    for _ in range(10):
        obs, _ = env.reset()
        checker += env.command is not initial_command
    
    assert checker != 0
    
    env.set_and_fix_command(new_command=3)

    assert env.is_command_fixed is True

    initial_command = env.command
    checker = 0
    for _ in range(10):
        obs, _ = env.reset()
        checker += env.command is not initial_command

    assert checker == 0


def test_run_domain_randomization():
    env = RealAntBaseEnv(domain_randomization=True)
    for _ in range(3):
        env.reset()
        for i in range(10):
            env.render()
            env.step(env.action_space.sample())
    env.close()


def test_run_domain_randomization_command():
    env = RealAntBaseCommandEnv(domain_randomization=True)
    for _ in range(4):
        env.reset()
        for i in range(10):
            env.render()
            env.step(env.action_space.sample())
    env.close()
    
def test_run_domain_randomization_navigation():
    env = RealAntBaseNavigationEnv(domain_randomization=True)
    for _ in range(3):
        env.reset()
        for i in range(10):
            env.render()
            env.step(env.action_space.sample())
    env.close()


def test_run_domain_randomization_playroom():
    env = RealAntBasePlayroomEnv(domain_randomization=True)
    for _ in range(10):
        env.reset()
        for i in range(100):
            env.render()
            env.step(env.action_space.sample())
    env.close()


def test_dim_playroom_obs_delay():
    obs_delay = 5
    obs_stack = 3
    
    env = RealAntBasePlayroomEnv(action_as_obs=True, obs_delay=obs_delay, obs_stack=obs_stack)
    obs, _ = env.reset()
    
    robo_obs = 3 + 3 + 8 + 8  # sensor x 8 + joint x 2
    
    # action as observation is only about the agent's motor action
    assert len(env.observation_space.high) == 2 + robo_obs * obs_stack + 8 + 20 + 2
    assert len(env.action_space.high) == 8
    assert len(obs) == 2 + robo_obs * obs_stack + 8 + 20 + 2
    assert len(env.action_space.sample()) == 8


def test_dim_playroom2_obs_delay():
    obs_delay = 1
    obs_stack = 3
    
    env = RealAntBasePlayroomEnv2(joint_only=True, action_as_obs=True, obs_delay=obs_delay, obs_stack=obs_stack, no_position_obs=False)
    obs, _ = env.reset()
    
    robo_obs = 8 * obs_stack + 8 # joint only
    
    # action as observation is only about the agent's motor action
    assert len(env.observation_space.high) == robo_obs + (20 + 2) + (8 + 1) # proprio, extero, intero
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs + (20 + 2) + (8 + 1) # proprio, extero, intero
    assert len(env.action_space.sample()) == 8

def test_dim_playroom2_obs_no_position():
    obs_delay = 1
    obs_stack = 3
    
    env = RealAntBasePlayroomEnv2(joint_only=True, action_as_obs=True, obs_delay=obs_delay, obs_stack=obs_stack, no_position_obs=True )
    obs, _ = env.reset()
    
    robo_obs = 8 * obs_stack + 8 # joint only + action_as_obs
    
    # action as observation is only about the agent's motor action
    assert len(env.observation_space.high) == robo_obs + 20 + (8 + 1) # proprio, extero, intero
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs + 20 + (8 + 1) # proprio, extero, intero
    assert len(env.action_space.sample()) == 8

def test_dim_command_obs_delay():
    obs_delay = 5
    obs_stack = 3

    env = RealAntBaseCommandEnv(action_as_obs=True, obs_delay=obs_delay, obs_stack=obs_stack)
    obs, _ = env.reset()
    
    robo_obs = 3 + 3 + 8 + 8   # sensor x 8 + joint x 2
    
    # action as observation is only about the agent's motor action
    assert len(env.observation_space.high) == robo_obs * obs_stack + 8 + len(Commands)
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs * obs_stack + 8 + len(Commands)
    assert len(env.action_space.sample()) == 8


def test_dim_navigation_obs_delay():
    obs_delay = 5
    obs_stack = 3
    
    env = RealAntBaseNavigationEnv(action_as_obs=True, obs_delay=obs_delay, obs_stack=obs_stack)
    obs, _ = env.reset()
    
    robo_obs = 3 + 3 + 8 + 8  # sensor x 8 + joint x 2
    
    # action as observation is only about the agent's motor action
    assert len(env.observation_space.high) == robo_obs * obs_stack + 8 + 20 + 2
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs * obs_stack + 8 + 20 + 2
    assert len(env.action_space.sample()) == 8


def test_dim_realant():
    obs_delay = 5
    obs_stack = 3

    env = RealAntBaseEnv(action_as_obs=True, obs_delay=obs_delay, obs_stack=obs_stack)
    obs, _ = env.reset()
    
    robo_obs = 3 + 3 + 8 + 8
    
    assert len(env.observation_space.high) == robo_obs * obs_stack + 8
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs * obs_stack + 8
    assert len(env.action_space.sample()) == 8


def test_dim_playroom2_average_temp():
    obs_delay = 1
    obs_stack = 3
    
    env = RealAntBasePlayroomEnv2(joint_only=True, action_as_obs=True, obs_delay=obs_delay, obs_stack=obs_stack, no_position_obs=False, average_temperature=True)
    obs, _ = env.reset()
    
    robo_obs = 8 * obs_stack + 8  # joint only
    
    # action as observation is only about the agent's motor action
    assert len(env.observation_space.high) == robo_obs + (20 + 2) + (1 + 1)  # proprio, extero, intero
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs + (20 + 2) + (1 + 1)  # proprio, extero, intero
    assert len(env.action_space.sample()) == 8


def test_dim_playroom2_energy_only():
    obs_delay = 1
    obs_stack = 3
    
    env = RealAntBasePlayroomEnv2(joint_only=True, action_as_obs=True, obs_delay=obs_delay, obs_stack=obs_stack, no_position_obs=False, energy_only=True)
    obs, _ = env.reset()
    
    robo_obs = 8 * obs_stack + 8  # joint only
    
    # action as observation is only about the agent's motor action
    assert len(env.observation_space.high) == robo_obs + (20 + 2) + 1  # proprio, extero, intero
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs + (20 + 2) + 1  # proprio, extero, intero
    assert len(env.action_space.sample()) == 8


def test_dim_playroom2_energy_only_run():
    obs_delay = 1
    obs_stack = 3
    
    env = RealAntBasePlayroomEnv2(joint_only=True, action_as_obs=True, obs_delay=obs_delay, obs_stack=obs_stack, no_position_obs=False, energy_only=True)
    obs, _ = env.reset()
    
    for _ in range(10):
        env.step(env.action_space.sample())


def test_random_position():
    env = RealAntBasePlayroomEnv2(random_position=True)
    
    error = 0
    for i in range(30):
        env.reset()
        first_init_pos = env.wrapped_env.data.qpos[:3].copy()
        
        env.reset()
        second_init_pos = env.wrapped_env.data.qpos[:3].copy()
        
        error += np.linalg.norm(first_init_pos - second_init_pos)

    assert error > 1.0
    env.close()
