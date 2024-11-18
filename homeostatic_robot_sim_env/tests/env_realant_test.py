import pytest
import numpy as np

from pytest import approx

from playroom_env.envs.command_env import Commands
from playroom_env.envs.realant import RealAntPlayroomEnv, RealAntEnv, RealAntCommandEnv, RealAntNavigationEnv


def test_instance():
    env = RealAntPlayroomEnv()


def test_reset_env():
    env = RealAntPlayroomEnv()
    env.reset()


def test_run_env():
    env = RealAntPlayroomEnv()
    for _ in range(3):
        env.reset()
        for i in range(10):
            env.render()
            env.step(env.action_space.sample())
    env.close()

def test_run_navigation_env():
    env = RealAntNavigationEnv()
    for _ in range(3):
        env.reset()
        for i in range(10):
            env.render()
            env.step(env.action_space.sample())
    env.close()

def test_dim():
    env = RealAntPlayroomEnv(obs_stack=1)
    obs, _ = env.reset()
    
    robo_obs = 8 # 3 + 3 + 8 + 8
    assert len(env.observation_space.high) == 2 + robo_obs + 20 + 2
    assert len(env.action_space.high) == 8
    assert len(obs) == 2 + robo_obs + 20 + 2
    assert len(env.action_space.sample()) == 8

def test_run_length_env():
    env = RealAntEnv(max_episode_steps=50)
    env.reset()
    steps = 0
    done = False
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        steps += 1
        
    assert steps == 50


def test_run_length_playroom_env():
    env = RealAntPlayroomEnv(max_episode_steps=50)
    env.reset()
    steps = 0
    done = False
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        steps += 1
    
    assert steps == 50


def test_dim_action_not_included():
    env = RealAntEnv(obs_stack=1, action_as_obs=False)
    obs, _ = env.reset()
    
    robo_obs = 8 # 3 + 3 + 8 + 8
    
    assert len(env.observation_space.high) == robo_obs
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs
    assert len(env.action_space.sample()) == 8
    
def test_dim_action_included():
    env = RealAntEnv(obs_stack=1, action_as_obs=True)
    obs, _ = env.reset()
    
    robo_obs =  8 # 3 + 3 + 8 + 8

    assert len(env.observation_space.high) == robo_obs + 8
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs + 8
    assert len(env.action_space.sample()) == 8


def test_dim_action_included_playroom():
    env = RealAntPlayroomEnv(obs_stack=1, action_as_obs=True)
    obs, _ = env.reset()
    
    robo_obs = 8 # 3 + 3 + 8 + 8

    # action as observation is only about the agent's motor action
    assert len(env.observation_space.high) == 2 + robo_obs + 20 + 2 + 8
    assert len(env.action_space.high) == 8
    assert len(obs) == 2 + robo_obs + 20 + 2 + 8
    assert len(env.action_space.sample()) == 8


def test_dim_command_env():
    env = RealAntCommandEnv(obs_stack=1)
    obs, _ = env.reset()
    
    robo_obs = 8 # 3 + 3 + 8 + 8
    
    # action as observation is only about the agent's motor action
    assert len(env.observation_space.high) == robo_obs + len(Commands)
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs + len(Commands)
    assert len(env.action_space.sample()) == 8


def test_command_env_fixed_command():
    env = RealAntCommandEnv(fixed_command=1)
    
    assert env.is_command_fixed is True

    for _ in range(20):
        obs, _ = env.reset()
        assert env.command is Commands.FORWARD

def test_command_env_fix_command():
    env = RealAntCommandEnv(fixed_command=None)

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
    env = RealAntEnv(domain_randomization=True)
    for _ in range(3):
        env.reset()
        for i in range(10):
            env.render()
            env.step(env.action_space.sample())
    env.close()


def test_run_domain_randomization_command():
    env = RealAntCommandEnv(domain_randomization=True)
    for _ in range(4):
        env.reset()
        for i in range(10):
            env.render()
            env.step(env.action_space.sample())
    env.close()
    
def test_run_domain_randomization_navigation():
    env = RealAntNavigationEnv(domain_randomization=True)
    for _ in range(3):
        env.reset()
        for i in range(10):
            env.render()
            env.step(env.action_space.sample())
    env.close()


def test_run_domain_randomization_playroom():
    env = RealAntPlayroomEnv(domain_randomization=True)
    for _ in range(3):
        env.reset()
        for i in range(10):
            env.render()
            env.step(env.action_space.sample())
    env.close()


def test_dim_playroom_obs_delay():
    obs_delay = 5
    obs_stack = 3
    
    env = RealAntPlayroomEnv(action_as_obs=True, obs_delay=obs_delay, obs_stack=obs_stack)
    obs, _ = env.reset()
    
    robo_obs = 8  # 3 + 3 + 8 + 8  # sensor x 8 + joint x 2
    
    # action as observation is only about the agent's motor action
    assert len(env.observation_space.high) == 2 + robo_obs * obs_stack + 8 + 20 + 2
    assert len(env.action_space.high) == 8
    assert len(obs) == 2 + robo_obs * obs_stack + 8 + 20 + 2
    assert len(env.action_space.sample()) == 8


def test_dim_command_obs_delay():
    obs_delay = 5
    obs_stack = 3

    env = RealAntCommandEnv(action_as_obs=True, obs_delay=obs_delay, obs_stack=obs_stack)
    obs, _ = env.reset()
    
    robo_obs = 8 # 3 + 3 + 8 + 8   # sensor x 8 + joint x 2
    
    # action as observation is only about the agent's motor action
    assert len(env.observation_space.high) == robo_obs * obs_stack + 8 + len(Commands)
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs * obs_stack + 8 + len(Commands)
    assert len(env.action_space.sample()) == 8


def test_dim_navigation_obs_delay():
    obs_delay = 5
    obs_stack = 3
    
    env = RealAntNavigationEnv(action_as_obs=True, obs_delay=obs_delay, obs_stack=obs_stack)
    obs, _ = env.reset()
    
    robo_obs = 8 # 3 + 3 + 8 + 8  # sensor x 8 + joint x 2
    
    # action as observation is only about the agent's motor action
    assert len(env.observation_space.high) == robo_obs * obs_stack + 8 + 20 + 2
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs * obs_stack + 8 + 20 + 2
    assert len(env.action_space.sample()) == 8


def test_dim_realant():
    obs_delay = 5
    obs_stack = 3

    env = RealAntEnv(action_as_obs=True, obs_delay=obs_delay, obs_stack=obs_stack)
    obs, _ = env.reset()
    
    robo_obs = 8 # 3 + 3 + 8 + 8
    
    assert len(env.observation_space.high) == robo_obs * obs_stack + 8
    assert len(env.action_space.high) == 8
    assert len(obs) == robo_obs * obs_stack + 8
    assert len(env.action_space.sample()) == 8
