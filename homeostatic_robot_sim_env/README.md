# trp_env
Two Resource Problem Environment with Mujoco


![trp](/uploads/c50634707761f6c680e3b5c2688de1cc/trp.png)


## Install
```shell
git clone ssh://git@gitlab:50002/n-yoshida/homeostatic_robot_sim_env.git
pip install -e homeostatic_robot_sim_env
```

## Usage

```python
import gymnasium
import playroom_env

env = gymnasium.make("homeostatic_robot_sim_env:AntTRP-v1")

env.seed(42)  # Seeding

obs, info = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    
    obs, reward, done, truncated, info = env.step(action)
```

## Environment List
```shell
AntTRP-v1

SmallAntTRP-v1

LowGearAntTRP-v1

SmallLowGearAntTRP-v1  <-- default use. above figure

SnakeTRP-v1

SmallSnakeTRP-v1

SwimmerTRP-v1

SmallSwimmerTRP-v1
```

## Optional Settings

```python
import gymnasium
import playroom_env

# set object range sensor bins to 20, modify reset condition to uniformly random samples in internal_random_range
env = gymnasium.make("SmallLowGearAntTRP-v1",
                     reward_setting="homeostatic",
                     internal_reset="random",
                     internal_random_range=(-0.1, 0.1),
                     n_bins=20)

# you can also set any number of food sources in the environment afterword
obs, info = env.reset(n_blue=3, n_red=10)
```

See `trp_env/envs/two_resource_env.py` to identify modifiable variables
CAUTION: those setting are modified as well in individual environments (for example `trp_env/envs/ant_trp_env.py`)

#### Reward Setting Options
```
"homeostatic_shaped": Default setting. quadratic "homeostatic" reward with state-based reward shaping.
"homeostatic":  Quadratic reward setting. assuming gaussian.
"one": return one while alive. -1 at the death.
"homeostatic_biased": "homeostatic" + bias. but add a constant as a baseline (use "reward_bias" configuration to set a bias)
"greedy": +1 reward if any food is obtained. not a homeostatic reward.
```

#### RGBD Vision Experiment

You can get the rendered image as the visual input for the agent.
Ego-centric vision setting: camera_id=0.

```python
import gymnasium
import playroom_env

env = gymnasium.make("SmallLowGearAntTRP-v1", vision=True, width=64, height=64)
env.reset()

# RGB (0-255, size=64x64x3)
vision = env.render(mode="rgb_array", camera_id=0)

# RGBD (0-255, size=64x64x4, depth values are also normalized into 0-255)
vision = env.render(mode="rgbd_array", camera_id=0)
```

![image](/uploads/550393d89c5bbbb0aef10a520816f115/image.png)
# playroom_env
