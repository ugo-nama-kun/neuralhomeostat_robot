
import numpy as np

import gymnasium
import homeostatic_robot_sim_env


env = gymnasium.make("HomeostaticRobotSim-v1",
                     n_food=1,
                     joint_only=True,
                     action_as_obs=True,
                     obs_stack=3,
                     obs_delay=0,
                     # activity_range=1.5,
                     n_bins=20,
                     show_sensor_range=True,
                     no_wall=True,
                     position_homeostasis=False,
                     # realmode=True,
                     domain_randomization=True,
                     parametric_thermal_model=True,
                     energy_only=True,
                     random_position=True,
                     )

print(env.action_space)
print(env.observation_space)

while True:
    print("### RESET: ", env.random_position_at_reset)
    env.reset()
    steps = 0
    done = False
    action = np.ones(env.action_space.shape)
    while not done and steps < 50:
        action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)
        done = done | truncated
        
        env.render()
        
        steps += 1
        
    print(f"Finish at {steps}")
