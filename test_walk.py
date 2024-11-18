'''
Walking robot test experiment for checking system works
'''

import torch
import torch.nn as nn

import gymnasium as gym

from model_utils import layer_init, NewGELU, BetaHead
from realant_env import RealAntEnv


class Agent(nn.Module):
    def __init__(self, n_stack=1, action_as_obs=False, leg_obs_only=False, joint_only=False, no_joint_vel=False):
        super().__init__()
        
        self.obs_size = 22 * n_stack + 8 * int(action_as_obs)
        if leg_obs_only:
            self.obs_size = 16 * n_stack + 8 * int(action_as_obs)
        if joint_only:
            self.obs_size = 8 * n_stack + 8 * int(action_as_obs)
        if no_joint_vel:
            self.obs_size = (8 + 6) * n_stack + 8 * int(action_as_obs)
            
        self.action_size = 8

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, 256)),
            nn.LayerNorm(256),
            NewGELU(),
            layer_init(nn.Linear(256, 256)),
            nn.LayerNorm(256),
            NewGELU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, 256)),
            nn.LayerNorm(256),
            NewGELU(),
            layer_init(nn.Linear(256, 256)),
            nn.LayerNorm(256),
            NewGELU(),
            BetaHead(256, self.action_size),
        )
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        probs = self.actor(x)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


agent = Agent(n_stack=3, action_as_obs=True, joint_only=True)
agent.eval()

env = RealAntEnv(obs_stack=3, action_as_obs=True, joint_only=True)
env = gym.wrappers.ClipAction(env)
env = gym.wrappers.RescaleAction(env, 0, 1)  # for Beta policy

print(env.observation_space)

FILE = "RealAntBase-v0__ppo_small__1__1686731101_final.pth"
agent.load_state_dict(torch.load(f"saved_model/{FILE}", map_location="cpu"))

obs, info = env.reset()

for _ in range(1000):
    obs_ = torch.Tensor(obs[None])
    with torch.no_grad():
        action = agent.get_action_and_value(obs_)[0].cpu().numpy()[0]
       
    obs, reward, terminated, truncated, info = env.step(action)
    
print("Finish.")
