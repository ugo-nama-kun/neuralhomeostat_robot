from gymnasium.envs.registration import register


register(
    id='RealAntBase-v0',
    entry_point='homeostatic_robot_sim_env.envs:RealAntBaseEnv',
)

register(
    id='OnTheBoardBase-v0',
    entry_point='homeostatic_robot_sim_env.envs:RealAntBaseOnTheBoardEnv',
)

register(
    id='CommandAntBase-v0',
    entry_point='homeostatic_robot_sim_env.envs:RealAntBaseCommandEnv',
)

register(
    id='HomeostaticRobotSim-v1',
    entry_point='homeostatic_robot_sim_env.envs:RealAntBasePlayroomEnv2',
)
