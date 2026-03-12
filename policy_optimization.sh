export MUJOCO_GL="glfw"

python ppo_cooling.py \
--seed 0 \
--env-id PlayroomBase-v2 \
--n-test-runs 5 \
--obs-stack 3 \
--obs-delay 0 \
--total-timesteps 300000000 \
--action-as-obs True \
--n-food 1 \
--num-envs 10 \
--num-steps 30000 \
--joint-only \
--position-cost 0 \
--ctrl-cost 0 \
--head-angle-cost 0 \
--no-position-obs \
--no-wall \
--random-position \
--thermal-model-version v3 \
--small \
--average-temperature \
--domain-randomization \
# --cuda True \
# --gpu 0


export MUJOCO_GL="egl"
