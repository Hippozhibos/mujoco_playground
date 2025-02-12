import os
import subprocess
import functools
from datetime import datetime
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import jax
from jax import numpy as jp
from brax import base
from brax import envs
from brax import math
from brax.base import State as PipelineState
from brax.envs.base import Env, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from IPython.display import HTML, clear_output
from mujoco_playground import registry
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground import wrapper
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from orbax import checkpoint as ocp
import orbax.checkpoint
import tensorflow as tf
from datetime import datetime

# 设置 TensorBoard 的 log 目录
log_dir = "logs/ppo"
summary_writer = tf.summary.create_file_writer(log_dir)

# Check GPU availability
if subprocess.run('nvidia-smi').returncode:
    raise RuntimeError(
        'Cannot communicate with GPU. '
        'Make sure you are using a GPU runtime.'
    )

# Set environment variable to use GPU rendering
print('Setting environment variable to use GPU rendering:')
os.environ['MUJOCO_GL'] = 'egl'

# Verify MuJoCo installation
try:
    print('Checking that the installation succeeded:')
    import mujoco
    mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e:
    raise e from RuntimeError(
        'Something went wrong during installation. Check the shell output above '
        'for more information.'
    )

print('Installation successful.')

# XLA optimization flag
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# Environment setup
env_name = "CyberMiceWalk"
camera_name = "side"

# Get environment config and load it
env_cfg = registry.get_default_config(env_name)
env = registry.load(env_name, config=env_cfg)

# Define PPO and SAC parameters
ppo_params = dm_control_suite_params.brax_ppo_config(env_name)
sac_params = dm_control_suite_params.brax_sac_config(env_name)

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

# # Progress function to update and plot metrics
# def progress(num_steps, metrics):
#     clear_output(wait=True)
#     times.append(datetime.now())
#     x_data.append(num_steps)
#     y_data.append(metrics["eval/episode_reward"])
#     y_dataerr.append(metrics["eval/episode_reward_std"])

#     plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
#     plt.ylim([0, 1100])
#     plt.xlabel("# environment steps")
#     plt.ylabel("reward per episode")
#     plt.title(f"y={y_data[-1]:.3f}")
#     plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

#     plt.show()

# 更新 progress 函数以记录数据到 TensorBoard
def progress(num_steps, metrics):
    # 每 1000 步记录一次
    if num_steps % 1000 == 0:
        with summary_writer.as_default():
            # 记录评估奖励
            tf.summary.scalar("eval/episode_reward", metrics["eval/episode_reward"], step=num_steps)
            tf.summary.scalar("eval/episode_reward_std", metrics["eval/episode_reward_std"], step=num_steps)

            # 记录其他指标，比如训练奖励、损失等
            tf.summary.scalar("train/episode_reward", metrics["train/episode_reward"], step=num_steps)

        print(f"Step {num_steps}: Recorded to TensorBoard")

# 创建并启动 TensorBoard 可视化服务器
# %load_ext tensorboard
# %tensorboard --logdir logs/ppo

# 指定 checkpoint 保存路径
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_manager = ocp.CheckpointManager(checkpoint_dir)

# 创建 Checkpoint Manager
checkpoint_manager = ocp.Checkpointer(checkpoint_dir)

# 训练时保存 checkpoint
def save_checkpoint(params, step):
    checkpoint_manager.save(step, {"params": params})
    print(f"Checkpoint saved at step {step}")

def load_checkpoint():
    """加载 checkpoint"""
    latest_step = checkpoint_manager.latest_step()
    if latest_step is not None:
        state = checkpoint_manager.restore(latest_step)
        print(f"Checkpoint loaded from step {latest_step}")
        return state["params"], latest_step
    else:
        print("No checkpoint found. Starting from scratch.")
        return None, 0
    
# 使用加载的 checkpoint 或从头开始训练
params, start_step = load_checkpoint()
    
# Define the training function
ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    progress_fn=progress
)

# Train PPO agent
make_inference_fn, params, metrics = train_fn(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    initial_params=params,  # 使用加载的参数
    start_step=start_step    # 从上次 checkpoint 开始
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

# JIT-compile reset and step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

# Simulation loop
rng = jax.random.PRNGKey(42)
rollout = []
n_episodes = 1

for _ in range(n_episodes):
    state = jit_reset(rng)
    rollout.append(state)
    for i in range(env_cfg.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)
    save_checkpoint(params, _)

save_checkpoint(params, ppo_params["num_timesteps"])

# Rendering and visualization
render_every = 1
frames = env.render(rollout[::render_every], camera=camera_name)
rewards = [s.reward for s in rollout]

# Display the simulation video
media.show_video(frames, fps=1.0 / env.dt / render_every)
