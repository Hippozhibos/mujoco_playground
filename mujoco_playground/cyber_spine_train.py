## Define the training process of CyberSpine network

import jax
import jax.numpy as jp
import jax.nn as jnn
import optax
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state
from tensorboardX import SummaryWriter
import os

# 
logdir = './logs'
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(logdir=logdir)

# ============= 2. 定义通用 TrainState 和创建函数 =============
class TrainState(train_state.TrainState):
    """包含参数, 优化器状态, apply_fn 的训练状态"""
    pass

# def create_train_state(model: nn.Module, input_shape, seed=42, learning_rate=1e-3):
#     """
#     给定模型, 输入形状, 以及学习率, 
#     创建一个带有Adam优化器的 TrainState
#     """
#     key = jax.random.PRNGKey(seed)
#     dummy_input = jp.ones(input_shape)
#     # 初始化模型参数
#     params = model.init(key, dummy_input)
#     # 定义优化器
#     tx = optax.adam(learning_rate)
#     # 创建 TrainState
#     return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def create_train_state(model: nn.Module, params: dict, learning_rate: float = 1e-3) -> TrainState:
    """
    创建并返回一个带有优化器（Optax）的训练状态 TrainState
    """
    tx = optax.adam(learning_rate)  # 使用 Adam 优化器
    return TrainState.create(
        apply_fn=model.apply,  # 用于前向传播
        params=params,         # 神经网络参数
        tx=tx                  # 优化器
    )


# ============= 3. 定义联合训练步骤（同时更新CSP1与CC_net） =============

def mse_loss_fn(obs: jp.ndarray, obs_hat: jp.ndarray):
    """
    一个简单的损失函数示例：使用 MSE 计算 obs_hat 与 obs 的差异
    """
    # obs_hat = model.apply(params, muscle_activity)
    loss = jp.mean((obs_hat - obs) ** 2)
    return loss

def kl_divergence(p, q):
    """
    计算KL散度 D_KL(P || Q)
    p: 真实分布（obs_batch）
    q: 预测分布（obs_hat_batch）
    """
    # 防止数值问题，避免log(0)，在q中加入很小的值
    epsilon = 1e-8
    return jp.sum(p * jp.log((p + epsilon) / (q + epsilon)), axis=-1)

def log_loss(step, loss):
    writer.add_scalar('CyberSpine_loss', loss, step)
    print(f"CyberSpine_loss recorded:{step}_th")

@jax.jit
def train_step_joint(csp1_state: TrainState,
                     cc_state: TrainState,
                     obs_batch: jp.ndarray,
                     obs_hat_batch: jp.ndarray):
    """
    一次训练步骤, 同时更新 CSP1 和 CC_net.
    - action_batch: shape [batch_size, action_dim]
    - true_obs_batch: shape [batch_size, obs_dim]
    """

    # 计算对(csp1_params, cc_params)的梯度
    grad_fn = jax.value_and_grad(mse_loss_fn(obs_batch, obs_hat_batch), argnums=(0, 1))
    loss, (grads_csp1, grads_cc) = grad_fn(csp1_state.params, cc_state.params)

    # 分别更新CSP1和CC_net的参数
    new_csp1_state = csp1_state.apply_gradients(grads=grads_csp1)
    new_cc_state = cc_state.apply_gradients(grads=grads_cc)

    return new_csp1_state, new_cc_state, loss