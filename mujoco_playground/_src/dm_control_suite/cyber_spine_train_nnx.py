## Define the training process of CyberSpine network

import jax
import jax.numpy as jp
import jax.nn as jnn
import optax
from flax import linen as nn
from flax import nnx
from flax.core import freeze, unfreeze
from flax.training import train_state
from tensorboardX import SummaryWriter
import os

# 
# logdir = './logs'
# os.makedirs(logdir, exist_ok=True)
# writer = SummaryWriter(logdir=logdir)

# ============= 2. 定义通用 TrainState 和创建函数 =============
def create_train_state(model: nnx.Module, learning_rate: float = 1e-3):
    """
    创建并返回一个带有优化器（Optax）的训练状态 TrainState
    """
    tx = optax.adam(learning_rate)  # 使用 Adam 优化器
    return model, tx  # 返回模型和优化器

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

# def create_train_state(model: nnx.Module, learning_rate: float = 1e-3) -> TrainState:
#     """
#     创建并返回一个带有优化器（Optax）的训练状态 TrainState
#     """
#     tx = optax.adam(learning_rate)  # 使用 Adam 优化器
#     return TrainState.create(
#         apply_fn=model,  # 用于前向传播
#         tx=tx            # 优化器
#     )


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

@nnx.jit
def train_step_joint(csp1_model,
                     cc_model,
                     obs_batch: jp.ndarray,
                     obs_hat_batch: jp.ndarray):
    """
    一次训练步骤, 同时更新 CSP1 和 CC_net.
    - action_batch: shape [batch_size, action_dim]
    - true_obs_batch: shape [batch_size, obs_dim]
    """

    def loss_fn(csp1_model, cc_model, obs_batch, obs_hat_batch):
        return mse_loss_fn(obs_batch, obs_hat_batch)

    # loss = mse_loss_fn(obs_batch, obs_hat_batch)
    
    # grads = nnx.grad(loss_fn)(csp1_model, cc_model, obs_batch, obs_hat_batch)
    grads = nnx.grad(loss_fn)(csp1_model, cc_model, obs_batch, obs_hat_batch)

    _, csp1_params, csp1_rest = nnx.split(csp1_model, nnx.Param, ...)
    csp1_params = jax.tree.map(lambda p, g: p - 0.1 * g, csp1_params, grads)
    nnx.update(csp1_model, nnx.GraphState.merge(csp1_params, csp1_rest))

    # cc_grads = nnx.grad(loss_fn)(csp1_model, cc_model, obs_batch, obs_hat_batch)
    
    _, cc_params, cc_rest = nnx.split(cc_model, nnx.Param, ...)
    cc_params = jax.tree.map(lambda p, g: p - 0.1 * g, cc_params, grads)
    nnx.update(cc_model, nnx.GraphState.merge(cc_params, cc_rest))

    # return loss


@nnx.jit
def cyberspine_v2_train_step(model,
                             optimizer,
                            #  action_batch: jp.ndarray,
                             obs_batch: jp.ndarray,
                             obs_hat_batch:jp.ndarray):
    
    def loss_fn(model):
        loss = jp.mean((obs_hat_batch - obs_batch) ** 2)
        return loss
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)  # In place updates.

    # _, params, rest = nnx.split(model, nnx.Param, ...)
    # params = jax.tree.map(lambda p, g: p - 0.1 * g, params, grads)
    # nnx.update(model, nnx.GraphState.merge(params, rest))
    
    return loss
    