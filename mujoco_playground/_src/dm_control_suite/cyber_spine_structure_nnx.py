## Define the structure of CyberSpine network

import jax
import jax.numpy as jp, random
import jax.nn as jnn
import optax
from flax import linen as nn
from flax import nnx as nnx
from flax.core import freeze, unfreeze
from flax.training import train_state

class Weights(nnx.Module):
  def __init__(self, kernel: jax.Array, bias: jax.Array):
    self.kernel, self.bias = nnx.Param(kernel), nnx.Param(bias)

def create_weights(seed: jax.Array):
  return Weights(
    kernel=random.uniform(random.key(seed), (2, 3)),
    bias=jp.zeros((3,)),
  )

class CyberSpine_P1(nnx.Module):
    # action_size: int
    # MSJcomplexity: int
    # hidden_size: int = 512  # 隐藏层大小，可调

    # def setup(self, action_size: int, MSJcomplexity: int, hidden_size: int = 512 ):
    #     """定义神经网络的层"""
    #     self.muscle_activity_size = action_size * MSJcomplexity
    #     print(f"Initializing with action_size={self.action_size}, muscle_activity_size={self.muscle_activity_size}")
    #     self.dense1 = nnx.Dense(hidden_size)
    #     self.dense2 = nnx.Dense(hidden_size)
    #     self.output_layer = nnx.Dense(self.muscle_activity_size)

    def __init__(self, action_size: int, MSJcomplexity: int, rngs:nnx.Rngs,hidden_size: int = 512):
        """定义神经网络的层"""
        self.muscle_activity_size = action_size * MSJcomplexity
        print(f"Initializing with action_size={action_size}, muscle_activity_size={self.muscle_activity_size}")
        self.dense1 = nnx.Linear(action_size, hidden_size, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.output_layer = nnx.Linear(hidden_size, self.muscle_activity_size, rngs=rngs)

    def __call__(self, action: jp.ndarray) -> jp.ndarray:
        """前向计算：从 action 计算 muscle_activity"""
        print(f"Input to dense1: {action.shape}")
        x = nnx.relu(self.dense1(action))
        print(f"Output of dense1: {x.shape}")
        x = nnx.relu(self.dense2(x))
        print(f"Output of dense2: {x.shape}")
        muscle_activity = nnx.sigmoid(self.output_layer(x))  # 限制在 [0,1]
        print(f"Output of network: {muscle_activity.shape}")
        return muscle_activity

    # def update(self, params, gradients):
    #     """预留的更新接口 (未来用于优化网络权重)"""
    #     # 目前不实现更新逻辑，留作未来训练使用
    #     raise NotImplementedError("CyberSpine_P1.update() 方法未实现")

# 初始化 CyberSpine_P1
def init_cyberspine_p1(action_size, MSJcomplexity, seed=42):
    """初始化神经网络并生成初始参数"""
    model = CyberSpine_P1(action_size=action_size, MSJcomplexity=MSJcomplexity)
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jp.ones((action_size,)))  # 用 1 作为 dummy 输入初始化参数
    return model, params


class CyberSpine_P2(nn.Module):
    muscle_activity_size: int
    hidden_size: int = 1024  # 隐藏层大小，可调

    def setup(self):
        """定义神经网络的层"""
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.hidden_size)
        self.output_layer = nn.Dense(self.muscle_activity_size)  # 输出预测的感觉输入

    def __call__(self, muscle_activity: jp.ndarray) -> jp.ndarray:
        """前向计算：从 P1 的输出的高维向量计算下一时刻的感觉输入"""
        print(f"Input to dense1 (muscle_activity): {muscle_activity.shape}")
        x = jnn.relu(self.dense1(muscle_activity))
        print(f"Output of dense1: {x.shape}")
        x = jnn.relu(self.dense2(x))
        print(f"Output of dense2: {x.shape}")
        predicted_sensory_input = self.output_layer(x)
        print(f"Output of P2 network (predicted sensory input): {predicted_sensory_input.shape}")
        return predicted_sensory_input

    # def update(self, params, gradients):
    #     """预留的更新接口 (未来用于优化网络权重)"""
    #     # 目前不实现更新逻辑，留作未来训练使用
    #     raise NotImplementedError("CyberSpine_P2.update() 方法未实现")

# 初始化 CyberSpine_P2
def init_cyberspine_p2(muscle_activity_size, seed=42):
    """初始化神经网络并生成初始参数"""
    model = CyberSpine_P2(muscle_activity_size=muscle_activity_size)
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jp.ones((muscle_activity_size,)))  # 用 1 作为 dummy 输入初始化参数
    return model, params


class CC_net(nnx.Module):
    """CC_net：从 CyberSpine_P1 输出的高维肌肉活动预测 obs_hat"""
    def __init__(self, muscle_activity_size: int, output_size: int, rngs:nnx.Rngs, hidden_size: int = 512):
        """定义神经网络的层"""
        self.dense1 = nnx.Linear(muscle_activity_size, hidden_size, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.output_layer = nnx.Linear(hidden_size, output_size, rngs=rngs)  # 输出预测的观察值（obs_hat）

    def __call__(self, muscle_activity: jp.ndarray) -> jp.ndarray:
        """
        前向计算：从 CyberSpine_P1 输出的高维肌肉活动预测 obs_hat
        """
        x = nnx.relu(self.dense1(muscle_activity))
        x = nnx.relu(self.dense2(x))
        obs_hat = self.output_layer(x)  # 预测观察值
        return obs_hat

def init_cc_net(muscle_activity_size: int, output_size: int, seed: int = 42):
    """
    初始化 CC_net 并返回 (model, params) 元组
    
    muscle_activity_size: 来自P1的肌肉活动向量维度
    output_size: 需要预测的观察值维度
    """
    model = CC_net(output_size=output_size)
    key = jax.random.PRNGKey(seed)
    # 用全1的 muscle_activity 作为 dummy 输入来初始化参数
    params = model.init(key, jp.ones((muscle_activity_size,)))  
    return model, params
