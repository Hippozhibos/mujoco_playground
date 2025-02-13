import jax
import jax.numpy as jp
import jax.nn as jnn
from flax import linen as nn
from flax.core import freeze, unfreeze

class CyberSpine_P1(nn.Module):
    action_size: int
    MSJcomplexity: int
    hidden_size: int = 128  # 隐藏层大小，可调

    def setup(self):
        """定义神经网络的层"""
        self.muscle_activity_size = self.action_size * self.MSJcomplexity
        print(f"Initializing with action_size={self.action_size}, muscle_activity_size={self.muscle_activity_size}")
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.hidden_size)
        self.output_layer = nn.Dense(self.muscle_activity_size)

    def __call__(self, action: jp.ndarray) -> jp.ndarray:
        """前向计算：从 action 计算 muscle_activity"""
        print(f"Input to dense1: {action.shape}")
        x = jnn.relu(self.dense1(action))
        print(f"Output of dense1: {x.shape}")
        x = jnn.relu(self.dense2(x))
        print(f"Output of dense2: {x.shape}")
        muscle_activity = jnn.sigmoid(self.output_layer(x))  # 限制在 [0,1]
        print(f"Output of network: {muscle_activity.shape}")
        return muscle_activity

    def update(self, params, gradients):
        """预留的更新接口 (未来用于优化网络权重)"""
        # 目前不实现更新逻辑，留作未来训练使用
        raise NotImplementedError("CyberSpine_P1.update() 方法未实现")

# 初始化 CyberSpine_P1
def init_cyberspine_p1(action_size, MSJcomplexity, seed=42):
    """初始化神经网络并生成初始参数"""
    model = CyberSpine_P1(action_size=action_size, MSJcomplexity=MSJcomplexity)
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jp.ones((action_size,)))  # 用 1 作为 dummy 输入初始化参数
    return model, params
