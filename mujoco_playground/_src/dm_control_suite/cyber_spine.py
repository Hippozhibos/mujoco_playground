import jax
import jax.numpy as jp
import jax.nn as jnn
import optax
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state

class CyberSpine_P1(nn.Module):
    action_size: int
    MSJcomplexity: int
    hidden_size: int = 1024  # 隐藏层大小，可调

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


class CC_net(nn.Module):
    """CC_net：从 CyberSpine_P1 输出的高维肌肉活动预测 obs_hat"""
    output_size: int  # 预测的输出维度（如：观察值的维度）
    hidden_size: int = 512  # 隐藏层的大小，可以调节

    def setup(self):
        """定义神经网络的层"""
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.hidden_size)
        self.output_layer = nn.Dense(self.output_size)  # 输出预测的观察值（obs_hat）

    def __call__(self, muscle_activity: jp.ndarray) -> jp.ndarray:
        """
        前向计算：从 CyberSpine_P1 输出的高维肌肉活动预测 obs_hat
        """
        x = nn.relu(self.dense1(muscle_activity))
        x = nn.relu(self.dense2(x))
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

# 定义一个TrainState来管理参数和优化器
class TrainState(train_state.TrainState):
    """包含参数, 优化器状态, apply_fn 的训练状态"""
    pass

def create_cc_net_train_state(model: nn.Module, params: dict, learning_rate: float = 1e-3):
    """
    创建并返回一个带有优化器（Optax）的训练状态 TrainState
    """
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,  # 用于前向传播
        params=params,         # 神经网络参数
        tx=tx                  # 优化器
    )

def mse_loss_fn(params: dict, model: nn.Module, muscle_activity: jp.ndarray, obs: jp.ndarray):
    """
    一个简单的损失函数示例：使用 MSE 计算 obs_hat 与 obs 的差异
    """
    obs_hat = model.apply(params, muscle_activity)
    loss = jp.mean((obs_hat - obs) ** 2)
    return loss

@jax.jit
def train_step(state: TrainState, model: nn.Module,
               muscle_activity: jp.ndarray,
               obs: jp.ndarray) -> (TrainState, jp.ndarray):
    """
    进行一次训练步 (forward + backward + update)
    返回新的 state 和当前 batch 的 loss
    """
    # 定义损失函数, 并计算梯度
    def loss_fn(params):
        loss = mse_loss_fn(params, model, muscle_activity, obs)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    # 用优化器更新参数
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss



def create_train_state(model, learning_rate=1e-4):
    """初始化训练状态"""
    params = model.init(jax.random.PRNGKey(42), jp.ones((1,)))
    tx = optax.adam(learning_rate)  # 使用 Adam 优化器
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# 为 CSP1 和 CC_net 创建不同的训练状态
csp1_state = create_train_state(CyberSpine_P1, learning_rate=1e-4)
cc_state = create_train_state(CC_net, learning_rate=1e-4)


# 损失函数，计算预期的obs_hat和真实obs之间的损失
def compute_loss(params, model, action, true_obs):
    predicted_obs = model.apply(params, action)
    return jp.mean((predicted_obs - true_obs) ** 2)

@jax.jit
def train_step(state, csp1_state, cc_state, action_batch, true_obs_batch):
    """训练步骤，更新 CSP1 和 CC_net"""
    
    # 计算损失和梯度
    def loss_fn(csp1_params, cc_params):
        # 计算 CSP1 输出的肌肉活动
        muscle_activity = csp1_state.apply_fn(csp1_params, action_batch)
        
        # 用 CC_net 预测观察值
        obs_hat = cc_state.apply_fn(cc_params, muscle_activity)
        
        # 计算 MSE 损失
        loss = jp.mean((obs_hat - true_obs_batch) ** 2)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
    loss, (grads_csp1, grads_cc) = grad_fn(csp1_state.params, cc_state.params)
    
    # 更新 CSP1 和 CC_net 的参数
    new_csp1_state = csp1_state.apply_gradients(grads=grads_csp1)
    new_cc_state = cc_state.apply_gradients(grads=grads_cc)
    
    return new_csp1_state, new_cc_state, loss


