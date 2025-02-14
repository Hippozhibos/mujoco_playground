import jax
import jax.numpy as jp
import jax.nn as jnn
import optax
from flax import linen as nn
from flax.training import train_state

# ============= 1. 定义模型 =============

# 1.1 CyberSpine_P1：低维动作 -> 高维肌肉活动
class CyberSpine_P1(nn.Module):
    action_size: int
    MSJcomplexity: int
    hidden_size: int = 1024  # 隐藏层大小

    def setup(self):
        self.muscle_activity_size = self.action_size * self.MSJcomplexity
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.hidden_size)
        self.output_layer = nn.Dense(self.muscle_activity_size)

    def __call__(self, action: jp.ndarray) -> jp.ndarray:
        x = jnn.relu(self.dense1(action))
        x = jnn.relu(self.dense2(x))
        muscle_activity = jnn.sigmoid(self.output_layer(x))  # 限制在 [0,1]
        return muscle_activity


# 1.2 CC_net：高维肌肉活动 -> 预测观察值 obs_hat
class CC_net(nn.Module):
    output_size: int  # 预测的观察值维度
    hidden_size: int = 512

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense2 = nn.Dense(self.hidden_size)
        self.output_layer = nn.Dense(self.output_size)

    def __call__(self, muscle_activity: jp.ndarray) -> jp.ndarray:
        x = jnn.relu(self.dense1(muscle_activity))
        x = jnn.relu(self.dense2(x))
        obs_hat = self.output_layer(x)
        return obs_hat


# ============= 2. 定义通用 TrainState 和创建函数 =============
class TrainState(train_state.TrainState):
    """包含参数, 优化器状态, apply_fn 的训练状态"""


def create_train_state(model: nn.Module, input_shape, seed=42, learning_rate=1e-3):
    """
    给定模型, 输入形状, 以及学习率, 
    创建一个带有Adam优化器的 TrainState
    """
    key = jax.random.PRNGKey(seed)
    dummy_input = jp.ones(input_shape)
    # 初始化模型参数
    params = model.init(key, dummy_input)
    # 定义优化器
    tx = optax.adam(learning_rate)
    # 创建 TrainState
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# ============= 3. 定义联合训练步骤（同时更新CSP1与CC_net） =============
@jax.jit
def train_step_joint(csp1_state: TrainState,
                     cc_state: TrainState,
                     action_batch: jp.ndarray,
                     true_obs_batch: jp.ndarray):
    """
    一次训练步骤, 同时更新 CSP1 和 CC_net.
    - action_batch: shape [batch_size, action_dim]
    - true_obs_batch: shape [batch_size, obs_dim]
    """

    def loss_fn(csp1_params, cc_params):
        # (1) CSP1 前向 -> muscle_activity
        muscle_activity = csp1_state.apply_fn(csp1_params, action_batch)
        # (2) CC_net 前向 -> obs_hat
        obs_hat = cc_state.apply_fn(cc_params, muscle_activity)
        # (3) MSE 损失
        loss = jp.mean((obs_hat - true_obs_batch) ** 2)
        return loss

    # 计算对(csp1_params, cc_params)的梯度
    grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))
    loss, (grads_csp1, grads_cc) = grad_fn(csp1_state.params, cc_state.params)

    # 分别更新CSP1和CC_net的参数
    new_csp1_state = csp1_state.apply_gradients(grads=grads_csp1)
    new_cc_state = cc_state.apply_gradients(grads=grads_cc)

    return new_csp1_state, new_cc_state, loss


# ============= 4. 演示如何使用 =============
if __name__ == "__main__":
    # (A) 创建CSP1模型及其TrainState
    csp1_model = CyberSpine_P1(action_size=5, MSJcomplexity=10)
    #  假设输入形状是 (batch, action_size), 这里省略 batch 只初始化 (1, 5)
    csp1_state = create_train_state(csp1_model, input_shape=(1, 5), learning_rate=1e-3)

    # (B) 创建CC_net模型及其TrainState
    cc_model = CC_net(output_size=4)  # 假设要预测4维obs
    cc_state = create_train_state(cc_model, input_shape=(1, csp1_model.muscle_activity_size), learning_rate=1e-3)

    # (C) 假设我们有一些训练数据
    batch_size = 8
    # actions: shape (batch_size, 5)
    action_batch = jp.ones((batch_size, 5))
    # true_obs: shape (batch_size, 4)
    true_obs_batch = jp.array([[1.0, 2.0, 3.0, 4.0]] * batch_size)

    # (D) 单步训练
    new_csp1_state, new_cc_state, loss = train_step_joint(
        csp1_state, cc_state,
        action_batch, true_obs_batch
    )

    print(f"Loss after train_step: {loss}")
    # 后续可以用 new_csp1_state, new_cc_state 替换旧的 state, 进行下个step
