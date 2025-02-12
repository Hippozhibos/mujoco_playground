import jax.numpy as jp

class MS_Jacobian:
  '''Simulate Musculoskeleton System, Transfer muscle activity to Torque of each Dof.'''

  def __init__(self, muscle_activity_size: int, action_size: int, seed: int = 42):
    """
    初始化 Jacobian 矩阵 J
    - `muscle_activity_size`: 输入维度 (肌肉活动向量的维度)
    - `action_size`: 输出维度 (即力矩的维度)
    - `seed`: 随机种子，保证每次训练时相同的初始化
    """
    self.muscle_activity_size = muscle_activity_size
    self.action_size = action_size
    key = jax.random.PRNGKey(seed)
    
    # 在训练开始时初始化一个固定的随机的Jacobian矩阵 J
    self.J = jax.random.normal(key, (action_size, muscle_activity_size)) * 0.1
    self.J = jp.array(self.J)  # 保证 J 是一个 jax 数组，防止不可变
    
  def compute_torque(self, muscle_activity: jp.ndarray) -> jp.ndarray:
    """
    计算从肌肉激活度到关节力矩的转换，并保证 torque 在 [-1, 1] 之间
    """
    # 计算 raw torque
    torque = jp.dot(self.J, muscle_activity)

    # 限制 torque 在 [-1, 1] 之间
    torque = jp.clip(torque, -1.0, 1.0)

    return torque