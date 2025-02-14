import jax
import jax.numpy as jp
import numpy as np
import os

class MS_Jacobian:
    '''Simulate Musculoskeleton System, Transfer muscle activity to Torque of each Dof.'''

    def __init__(self, MSJcomplexity: int, action_size: int, seed: int = 42, force_update: bool = False, file_path="J.npy"):
        """
        初始化 Jacobian 矩阵 J
        - `muscle_activity_size`: 输入维度 (肌肉活动向量的维度)
        - `action_size`: 输出维度 (即力矩的维度)
        - `seed`: 随机种子，保证可复现
        - `force_update`: 是否强制重新生成 J（默认为 False）
        - `file_path`: J 存储路径
        """
        self.MSJcomplexity = MSJcomplexity
        self.action_size = action_size
        self.muscle_activity_size = self.MSJcomplexity * self.action_size
        self.file_path = file_path

        # 加载或生成 Jacobian 矩阵
        self.J = self.load_or_generate_jacobian(seed, force_update)

    def load_or_generate_jacobian(self, seed: int, force_update: bool) -> jp.ndarray:
        """
        加载或生成 Jacobian 矩阵
        - 若 `force_update=False` 且文件存在，则直接加载 J
        - 若 `force_update=True` 或文件不存在，则生成新的 J 并存储
        """
        if os.path.exists(self.file_path) and not force_update:
            print(f"🔹 Loading existing J from {self.file_path}")
            return jp.array(np.load(self.file_path))  # 读取已存储的 J
        else:
            print("⚠️ Generating new J matrix")

            # 初始化一个全零的矩阵：每一行对应一个关节，每一列对应一个肌肉
            J = jp.zeros((self.action_size, self.muscle_activity_size))

            # 针对每个关节生成Jacobian矩阵
            for joint_index in range(self.action_size):
                # 为当前关节生成Jacobian矩阵
                J = self.generate_jacobian_for_joint(J, joint_index)


            # J = jax.random.normal(key, (self.action_size, self.muscle_activity_size)) * 0.1
            # 施加约束
            # J = self.apply_constraints(J)

            np.save(self.file_path, np.array(J))  # 存储 J
            return jp.array(J)  # 转换为 jax 数组
   
    
    def generate_jacobian_for_joint(self, J: jp.ndarray, joint_index: int, overlap_ratio: float = 0.3) -> jp.ndarray:
      """
      为某个关节生成Jacobian矩阵，并确保肌肉分布集中且相邻关节有一定重叠。
      - overlap_ratio: 相邻关节肌肉共享的比例 (0.0 ~ 1.0)
      """
      key = jax.random.PRNGKey(joint_index)  # 生成随机种子

      # 使用递减概率分布来生成每行的非零值数量

      # # 假设在 (MSJcomplexity, MSJcomplexity*2) 范围内，较小的值（接近 MSJcomplexity）出现的概率较高
      # muscle_count = jax.random.poisson(key, self.MSJcomplexity, shape=(1,)) + self.MSJcomplexity  # 使用泊松分布生成数量

      # 几何分布的 p 值，使得峰值接近 MSJcomplexity
      p = 0.5  # 控制下降速率，p 越小，下降越慢
      muscle_count = jax.random.geometric(key, p) + self.MSJcomplexity - 1  # 保证最小值为 MSJcomplexity

      muscle_count = jp.clip(muscle_count, self.MSJcomplexity, self.muscle_activity_size).item()  # 限制范围

      # 确保 muscle_count 是一个整数
      muscle_count = int(muscle_count)

      # 确定每行的起始列
      start_column = joint_index * self.MSJcomplexity  # 当前关节的起始列位置

      # 生成非零列索引：从start_column开始，连续选择muscle_count个列
      selected_indices = jp.arange(start_column, start_column + muscle_count)

      # 初始化当前行 Jacobian
      J_row = jp.zeros(self.muscle_activity_size)

      # 随机生成非零元素值
      muscle_contribution = jax.random.normal(key, (muscle_count,)) * 0.1

      # 赋值到选中的位置
      J_row = J_row.at[selected_indices].set(muscle_contribution)

      # 施加约束：确保正负部分归一化
      J_pos = jp.maximum(J_row, 0)
      J_neg = jp.minimum(J_row, 0)

      # 归一化正值部分
      J_pos_sum = jp.sum(J_pos)
      J_pos_normalized = J_pos / (J_pos_sum + 1e-6)  # 避免除零
      J_pos_normalized = J_pos_normalized * 1.0  # 限制最大加和为1

      # 归一化负值部分
      J_neg_sum = jp.sum(J_neg)
      J_neg_normalized = J_neg / (J_neg_sum - 1e-6)  # 避免除零
      J_neg_normalized = J_neg_normalized * -1.0  # 限制最小加和为-1

      # 组合正负部分
      J_row = J_pos_normalized + J_neg_normalized

      # 更新整体Jacobian矩阵J
      J = J.at[joint_index].set(J_row)

      return J


    def compute_torque(self, muscle_activity: jp.ndarray) -> jp.ndarray:
        """
        计算从肌肉激活度到关节力矩的转换，并保证 torque 在 [-1, 1] 之间
        """
        # 计算 raw torque
        torque = jp.dot(self.J, muscle_activity)

        # 限制 torque 在 [-1, 1] 之间
        torque = jp.clip(torque, -1.0, 1.0)

        return torque

