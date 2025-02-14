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

            J = self.generate_jacobian_matrix(seed)

            J = self.rearrange(J) # 根据每个肌肉列的主要关节（行）重新排列列索引

            np.save(self.file_path, np.array(J))  # 存储 J
            return jp.array(J)
        
    def rearrange(self, J:jp.ndarray) -> jp.ndarray:
        # 确定每个肌肉列的主要关节（行）
        main_joints = np.argmax(np.abs(J), axis=0)

        # 创建一个字典，将列索引按主要关节分组
        joint_to_columns = {}
        for col, joint in enumerate(main_joints):
            joint_to_columns.setdefault(joint, []).append(col)

        # 根据行的顺序重新排列列索引
        sorted_columns = []
        for joint in sorted(joint_to_columns.keys()):
            sorted_columns.extend(joint_to_columns[joint])

        # 重新排列 J 的列，生成 J_rearrange
        J_rearrange = J[:, sorted_columns]

        # 保存结果
        # np.save('J_rearrange.npy', J_rearrange)
        return J_rearrange
        
    def generate_jacobian_matrix_wo_bio(self, seed: int) -> jp.ndarray:
      '''generate J without bio-restrain'''
      key = jax.random.PRNGKey(seed)
      J = jp.zeros((self.action_size, self.muscle_activity_size))

      # 为每一列生成非零值，并分布在随机的位置
      for col in range(self.muscle_activity_size):
          # 为每列生成不同的随机数
          key, subkey = jax.random.split(key)
          count_key, rows_key, values_key = jax.random.split(subkey, 3)

          # 随机决定当前列中非零值的数量，至少为1
          non_zero_count = jax.random.randint(count_key, (1,), 1, self.action_size // 2 + 1).item()

          # 随机选择非零元素的行索引
          selected_rows = jax.random.choice(rows_key, jp.arange(self.action_size), (non_zero_count,), replace=False)

          # 随机生成非零值
          values = jax.random.normal(values_key, (non_zero_count,)) * 0.1

          # 将非零值赋给对应的矩阵位置
          J = J.at[selected_rows, col].set(values)

      # 对每一行进行 l2 归一化
      row_norms = jp.linalg.norm(J, axis=1, keepdims=True)
      row_norms = jp.where(row_norms == 0, 1.0, row_norms)  # 避免除以零
      J /= row_norms

      return J
    
    def generate_jacobian_matrix(self, seed: int) -> jp.ndarray:
        key = jax.random.PRNGKey(seed)
        J = jp.zeros((self.action_size, self.muscle_activity_size))

        # 为每列生成非零值，并分布在随机的位置
        for col in range(self.muscle_activity_size):
            key, subkey = jax.random.split(key)
            count_key, rows_key, values_key = jax.random.split(subkey, 3)

            # 生成非零值的数量，服从给定分布：众数为3，最小值为1
            non_zero_count = self.sample_non_zero_count(count_key)

            # # 随机选择非零元素的行索引
            # selected_rows = jax.random.choice(rows_key, jp.arange(self.action_size), (non_zero_count,), replace=False)

            # 生成非零值的位置
            selected_rows = self.sample_non_zero_positions(subkey, col, non_zero_count)

            # 随机生成非零值
            values = jax.random.normal(values_key, (non_zero_count,)) * 0.1

            # 将非零值赋给对应的矩阵位置
            J = J.at[selected_rows, col].set(values)

        # # 对每行进行非零值位置分布优化（接近最大值）
        # J = self.optimize_row_distribution(J)

        # # Normalize每一行
        # row_norms = jp.linalg.norm(J, axis=1, keepdims=True)
        # row_norms = jp.where(row_norms == 0, 1.0, row_norms)  # 避免除以零
        # J /= row_norms

        # 归一化每一行
        for row in range(self.action_size):
            J_row = J[row]
            J_pos = jp.maximum(J_row, 0)
            J_neg = jp.minimum(J_row, 0)

            # 归一化正值部分
            pos_sum = jp.sum(J_pos)
            if pos_sum > 0:
                J_pos /= pos_sum

            # 归一化负值部分
            neg_sum = jp.sum(J_neg)
            if neg_sum < 0:
                J_neg /= -neg_sum

            # 组合正负部分
            J = J.at[row].set(J_pos + J_neg)

        return J

    def sample_non_zero_count(self, key: jp.ndarray) -> int:
      """
      通过高斯分布生成每列的非零元素数量，众数为3，最小为1。
      """
      mean = 3
      stddev = 1  # 可以根据需要调整标准差
      count = jax.random.normal(key, (1,)) * stddev + mean
      count = jp.clip(count, 1, self.action_size)  # 限制在 [1, self.action_size] 之间
      return int(jp.round(count).item())  # 返回整数
    
    def sample_non_zero_positions(self, key: jp.ndarray, col: int, non_zero_count: int) -> jp.ndarray:
      """
      通过加权分布生成每一列非零值的位置。
      最大值的位置随机选择，其他非零值按距离最大值的位置加权选择。
      """
      # 随机选择一个位置作为最大值的位置
      max_pos = jax.random.randint(key, (1,), 0, self.action_size).item()

      # 计算每个位置的加权值，距离越近的权重越大
      # 使用一个指数衰减的权重函数，距离越远权重越小
      sigma = 1 # 
      distances = jp.abs(jp.arange(self.action_size) - max_pos)  # 计算每个位置与最大值位置的距离
      weights = jp.exp(-distances**2 / (2 * sigma**2))  # 高斯分布，控制标准差（sigma控制衰减速度,larger, slower）

      # 权重归一化
      weights /= jp.sum(weights)

      # 随机选择非零值的位置，概率由weights控制
      selected_positions = jax.random.choice(key, jp.arange(self.action_size), (non_zero_count,), replace=False, p=weights)

      return selected_positions

    def optimize_row_distribution(self, J: jp.ndarray) -> jp.ndarray:
        """
        对每一行的非零值进行位置优化，确保最大值邻近位置的非零值更多。
        """
        for row in range(self.action_size):
            row_values = J[row]

            # 找到最大值的索引
            max_value_index = jp.argmax(row_values)

            # 根据高斯分布来调整非零值的分布，最大值附近有更大的概率
            distances = jp.abs(jp.arange(self.muscle_activity_size) - max_value_index)
            probabilities = jp.exp(-distances**2 / (2 * 1.0**2))  # 使用高斯分布

            # 选择非零位置，按照概率分布选择
            key, subkey = jax.random.split(jax.random.PRNGKey(row))
            selected_positions = jax.random.choice(subkey, jp.arange(self.muscle_activity_size), 
                                                  shape=(int(jp.sum(probabilities > 0.1)),), 
                                                  replace=False, p=probabilities / jp.sum(probabilities))

            # 将这些位置赋值为非零值
            J = J.at[row, selected_positions].set(jax.random.normal(subkey, (len(selected_positions),)) * 0.1)

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

