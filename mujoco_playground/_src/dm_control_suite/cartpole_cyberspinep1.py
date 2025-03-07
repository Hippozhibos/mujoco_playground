# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cartpole environment."""

from typing import Any, Dict, Optional, Union
import warnings

import jax
import jax.numpy as jp
from flax import nnx
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

from mujoco_playground._src.dm_control_suite import ms_jacobian
from mujoco_playground._src.dm_control_suite import cyber_spine_structure
from mujoco_playground._src.dm_control_suite import cyber_spine_train
from mujoco_playground._src.dm_control_suite import cyber_spine_structure_nnx
from mujoco_playground._src.dm_control_suite import cyber_spine_train_nnx



_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "cartpole.xml"


def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      gpu_id=0,
      render_batch_size=512,
      render_width=64,
      render_height=64,
      enabled_geom_groups=[0, 1, 2],
      use_rasterizer=False,
      history=3,
  )


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.01,
      episode_length=1000,
      action_repeat=1,
      vision=False,
      vision_config=default_vision_config(),
  )


def _rgba_to_grayscale(rgba: jax.Array) -> jax.Array:
  """
  Intensity-weigh the colors.
  This expects the input to have the channels in the last dim.
  """
  r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray


class Balance(mjx_env.MjxEnv):
  """Cartpole environment with balance task."""

  _CART_RANGE = (-0.25, 0.25)
  _ANGLE_COSINE_RANGE = (0.995, 1)

  def __init__(
      self,
      swing_up: bool,
      sparse: bool,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides=config_overrides)
    self._vision = self._config.vision

    if swing_up:
      self._reset_randomize = self._reset_swing_up
    else:
      self._reset_randomize = self._reset_balance
    if sparse:
      self._get_reward = self._sparse_reward
    else:
      self._get_reward = self._dense_reward

    self._xml_path = _XML_PATH.as_posix()
    self._mj_model = mujoco.MjModel.from_xml_string(
        _XML_PATH.read_text(), common.get_assets()
    )
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model)
    self._post_init()

    self.MSJcomplexity = 10
    self.batch = 8

    # 调用 _get_obs 函数获取 obs
    rng = jax.random.PRNGKey(0)
    self.output_size = self.obs_size(rng)
    # self.output_size = 4

    self.ms_jacobian = ms_jacobian.MS_Jacobian(MSJcomplexity= self.MSJcomplexity, action_size=self.action_size)

    # # if self.csp1_model is None:
    # self.csp1_model = cyber_spine_structure.CyberSpine_P1(action_size=self.action_size, MSJcomplexity=self.MSJcomplexity, rngs=nnx.Rngs(0))
    # # self.csp1_params = self.csp1_model.init(jax.random.PRNGKey(42), jp.ones((self.action_size,)))
    
    # print("Caution: Reset csp1 params !!!!")
    # self.csp1_state = cyber_spine_train.create_train_state(self.csp1_model, self.csp1_params)

    # # if self.cc_model is None:
    # self.cc_model = cyber_spine_structure.CC_net(output_size=self.output_size)
    # self.muscle_activity_size = self.MSJcomplexity*self.action_size
    # self.cc_params = self.cc_model.init(jax.random.PRNGKey(42), jp.ones((self.muscle_activity_size,)))
    
    # print("Caution: Reset CC params !!!!")
    # self.cc_state = cyber_spine_train.create_train_state(self.cc_model, self.cc_params)

    ## nnx part
    self.csp1_model = cyber_spine_structure_nnx.CyberSpine_P1(action_size=self.action_size, MSJcomplexity=self.MSJcomplexity, rngs=nnx.Rngs(0))
    # self.csp1_params = self.csp1_model.init(jax.random.PRNGKey(42), jp.ones((self.action_size,)))
    
    print("Caution: Reset csp1 params !!!!")
    self.csp1_state = cyber_spine_train_nnx.create_train_state(self.csp1_model)

    self.muscle_activity_size = self.MSJcomplexity*self.action_size
    self.cc_model = cyber_spine_structure_nnx.CC_net(muscle_activity_size=self.muscle_activity_size, output_size=self.output_size, rngs=nnx.Rngs(0))
    
    # self.cc_params = self.cc_model.init(jax.random.PRNGKey(42), jp.ones((self.muscle_activity_size,)))
    
    print("Caution: Reset CC params !!!!")
    self.cc_state = cyber_spine_train_nnx.create_train_state(self.cc_model)

    ## CyberSpine_v2
    self.CyberSpine_v2 = cyber_spine_structure_nnx.CyberSpine_v2(self.action_size, 
                                                                 self.MSJcomplexity,
                                                                 self.output_size,
                                                                 rngs=nnx.Rngs(0))
    
    self.CyberSpine_v2_optimizer = cyber_spine_structure_nnx.get_optimizer(self.CyberSpine_v2)


    # self.buffer = []  # 用于存储 (action, obs, obs_hat) 对
    # self.buffer_size = 10  # 每满 10 对就进行一次更新

    self.loss_history = []

    if self._vision:
      try:
        # pylint: disable=import-outside-toplevel
        from madrona_mjx.renderer import BatchRenderer  # pytype: disable=import-error
      except ImportError:
        warnings.warn("Madrona MJX not installed. Cannot use vision with.")
        return
      self.renderer = BatchRenderer(
          m=self._mjx_model,
          gpu_id=self._config.vision_config.gpu_id,
          num_worlds=self._config.vision_config.render_batch_size,
          batch_render_view_width=self._config.vision_config.render_width,
          batch_render_view_height=self._config.vision_config.render_height,
          enabled_geom_groups=np.asarray(
              self._config.vision_config.enabled_geom_groups
          ),
          enabled_cameras=np.asarray([
              0,
          ]),
          add_cam_debug_geo=False,
          use_rasterizer=self._config.vision_config.use_rasterizer,
          viz_gpu_hdls=None,
      )

  def _post_init(self) -> None:
    slider_jid = self._mj_model.joint("slider").id
    self._slider_qposadr = self._mj_model.jnt_qposadr[slider_jid]
    hinge_1_jid = self._mj_model.joint("hinge_1").id
    self._hinge_1_qposadr = self._mj_model.jnt_qposadr[hinge_1_jid]

  def _reset_swing_up(self, rng: jax.Array) -> jax.Array:
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[self._slider_qposadr].set(0.01 * jax.random.normal(rng1))
    qpos = qpos.at[self._hinge_1_qposadr].set(
        jp.pi + 0.01 * jax.random.normal(rng2)
    )
    qpos = qpos.at[2:].set(
        0.1 * jax.random.uniform(rng3, (self.mjx_model.nq - 2,))
    )

    return qpos

  def _reset_balance(self, rng: jax.Array) -> jax.Array:
    rng1, rng2 = jax.random.split(rng, 2)

    qpos = jp.zeros(self.mjx_model.nq)
    qpos = qpos.at[self._slider_qposadr].set(
        jax.random.uniform(rng1, (), minval=-0.1, maxval=0.1)
    )
    qpos = qpos.at[1:].set(
        jax.random.uniform(
            rng2, (self.mjx_model.nq - 1,), minval=-0.034, maxval=0.034
        )
    )

    return qpos

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._reset_randomize(rng)

    rng, rng1 = jax.random.split(rng, 2)
    qvel = 0.01 * jax.random.normal(rng1, (self.mjx_model.nv,))

    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel)

    metrics = {
        "reward/upright": jp.zeros(()),
        "reward/centered": jp.zeros(()),
        "reward/small_control": jp.zeros(()),
        "reward/small_velocity": jp.zeros(()),
        "reward/cart_in_bounds": jp.zeros(()),
        "reward/angle_in_bounds": jp.zeros(()),
    }
    info = {"rng": rng}

    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name

    obs = self._get_obs(data, info)
    if self._vision:
      render_token, rgb, _ = self.renderer.init(data, self._mjx_model)
      info.update({"render_token": render_token})
      obs = _rgba_to_grayscale(rgb[0].astype(jp.float32)) / 255.0
      obs_history = jp.tile(obs, (self._config.vision_config.history, 1, 1))
      info.update({"obs_history": obs_history})
      obs = {"pixels/view_0": obs_history.transpose(1, 2, 0)}

    buffer = []

    return mjx_env.State(data, obs, reward, done, metrics, info, buffer)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # CyberSpine_P1: action -> muscle activity
    # muscle_activity = self.csp1_model(action)

    muscle_activity, obs_hat = self.CyberSpine_v2(action)
    # muscle_activity = self.CyberSpine_v2(action, predict = False)

    # MS_Jacobian: muscle activty -> torque
    torque = self.ms_jacobian.compute_torque(muscle_activity)

    # # MS_Jacobian: muscle activty -> torque
    # torque = self.ms_jacobian.compute_torque(action)

    data = mjx_env.step(self.mjx_model, state.data, torque, self.n_substeps)
    reward = self._get_reward(data, torque, state.info, state.metrics)  # pylint: disable=redefined-outer-name

    ## CCnet: muscle activity -> obs_hat
    # obs_hat = self.cc_model(muscle_activity)
 

    # data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    # reward = self._get_reward(data, action, state.info, state.metrics)  # pylint: disable=redefined-outer-name

    obs = self._get_obs(data, state.info)

    buffer = state.buffer
    buffer.append((obs, obs_hat))

    # # 如果缓冲区已满，进行一次训练步骤（更新 CSP1 和 CC_net）
    # if len(self.buffer) >= self.buffer_size:
    #     print("Training step triggered")
    #     # 进行训练（注意：我们可能需要提取合适的数据来进行训练）
    #     obs_batch, obs_hat_batch = zip(*self.buffer)  
    #     obs_batch = jp.stack(obs_batch) # 将列表转换为数组
    #     obs_hat_batch = jp.stack(obs_hat_batch)

    #     # 更新 CSP1 和 CC_net
    #     cyber_spine_train_nnx.train_step_joint(self.csp1_model, self.cc_model, obs_batch, obs_hat_batch)
    #     # self.cc_loss_history.append(loss)
    #     print("Training step completed")  # 确认训练步骤是否完成

    #     # 更新完毕后，清空缓冲区
    #     self.buffer.clear()

    if self._vision:
      _, rgb, _ = self.renderer.render(state.info["render_token"], data)
      # Update observation buffer
      obs_history = state.info["obs_history"]
      obs_history = jp.roll(obs_history, 1, axis=0)
      obs_history = obs_history.at[0].set(
          _rgba_to_grayscale(rgb[0].astype(jp.float32)) / 255.0
      )
      state.info["obs_history"] = obs_history
      obs = {"pixels/view_0": obs_history.transpose(1, 2, 0)}

    done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    done = done.astype(float)
    return mjx_env.State(data, obs, reward, done, state.metrics, state.info, buffer)
  
  def reset_buffer(self, state: mjx_env.State)-> mjx_env.State:
    buffer = []
    return mjx_env.State(state.data, state.obs, state.reward, state.done, state.metrics, state.info, buffer)

  def train_step(self,state: mjx_env.State, buffer_size:int=10) -> mjx_env.State:
      loss = None
      # 训练步骤：用全局缓冲区进行训练
      if len(state.buffer) >= buffer_size:
        print("Training step triggered")
        obs_batch, obs_hat_batch = zip(*state.buffer)
        obs_batch = jp.stack(obs_batch)
        obs_hat_batch = jp.stack(obs_hat_batch)

        # 更新模型
        # loss = cyber_spine_train_nnx.train_step_joint(self.csp1_model, self.cc_model, obs_batch, obs_hat_batch)
        # cyber_spine_train_nnx.train_step_joint(self.csp1_model, self.cc_model, obs_batch, obs_hat_batch)
        
        loss = cyber_spine_train_nnx.cyberspine_v2_train_step(self.CyberSpine_v2,
                                                              self.CyberSpine_v2_optimizer,
                                                              obs_batch,
                                                              obs_hat_batch)
        
        self.loss_history.append(loss)
        # self.loss = loss
        # loss_history.append(loss)

        state = self.reset_buffer(state)
        print("Training step completed")  # 确认训练步骤是否完成

      return loss, state
      

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    del info  # Unused.
    cart_position = data.qpos[self._slider_qposadr]
    pole_angle_cos = data.xmat[2:, 2, 2]  # zz.
    pole_angle_sin = data.xmat[2:, 0, 2]  # xz.
    return jp.concatenate([
        cart_position.reshape(1),
        pole_angle_cos,
        pole_angle_sin,
        data.qvel,
    ])

  def _dense_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del info  # Unused.
    pole_angle_cos = data.xmat[2, 2, 2]
    upright = (pole_angle_cos + 1) / 2
    metrics["reward/upright"] = upright

    cart_position = data.qpos[self._slider_qposadr]
    centered = reward.tolerance(cart_position, margin=2)
    centered = (1 + centered) / 2
    metrics["reward/centered"] = centered

    small_control = reward.tolerance(
        action[0], margin=1, value_at_margin=0, sigmoid="quadratic"
    )
    small_control = (4 + small_control) / 5
    metrics["reward/small_control"] = small_control

    angular_vel = data.qvel[1:]
    small_velocity = reward.tolerance(angular_vel, margin=5).min()
    small_velocity = (1 + small_velocity) / 2
    metrics["reward/small_velocity"] = small_velocity

    return upright * small_control * small_velocity * centered

  def _sparse_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
  ) -> jax.Array:
    del action, info  # Unused.

    cart_position = data.qpos[self._slider_qposadr]
    cart_in_bounds = reward.tolerance(cart_position, self._CART_RANGE)
    metrics["reward/cart_in_bounds"] = cart_in_bounds

    pole_angle_cos = data.xmat[2, 2, 2]
    angle_in_bounds = reward.tolerance(
        pole_angle_cos, self._ANGLE_COSINE_RANGE
    ).prod()
    metrics["reward/angle_in_bounds"] = angle_in_bounds

    return cart_in_bounds * angle_in_bounds

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu
  
  # @property
  # def action_size(self) -> int:
  #   return self.mjx_model.nu * self.MSJcomplexity
  
  def obs_size(self, rng: jax.Array) -> int:
    qpos = self._reset_randomize(rng)
    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qpos)  # 这应是你实际环境中的数据结构
    info = info = {"rng": rng} 
    obs = self._get_obs(data, info)
    # print(f"obs:{obs}")
    return len(obs)

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model