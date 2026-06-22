# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 235
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # PPO 基础超参数
        value_loss_coef = 1.0          # value loss 权重。常用 0.5 / 1.0 / 2.0；太大容易更偏向拟合 value。
        use_clipped_value_loss = True  # 是否对 value loss 也做 PPO 式 clipping；True 更稳，False 有时更直接。
        clip_param = 0.2               # PPO policy ratio 裁剪范围。常用 0.1 / 0.2 / 0.3；越大更新越激进。
        entropy_coef = 0.01           # 熵奖励权重，鼓励探索。常用 0.0 / 0.005 / 0.01 / 0.02。
        num_learning_epochs = 5        # 每次 rollout 反复学习多少轮。常用 3 / 5 / 8；太大可能过拟合当前 batch。
        num_mini_batches = 4           # mini-batch 数；单个 mini-batch 大小 = num_envs * nsteps / nminibatches。
        learning_rate = 1.e-3          # PPO 学习率。常用 3e-4 / 5e-4 / 1e-3；不稳时优先减小。
        schedule = 'adaptive'          # 学习率策略：'fixed' 固定，'adaptive' 根据 KL 自动调节。
        gamma = 0.99                   # 折扣因子。常用 0.99 / 0.995 / 0.998；越大越重视长期回报。
        lam = 0.95                     # GAE(lambda)。常用 0.95 / 0.97 / 0.99；越大方差更高但偏差更小。
        desired_kl = 0.01              # adaptive schedule 目标 KL。常用 0.005 / 0.01 / 0.02。
        max_grad_norm = 1.             # PPO 总梯度裁剪上限。常用 0.5 / 1.0 / 2.0。

        # Drifting 总开关
        use_drift = True               # 是否启用 drifting guidance。baseline PPO 对照实验时设为 False。

        # Drifting 时序与损失
        drift_model_warmup_updates = 300   # drifting model/guidance 从第几个 update 开始训练或计算。
        drift_actor_warmup_updates = 400   # drift loss 从第几个 update 开始影响 actor；通常应 >= drift_model_warmup_updates。
        drift_actor_loss_coef = 0.001      # drift loss 权重。常用 5e-4 / 1e-3 / 5e-3；过大可能压制 PPO 主目标。

        # 正优势样本筛选
        positive_adv_threshold = 0.0   # 只使用 raw advantage 大于该阈值的样本。Stage 1/2 通常取 0.0。
        min_positive_samples = 64      # 一个 mini-batch 至少需要多少正样本，否则跳过 drift。常用 16 / 32 / 64。
        use_top_positive_filter = False # 是否只使用 raw advantage 最高的一部分正样本，减少弱正样本噪声。
        positive_top_fraction = 0.35   # top-positive 比例；常用 0.25 / 0.35 / 0.5。

        # Drift 场强度与安全限制
        drift_step_size = 0.1          # actor 朝 drift target 走多大一步。常用 0.05 / 0.1 / 0.2。
        max_drift_velocity_norm = 1.0  # 单样本 drift field 范数上限；太小会常被裁剪，太大可能不稳定。
        max_drift_action_dist = 1.5    # drift 后目标动作离当前 actor mean 的最大距离。常用 0.5 / 1.0 / 1.5。
        drift_chunk_size = 1024        # 分块计算 drift kernel，降低大 batch 下的显存峰值；OOM 时可试 512 / 256。
        use_residual_drift = False      # 使用 positive action residual = action - old_mu，而不是绝对 action target。

        # Action-space kernel 超参数
        action_kernel_temperature = 0.3    # 动作距离 softmax 温度。小一些更尖锐，大一些更平滑；常用 0.3 / 0.5 / 1.0。
        use_temperature_schedule = True    # 是否将 action kernel 温度从较宽逐渐收窄。
        action_kernel_temperature_start = 0.5
        action_kernel_temperature_end = 0.3
        action_kernel_temperature_schedule_start = 400
        action_kernel_temperature_schedule_end = 1000
        advantage_temperature = 2.0        # advantage logit 的温度。越小越偏向高 advantage 样本；常用 1.0 / 2.0 / 4.0。
        advantage_clip = 3.0               # 正优势权重截断上限，防止少量极端样本主导。常用 2.0 / 3.0 / 5.0。

        # State-conditioned kernel 超参数
        use_state_kernel = False            # 是否把状态距离加入 kernel。Stage 2 推荐 True；做 action-only 对照可设 False。
        state_kernel_temperature = 0.5     # 状态距离温度。常用 0.5 / 1.0 / 2.0；越小越强调相似状态。
        state_feature_mode = "obs_norm"    # 状态特征模式：
                                           # "obs_norm" = 直接对整维 obs 做 batch normalize；
                                           # "selected" = 只取前一部分观测特征，适合想降低状态核维度时尝试。

        # Multi-temperature kernel
        use_multi_temperature = False      # 是否同时用多组 action temperature 再平均 drift field。Stage 2 可选增强项。
        action_kernel_temperatures = [0.3, 0.5, 1.0]  # 多温度列表；常见可试 [0.2, 0.5]、[0.3, 0.5, 1.0]。

        # Drift field 归一化
        normalize_drift_field = False      # 兼容旧配置保留；当前实现默认只做安全裁剪，不再放大弱 drift field。
        drift_field_norm_type = "batch"    # 兼容旧配置保留；当前实现中不再使用该开关。

        # 日志开关
        log_drift_debug = True             # 是否记录更完整的 drift 调试指标。训练调参阶段建议 True。

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
