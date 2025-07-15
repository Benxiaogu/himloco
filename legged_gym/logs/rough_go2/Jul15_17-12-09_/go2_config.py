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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.0}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0
        control_test = True
    
    class commands( LeggedRobotCfg.commands ):
            curriculum = False
            max_curriculum = 2.0
            num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
            resampling_time = 10. # time before command are changed[s]
            heading_command = False # if true: compute ang vel command from heading error
            class ranges( LeggedRobotCfg.commands.ranges):
                lin_vel_x = [-1.5, 1.5] # min max [m/s]
                lin_vel_y = [-1.0, 1.0]   # min max [m/s]
                ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
                height = [0.10, 0.35]
                heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        # privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up


    # class domain_rand(LeggedRobotCfg.domain_rand):

    #     # TODO randomize_default_dof_pos, randomize_action_delay
    #     push_robots = True
    #     push_interval_s = 8
    #     max_push_vel_xy = 0.1  # 0.2

    #     action_noise = 0.05 #0.0 # 0.02
    #     action_delay = 0. # 0.1

    #     rand_interval_s = 10    ## 
    #     randomize_rigids_after_start = False     #控制link的friction和restituion的随机化开关
    #     randomize_friction = True             # xxw True
    #     friction_range = [0.1, 2]
    #     randomize_base_mass = True #
    #     # randomize_mass_range = [0.5, 1.5]         # 乘负载
    #     added_mass_range = [-5, 5]              # 加负载
    #     randomize_restitution = True           # TODO
    #     restitution_range = [0, 1.0]            #加到priv里的东西

    #     randomize_com_displacement = True      #加到priv里的东西
    #     com_displacement_range = [-0.05, 0.05]  # base link com的随机化范围
    #     randomize_each_link = False
    #     link_com_displacement_range_factor = 0.02   # link com的随机化比例(与com_displacement_range相乘)
        
    #     randomize_inertia = True    
    #     randomize_inertia_range = [0.8, 1.2]

    #     randomize_motor_strength = True      
    #     motor_strength_range = [0.9, 1.1]      

    #     randomize_PD_factor = False #             
    #     Kp_factor_range = [0.9, 1.1]            
    #     Kd_factor_range = [0.9, 1.1]

    #     randomize_motor_offset = False #目前是使用torque的offset

    #     default_motor_offset = [0, 0.0, 0,\
    #                             0, 0.0, 0]
    #     motor_offset_range = [-0.03, 0.03]

    #     randomize_default_dof_pos = False # defautl dof pos位置没变，但数值上有rand的偏差
    #     randomize_default_dof_pos_range = [-0.1, 0.1]

    #     gravity_rand_interval_s = 7
    #     gravity_impulse_duration = 1.0

    #     randomize_gravity = False # 建议不加
    #     gravity_range = [-1.0, 1.0]         #

    #     randomize_lag_timesteps = False      # 模拟delay，对于lag用于给历史的action
    #     lag_timesteps = 2       #2~4ms walk these ways 加固定action延迟

    #     randomize_torque_delay = False
    #     torque_delay_steps = 2

    #     # randomize_obs_delay = False #用队列加固定obs延迟
    #     # obs_delay_steps = 1

    #     # agibot
    #     add_lag = True
    #     randomize_lag_timesteps = True
    #     randomize_lag_timesteps_perstep = True
    #     lag_timesteps_range = [3, 7]

    #     add_dof_lag = True
    #     randomize_dof_lag_timesteps = True
    #     randomize_dof_lag_timesteps_perstep = False
    #     dof_lag_timesteps_range = [0, 2] # 1~4ms

    #     add_imu_lag = True # 现在是euler，需要projected gravity                    # 这个是 imu 的延迟
    #     randomize_imu_lag_timesteps = True
    #     randomize_imu_lag_timesteps_perstep = False         # 不常用always False
    #     imu_lag_timesteps_range = [0, 4] # 实际10~22ms

    #     randomize_coulomb_friction = False
    #     joint_stick_friction_range = [0.1, 0.2]
    #     joint_coulomb_friction_range = [0.0, 0.0]
        
    #     randomize_joint_friction = False
    #     randomize_joint_friction_each_joint = False
        
    #     default_joint_friction = [0.01, 0.002, 0.2, \
    #                              0.01, 0.002, 0.2, ]
    #     joint_friction_range = [0.8, 1.2]
    #     # joint_friction_range = [1.5, 1.5]
    #     joint_1_friction_range = [0.9, 1.1]
    #     joint_2_friction_range = [0.9, 1.1]
    #     joint_3_friction_range = [0.9, 1.1]
    #     joint_4_friction_range = [0.9, 1.1]
    #     joint_5_friction_range = [0.9, 1.1]
    #     joint_6_friction_range = [0.9, 1.1]

    #     randomize_joint_damping = False
    #     randomize_joint_damping_each_joint = True
    #     default_joint_damping = [1.8, 4, 0.03, \
    #                              1, 4, 0.03, ]
    #     joint_damping_range = [0.8, 1.2]
    #     joint_1_damping_range = [0.8, 1.2]
    #     joint_2_damping_range = [0.8, 1.2] 
    #     joint_3_damping_range = [0.8, 1.2]
    #     joint_4_damping_range = [0.8, 1.2]
    #     joint_5_damping_range = [0.8, 1.2]
    #     joint_6_damping_range = [0.8, 1.2]

    #     randomize_joint_armature = False 
    #     randomize_joint_armature_each_joint = False
    #     joint_armature_range = [0.8, 1.2]     # Factor
    #     joint_1_armature_range = [0.95, 1.05]
    #     joint_2_armature_range = [0.95, 1.05]
    #     joint_3_armature_range = [0.95, 1.05]
    #     joint_4_armature_range = [0.95, 1.05]
    #     joint_5_armature_range = [0.9, 1.1]
    #     joint_6_armature_range = [0.9, 1.1]
    #     default_joint_armature = [0.138096, 0.08, 0.08,\
    #                               0.138096, 0.08, 0.08] 


    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            termination = -0.0
            tracking_lin_vel = 3.0
            tracking_ang_vel = 2.0
            lin_vel_z = -1.0  # -2.0   # 太大会与高度控制冲突
            ang_vel_xy = -0.05

            orientation = -0.2
            dof_acc = -2.5e-7
            dof_vel = -2.5e-7
            joint_power = -2e-5

            # orientation_roll = -0.5
            orientation_pitch = -1.0
            base_height_penalize = -2.0
            base_height_encourage = 1.0

            # foot_clearance = -0.01
            action_rate = -0.04
            smoothness = -0.02
            feet_air_time = 0.5
            # feet_air_time_penalize = -1.0    # 惩罚长时间 in the air
            collision = -20.0
            feet_stumble = -0.0
            stand_still = -0.
            # stand_still_vel = -5.0

            # torques = -1e-5
            dof_pos_limits = -0.1
            dof_vel_limits = -0.1
            torque_limits = -0.001

            # similar_legged = 2.0
            # foot_clearance_f = -1e-4
            # foot_clearance_r = -1e-4
            # vel_y_zero_penalize = -0.1


        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.26
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.2
        clip_single_reward = 3

        min_foot_clearance = 0.27
        max_foot_clearance = 0.31

        tracking_similar_legged_sigma = 0.1

class Go2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'

  