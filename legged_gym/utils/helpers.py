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

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _str_to_float_list(value):
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in value.split(",") if item.strip()]

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        if args.seed is not None:
            env_cfg.seed = args.seed
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        if args.use_drift is not None:
            cfg_train.algorithm.use_drift = args.use_drift
        if args.drift_model_warmup_updates is not None:
            cfg_train.algorithm.drift_model_warmup_updates = args.drift_model_warmup_updates
        if args.drift_actor_warmup_updates is not None:
            cfg_train.algorithm.drift_actor_warmup_updates = args.drift_actor_warmup_updates
        if args.drift_actor_loss_coef is not None:
            cfg_train.algorithm.drift_actor_loss_coef = args.drift_actor_loss_coef
        if args.positive_adv_threshold is not None:
            cfg_train.algorithm.positive_adv_threshold = args.positive_adv_threshold
        if args.min_positive_samples is not None:
            cfg_train.algorithm.min_positive_samples = args.min_positive_samples
        if args.use_top_positive_filter is not None:
            cfg_train.algorithm.use_top_positive_filter = args.use_top_positive_filter
        if args.positive_top_fraction is not None:
            cfg_train.algorithm.positive_top_fraction = args.positive_top_fraction
        if args.drift_step_size is not None:
            cfg_train.algorithm.drift_step_size = args.drift_step_size
        if args.max_drift_velocity_norm is not None:
            cfg_train.algorithm.max_drift_velocity_norm = args.max_drift_velocity_norm
        if args.max_drift_action_dist is not None:
            cfg_train.algorithm.max_drift_action_dist = args.max_drift_action_dist
        if args.drift_chunk_size is not None:
            cfg_train.algorithm.drift_chunk_size = args.drift_chunk_size
        if args.use_residual_drift is not None:
            cfg_train.algorithm.use_residual_drift = args.use_residual_drift
        if args.action_kernel_temperature is not None:
            cfg_train.algorithm.action_kernel_temperature = args.action_kernel_temperature
        if args.use_temperature_schedule is not None:
            cfg_train.algorithm.use_temperature_schedule = args.use_temperature_schedule
        if args.action_kernel_temperature_start is not None:
            cfg_train.algorithm.action_kernel_temperature_start = args.action_kernel_temperature_start
        if args.action_kernel_temperature_end is not None:
            cfg_train.algorithm.action_kernel_temperature_end = args.action_kernel_temperature_end
        if args.action_kernel_temperature_schedule_start is not None:
            cfg_train.algorithm.action_kernel_temperature_schedule_start = args.action_kernel_temperature_schedule_start
        if args.action_kernel_temperature_schedule_end is not None:
            cfg_train.algorithm.action_kernel_temperature_schedule_end = args.action_kernel_temperature_schedule_end
        if args.advantage_temperature is not None:
            cfg_train.algorithm.advantage_temperature = args.advantage_temperature
        if args.advantage_clip is not None:
            cfg_train.algorithm.advantage_clip = args.advantage_clip
        if args.use_state_kernel is not None:
            cfg_train.algorithm.use_state_kernel = args.use_state_kernel
        if args.state_kernel_temperature is not None:
            cfg_train.algorithm.state_kernel_temperature = args.state_kernel_temperature
        if args.state_feature_mode is not None:
            cfg_train.algorithm.state_feature_mode = args.state_feature_mode
        if args.use_multi_temperature is not None:
            cfg_train.algorithm.use_multi_temperature = args.use_multi_temperature
        if args.action_kernel_temperatures is not None:
            cfg_train.algorithm.action_kernel_temperatures = args.action_kernel_temperatures
        if args.normalize_drift_field is not None:
            cfg_train.algorithm.normalize_drift_field = args.normalize_drift_field
        if args.drift_field_norm_type is not None:
            cfg_train.algorithm.drift_field_norm_type = args.drift_field_norm_type
        if args.log_drift_debug is not None:
            cfg_train.algorithm.log_drift_debug = args.log_drift_debug
        if args.entropy_coef is not None:
            cfg_train.algorithm.entropy_coef = args.entropy_coef
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--use_drift", "type": _str_to_bool, "default": None, "help": "Enable or disable drifting module. Use True or False."},
        {"name": "--drift_model_warmup_updates", "type": int, "help": "Warmup updates before starting drift computation/logging."},
        {"name": "--drift_actor_warmup_updates", "type": int, "help": "Warmup updates before enabling drift loss."},
        {"name": "--drift_actor_loss_coef", "type": float, "help": "Weight of drift loss in PPO total loss."},
        {"name": "--positive_adv_threshold", "type": float, "help": "Minimum raw advantage to count as positive sample."},
        {"name": "--min_positive_samples", "type": int, "help": "Minimum number of positive samples required per minibatch."},
        {"name": "--use_top_positive_filter", "type": _str_to_bool, "default": None, "help": "Use only the top raw-advantage positive samples for drift."},
        {"name": "--positive_top_fraction", "type": float, "help": "Fraction of positive samples to keep when top-positive filtering is enabled."},
        {"name": "--drift_step_size", "type": float, "help": "Step size from actor mean toward drift target."},
        {"name": "--max_drift_velocity_norm", "type": float, "help": "Maximum norm of the drift field."},
        {"name": "--max_drift_action_dist", "type": float, "help": "Maximum target distance from current actor mean."},
        {"name": "--drift_chunk_size", "type": int, "help": "Rows per chunk for drift kernel computation. Lower this if drift causes CUDA OOM."},
        {"name": "--use_residual_drift", "type": _str_to_bool, "default": None, "help": "Use positive residuals action-old_mu instead of absolute positive actions."},
        {"name": "--action_kernel_temperature", "type": float, "help": "Temperature for action-space kernel."},
        {"name": "--use_temperature_schedule", "type": _str_to_bool, "default": None, "help": "Linearly schedule action kernel temperature."},
        {"name": "--action_kernel_temperature_start", "type": float, "help": "Initial action kernel temperature for schedule."},
        {"name": "--action_kernel_temperature_end", "type": float, "help": "Final action kernel temperature for schedule."},
        {"name": "--action_kernel_temperature_schedule_start", "type": int, "help": "Update at which action temperature schedule starts."},
        {"name": "--action_kernel_temperature_schedule_end", "type": int, "help": "Update at which action temperature schedule ends."},
        {"name": "--advantage_temperature", "type": float, "help": "Temperature for positive advantage logits."},
        {"name": "--advantage_clip", "type": float, "help": "Upper bound for positive advantage weighting."},
        {"name": "--use_state_kernel", "type": _str_to_bool, "default": None, "help": "Enable or disable state-conditioned kernel."},
        {"name": "--state_kernel_temperature", "type": float, "help": "Temperature for state-space kernel."},
        {"name": "--state_feature_mode", "type": str, "help": "State feature mode for the state kernel."},
        {"name": "--use_multi_temperature", "type": _str_to_bool, "default": None, "help": "Average drift fields from multiple action kernel temperatures."},
        {"name": "--action_kernel_temperatures", "type": _str_to_float_list, "help": "Comma-separated list of action kernel temperatures."},
        {"name": "--normalize_drift_field", "type": _str_to_bool, "default": None, "help": "Legacy switch kept for compatibility; current drift logic only uses max-norm clipping."},
        {"name": "--drift_field_norm_type", "type": str, "help": "Legacy compatibility option; current drift logic does not use field normalization."},
        {"name": "--log_drift_debug", "type": _str_to_bool, "default": None, "help": "Enable or disable detailed drift debug logging."},
        {"name": "--entropy_coef", "type": float, "help": "Entropy coefficient for PPO exploration. Overrides config file if provided."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args

def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
