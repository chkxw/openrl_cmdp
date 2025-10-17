#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""""

import torch
import torch.nn as nn

from openrl_cmdp.buffers.utils.util import get_critic_obs_space
from openrl_cmdp.modules.networks.base_value_network import BaseValueNetwork
from openrl_cmdp.modules.networks.utils.cnn import CNNBase
from openrl_cmdp.modules.networks.utils.mix import MIXBase
from openrl_cmdp.modules.networks.utils.mlp import MLPBase, MLPLayer
from openrl_cmdp.modules.networks.utils.popart import PopArt
from openrl_cmdp.modules.networks.utils.rnn import RNNLayer
from openrl_cmdp.modules.networks.utils.util import init
from openrl_cmdp.utils.util import check_v2 as check


class ValueNetwork(BaseValueNetwork):
    def __init__(
        self,
        cfg,
        input_space,
        action_space=None,
        use_half=False,
        device=torch.device("cpu"),
        extra_args=None,
    ):
        super(ValueNetwork, self).__init__(cfg, device)

        self.hidden_size = cfg.hidden_size
        self._use_orthogonal = cfg.use_orthogonal
        self._activation_id = cfg.activation_id
        self._use_naive_recurrent_policy = cfg.use_naive_recurrent_policy
        self._use_recurrent_policy = cfg.use_recurrent_policy
        self._use_influence_policy = cfg.use_influence_policy
        self._use_popart = cfg.use_popart
        self._use_fp16 = cfg.use_fp16 and cfg.use_deepspeed
        self._influence_layer_N = cfg.influence_layer_N
        self._recurrent_N = cfg.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._cmdp_cost_keys = list(getattr(cfg, "cmdp_cost_keys", []))
        self._cmdp_cost_count = len(self._cmdp_cost_keys)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]

        critic_obs_shape = get_critic_obs_space(input_space)

        if "Dict" in critic_obs_shape.__class__.__name__:
            self._mixed_obs = True
            self.base = MIXBase(
                cfg, critic_obs_shape, cnn_layers_params=cfg.cnn_layers_params
            )
        else:
            self._mixed_obs = False
            self.base = (
                CNNBase(cfg, critic_obs_shape)
                if len(critic_obs_shape) == 3
                else MLPBase(
                    cfg,
                    critic_obs_shape,
                    use_attn_internal=True,
                    use_cat_self=cfg.use_cat_self,
                )
            )

        input_size = self.base.output_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                input_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
                rnn_type=cfg.rnn_type,
            )
            input_size = self.hidden_size

        if self._use_influence_policy:
            self.mlp = MLPLayer(
                critic_obs_shape[0],
                self.hidden_size,
                self._influence_layer_N,
                self._use_orthogonal,
                self._activation_id,
            )
            input_size += self.hidden_size

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out_reward = init_(PopArt(input_size, 1, device=device))
        else:
            self.v_out_reward = init_(nn.Linear(input_size, 1))

        self.cost_value_heads = nn.ModuleList()
        for _ in range(self._cmdp_cost_count):
            self.cost_value_heads.append(init_(nn.Linear(input_size, 1)))

        self.to(device)

    def forward(self, critic_obs, rnn_states, masks):
        if self._mixed_obs:
            for key in critic_obs.keys():
                critic_obs[key] = check(critic_obs[key]).to(**self.tpdv)
        else:
            critic_obs = check(critic_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if self._use_fp16:
            critic_obs = critic_obs.half()

        critic_features = self.base(critic_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if self._use_influence_policy:
            mlp_critic_obs = self.mlp(critic_obs)
            critic_features = torch.cat([critic_features, mlp_critic_obs], dim=1)

        outputs = [self.v_out_reward(critic_features)]
        if self.cost_value_heads:
            cost_values = [head(critic_features) for head in self.cost_value_heads]
            outputs.extend(cost_values)

        values = torch.cat(outputs, dim=-1)

        return values, rnn_states
