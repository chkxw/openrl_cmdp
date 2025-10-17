#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
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

from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from openrl_cmdp.algorithms.base_algorithm import BaseAlgorithm
from openrl_cmdp.modules.networks.utils.distributed_utils import reduce_tensor
from openrl_cmdp.modules.utils.util import get_grad_norm, huber_loss, mse_loss
from openrl_cmdp.utils.util import check


class PPOAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        cfg,
        init_module,
        agent_num: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self._use_share_model = cfg.use_share_model
        self.use_joint_action_loss = cfg.use_joint_action_loss
        super(PPOAlgorithm, self).__init__(cfg, init_module, agent_num, device)
        self.train_list = [self.train_ppo]
        self.use_deepspeed = cfg.use_deepspeed

        self.gamma = cfg.gamma

        self.cmdp_cost_keys = list(getattr(cfg, "cmdp_cost_keys", []))
        self.cmdp_cost_count = len(self.cmdp_cost_keys)
        self.cmdp_dual_step_size = float(getattr(cfg, "cmdp_dual_step_size", 0.01))
        self.cmdp_adv_norm_per_stream = bool(
            getattr(cfg, "cmdp_adv_norm_per_stream", True)
        )
        self.cmdp_match_discount_for_cost = bool(
            getattr(cfg, "cmdp_match_discount_for_cost", True)
        )
        self.cmdp_ema_momentum = float(getattr(cfg, "cmdp_ema_momentum", 0.9))
        self.cmdp_use_lambda_tanh = bool(
            getattr(cfg, "cmdp_use_lambda_tanh_param", False)
        )

        if self.cmdp_cost_count > 0:
            budgets = getattr(cfg, "cmdp_constraint_budgets", None)
            if not budgets:
                budgets = [0.0] * self.cmdp_cost_count
            if len(budgets) != self.cmdp_cost_count:
                raise ValueError(
                    "cmdp_constraint_budgets must match cmdp_cost_keys length"
                )
            self.cmdp_budgets = np.array(budgets, dtype=np.float32)

            lambda_init = getattr(cfg, "cmdp_lambda_init", None)
            if not lambda_init:
                lambda_init = [0.0] * self.cmdp_cost_count
            if len(lambda_init) != self.cmdp_cost_count:
                raise ValueError(
                    "cmdp_lambda_init must match cmdp_cost_keys length"
                )

            lambda_max = getattr(cfg, "cmdp_lambda_max", None)
            if not lambda_max:
                lambda_max = [10.0] * self.cmdp_cost_count
            if len(lambda_max) != self.cmdp_cost_count:
                raise ValueError(
                    "cmdp_lambda_max must match cmdp_cost_keys length"
                )
            self.cmdp_lambda_max = np.array(lambda_max, dtype=np.float32)

            if self.cmdp_use_lambda_tanh:
                scaled = np.clip(
                    np.divide(lambda_init, self.cmdp_lambda_max, out=np.zeros_like(self.cmdp_lambda_max), where=self.cmdp_lambda_max != 0),
                    0.0,
                    1.0,
                )
                scaled = np.clip(2.0 * scaled - 1.0, -0.999999, 0.999999)
                self.cmdp_lambda_param = np.arctanh(scaled.astype(np.float32))
                self.cmdp_lambda = self._get_cmdp_lambdas()
            else:
                self.cmdp_lambda = np.clip(
                    np.array(lambda_init, dtype=np.float32),
                    0.0,
                    self.cmdp_lambda_max,
                )
                self.cmdp_lambda_param = None

            self.cmdp_cost_ema = np.zeros(
                self.cmdp_cost_count, dtype=np.float32
            )
        else:
            self.cmdp_budgets = np.array([], dtype=np.float32)
            self.cmdp_lambda = np.array([], dtype=np.float32)
            self.cmdp_lambda_param = None
            self.cmdp_lambda_max = np.array([], dtype=np.float32)
            self.cmdp_cost_ema = np.array([], dtype=np.float32)

    def ppo_update(self, sample, turn_on=True):
        for optimizer in self.algo_module.optimizers.values():
            if not self.use_deepspeed:
                optimizer.zero_grad()

        (
            critic_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            action_masks_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                (
                    loss_list,
                    value_loss,
                    policy_loss,
                    dist_entropy,
                    ratio,
                ) = self.prepare_loss(
                    critic_obs_batch,
                    obs_batch,
                    rnn_states_batch,
                    rnn_states_critic_batch,
                    actions_batch,
                    masks_batch,
                    action_masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    value_preds_batch,
                    return_batch,
                    active_masks_batch,
                    turn_on,
                )
            for loss in loss_list:
                self.algo_module.scaler.scale(loss).backward()
        else:
            loss_list, value_loss, policy_loss, dist_entropy, ratio = self.prepare_loss(
                critic_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                masks_batch,
                action_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                value_preds_batch,
                return_batch,
                active_masks_batch,
                turn_on,
            )
            if self.use_deepspeed:
                if self._use_share_model:
                    for loss in loss_list:
                        self.algo_module.models["model"].backward(loss)
                else:
                    actor_loss = loss_list[0]
                    critic_loss = loss_list[1]
                    self.algo_module.models["policy"].backward(actor_loss)
                    self.algo_module.models["critic"].backward(critic_loss)
            else:
                for loss in loss_list:
                    loss.backward()

        # else:
        if self._use_share_model:
            actor_para = self.algo_module.models["model"].get_actor_para()
        else:
            actor_para = self.algo_module.models["policy"].parameters()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(actor_para, self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(actor_para)

        if self._use_share_model:
            critic_para = self.algo_module.models["model"].get_critic_para()
        else:
            critic_para = self.algo_module.models["critic"].parameters()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(critic_para, self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(critic_para)

        if self.use_amp:
            for optimizer in self.algo_module.optimizers.values():
                self.algo_module.scaler.unscale_(optimizer)

            for optimizer in self.algo_module.optimizers.values():
                self.algo_module.scaler.step(optimizer)

            self.algo_module.scaler.update()
        else:
            if self.use_deepspeed:
                if self._use_share_model:
                    self.algo_module.optimizers["model"].step()
                else:
                    self.algo_module.optimizers["policy"].step()
                    self.algo_module.optimizers["critic"].step()
            else:
                for optimizer in self.algo_module.optimizers.values():
                    optimizer.step()

        if self.world_size > 1:
            torch.cuda.synchronize()

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            actor_grad_norm,
            ratio,
        )

    def cal_value_loss(
        self,
        value_normalizer,
        values,
        value_preds_batch,
        return_batch,
        active_masks_batch,
    ):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )

        if (self._use_popart or self._use_valuenorm) and value_normalizer is not None:
            value_normalizer.update(return_batch)
            error_clipped = (
                value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (
                value_loss * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()
        # print(value_loss)
        # import pdb;pdb.set_trace()
        return value_loss

    def to_single_np(self, input):
        reshape_input = input.reshape(-1, self.agent_num, *input.shape[1:])
        return reshape_input[:, 0, ...]

    def construct_loss_list(self, policy_loss, dist_entropy, value_loss, turn_on):
        loss_list = []
        if turn_on:
            final_p_loss = policy_loss - dist_entropy * self.entropy_coef

            loss_list.append(final_p_loss)

        final_v_loss = value_loss * self.value_loss_coef
        loss_list.append(final_v_loss)

        return loss_list

    def _get_cmdp_lambdas(self):
        if self.cmdp_cost_count == 0:
            return np.array([], dtype=np.float32)
        if self.cmdp_use_lambda_tanh:
            self.cmdp_lambda = self.cmdp_lambda_max * 0.5 * (
                np.tanh(self.cmdp_lambda_param) + 1.0
            )
        return self.cmdp_lambda

    def _masked_normalize(self, data, masks, eps: float = 1e-5):
        if data.size == 0:
            return data
        masked = np.where(masks > 0.0, data, np.nan)
        mean = np.nanmean(masked, axis=(0, 1, 2), keepdims=True)
        std = np.nanstd(masked, axis=(0, 1, 2), keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)
        std = np.where(std < eps, 1.0, std)
        normalized = (data - mean) / (std + eps)
        normalized = np.where(masks > 0.0, normalized, 0.0)
        return normalized.astype(np.float32)

    def _cmdp_discount_factors(self, buffer):
        if self.cmdp_cost_count == 0:
            return None
        timesteps = buffer.cmdp_costs.shape[0]
        gamma_cost = (
            self.gamma
            if self.cmdp_match_discount_for_cost
            else float(getattr(self.cfg, "cmdp_cost_gamma", self.gamma))
        )
        powers = np.arange(timesteps, dtype=np.float32)
        discounts = np.power(gamma_cost, powers, dtype=np.float32).reshape(
            timesteps, 1, 1, 1
        )
        return discounts

    def cmdp_update_duals(self, buffer):
        if self.cmdp_cost_count == 0 or buffer.cmdp_costs is None:
            return {}

        discounts = self._cmdp_discount_factors(buffer)
        discounted_costs = buffer.cmdp_costs * discounts
        active_masks = buffer.active_masks[:-1]
        discounted_costs = discounted_costs * active_masks
        cost_totals = discounted_costs.sum(axis=0)
        mean_costs = cost_totals.mean(axis=(0, 1))

        self.cmdp_cost_ema = (
            self.cmdp_ema_momentum * self.cmdp_cost_ema
            + (1.0 - self.cmdp_ema_momentum) * mean_costs
        )

        violations = self.cmdp_cost_ema - self.cmdp_budgets

        if self.cmdp_use_lambda_tanh:
            self.cmdp_lambda_param += self.cmdp_dual_step_size * violations
            # ensure numerical stability when mapping back
            self.cmdp_lambda = self._get_cmdp_lambdas()
            clipped = np.clip(
                np.divide(
                    self.cmdp_lambda,
                    self.cmdp_lambda_max,
                    out=np.zeros_like(self.cmdp_lambda_max),
                    where=self.cmdp_lambda_max != 0,
                ),
                0.0,
                1.0,
            )
            clipped = np.clip(2.0 * clipped - 1.0, -0.999999, 0.999999)
            self.cmdp_lambda_param = np.arctanh(clipped)
            lambdas = self._get_cmdp_lambdas()
        else:
            self.cmdp_lambda = np.clip(
                self.cmdp_lambda + self.cmdp_dual_step_size * violations,
                0.0,
                self.cmdp_lambda_max,
            )
            lambdas = self.cmdp_lambda

        metrics = {}
        for idx, key in enumerate(self.cmdp_cost_keys):
            metrics[f"cmdp/cost/{key}"] = float(mean_costs[idx])
            metrics[f"cmdp/budget/{key}"] = float(self.cmdp_budgets[idx])
            metrics[f"cmdp/lambda/{key}"] = float(lambdas[idx])
            metrics[f"cmdp/constraint_violation/{key}"] = float(violations[idx])

        return metrics

    def prepare_loss(
        self,
        critic_obs_batch,
        obs_batch,
        rnn_states_batch,
        rnn_states_critic_batch,
        actions_batch,
        masks_batch,
        action_masks_batch,
        old_action_log_probs_batch,
        adv_targ,
        value_preds_batch,
        return_batch,
        active_masks_batch,
        turn_on,
    ):
        if self.use_joint_action_loss:
            critic_obs_batch = self.to_single_np(critic_obs_batch)
            rnn_states_critic_batch = self.to_single_np(rnn_states_critic_batch)
            critic_masks_batch = self.to_single_np(masks_batch)
            value_preds_batch = self.to_single_np(value_preds_batch)
            return_batch = self.to_single_np(return_batch)
            adv_targ = adv_targ.reshape(-1, self.agent_num, 1)
            adv_targ = adv_targ[:, 0, :]

        else:
            critic_masks_batch = masks_batch

        (
            values,
            action_log_probs,
            dist_entropy,
            policy_values,
        ) = self.algo_module.evaluate_actions(
            critic_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            action_masks_batch,
            active_masks_batch,
            critic_masks_batch=critic_masks_batch,
        )

        if self.use_joint_action_loss:
            action_log_probs_copy = (
                action_log_probs.reshape(-1, self.agent_num, action_log_probs.shape[-1])
                .sum(dim=(1, -1), keepdim=True)
                .reshape(-1, 1)
            )
            old_action_log_probs_batch_copy = (
                old_action_log_probs_batch.reshape(
                    -1, self.agent_num, old_action_log_probs_batch.shape[-1]
                )
                .sum(dim=(1, -1), keepdim=True)
                .reshape(-1, 1)
            )

            active_masks_batch = active_masks_batch.reshape(-1, self.agent_num, 1)
            active_masks_batch = active_masks_batch[:, 0, :]

            ratio = torch.exp(action_log_probs_copy - old_action_log_probs_batch_copy)
        else:
            ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

        if self.dual_clip_ppo:
            ratio = torch.min(ratio, self.dual_clip_coeff)

        surr1 = ratio * adv_targ
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        )

        surr_final = torch.min(surr1, surr2)

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(surr_final, dim=-1, keepdim=True) * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(surr_final, dim=-1, keepdim=True).mean()

        if self._use_policy_vhead:
            if isinstance(self.algo_module.models["actor"], DistributedDataParallel):
                policy_value_normalizer = self.algo_module.models[
                    "actor"
                ].module.value_normalizer
            else:
                policy_value_normalizer = self.algo_module.models[
                    "actor"
                ].value_normalizer
            policy_value_loss = self.cal_value_loss(
                policy_value_normalizer,
                policy_values,
                value_preds_batch,
                return_batch,
                active_masks_batch,
            )
            policy_loss = (
                policy_action_loss + policy_value_loss * self.policy_value_loss_coef
            )
        else:
            policy_loss = policy_action_loss

        # critic update
        if self._use_share_model:
            value_normalizer = self.algo_module.models["model"].value_normalizer
        elif isinstance(self.algo_module.models["critic"], DistributedDataParallel):
            value_normalizer = self.algo_module.models["critic"].module.value_normalizer
        else:
            value_normalizer = self.algo_module.get_critic_value_normalizer()

        reward_value_loss = self.cal_value_loss(
            value_normalizer,
            values[..., :1],
            value_preds_batch[..., :1],
            return_batch[..., :1],
            active_masks_batch,
        )

        total_value_loss = reward_value_loss
        if self.cmdp_cost_count > 0:
            for idx in range(self.cmdp_cost_count):
                cost_loss = self.cal_value_loss(
                    None,
                    values[..., idx + 1 : idx + 2],
                    value_preds_batch[..., idx + 1 : idx + 2],
                    return_batch[..., idx + 1 : idx + 2],
                    active_masks_batch,
                )
                total_value_loss = total_value_loss + cost_loss

        value_loss = total_value_loss

        loss_list = self.construct_loss_list(
            policy_loss, dist_entropy, value_loss, turn_on
        )
        return loss_list, value_loss, policy_loss, dist_entropy, ratio

    def get_data_generator(self, buffer, advantages):
        if self._use_recurrent_policy:
            if self.use_joint_action_loss:
                data_generator = buffer.recurrent_generator_v3(
                    advantages, self.num_mini_batch, self.data_chunk_length
                )
            else:
                data_generator = buffer.recurrent_generator(
                    advantages, self.num_mini_batch, self.data_chunk_length
                )
        elif self._use_naive_recurrent:
            data_generator = buffer.naive_recurrent_generator(
                advantages, self.num_mini_batch
            )
        else:
            data_generator = buffer.feed_forward_generator(
                advantages, self.num_mini_batch
            )
        return data_generator

    def train_ppo(self, buffer, turn_on):
        value_normalizer = None
        if self._use_popart or self._use_valuenorm:
            if self._use_share_model:
                value_normalizer = self.algo_module.models["model"].value_normalizer
            elif isinstance(self.algo_module.models["critic"], DistributedDataParallel):
                value_normalizer = self.algo_module.models[
                    "critic"
                ].module.value_normalizer
            else:
                value_normalizer = self.algo_module.get_critic_value_normalizer()

        if self.cmdp_cost_count == 0:
            if value_normalizer is not None:
                advantages = buffer.returns[:-1] - value_normalizer.denormalize(
                    buffer.value_preds[:-1]
                )
            else:
                advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

            if self._use_adv_normalize:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            advantages_copy = advantages.copy()
            advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
            advantages = advantages.astype(np.float32)
        else:
            active_masks = buffer.active_masks[:-1]

            reward_returns = buffer.returns[:-1, :, :, :1]
            if value_normalizer is not None:
                reward_value_preds = value_normalizer.denormalize(
                    buffer.value_preds[:-1, :, :, :1]
                )
            else:
                reward_value_preds = buffer.value_preds[:-1, :, :, :1]
            adv_reward = reward_returns - reward_value_preds

            cost_returns = buffer.returns[:-1, :, :, 1:]
            cost_value_preds = buffer.value_preds[:-1, :, :, 1:]
            adv_costs = cost_returns - cost_value_preds

            if self.cmdp_adv_norm_per_stream:
                adv_reward = self._masked_normalize(adv_reward, active_masks)
                adv_costs = self._masked_normalize(adv_costs, active_masks)

            lambdas = self._get_cmdp_lambdas().reshape(1, 1, 1, self.cmdp_cost_count)
            objective_adv = adv_reward - np.sum(adv_costs * lambdas, axis=-1, keepdims=True)

            if self._use_adv_normalize:
                objective_adv = self._masked_normalize(objective_adv, active_masks)

            advantages = objective_adv.astype(np.float32)

        train_info = {}

        train_info["value_loss"] = 0
        train_info["policy_loss"] = 0

        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["critic_grad_norm"] = 0
        train_info["ratio"] = 0
        if self.world_size > 1:
            train_info["reduced_value_loss"] = 0
            train_info["reduced_policy_loss"] = 0

        for _ in range(self.ppo_epoch):
            data_generator = self.get_data_generator(buffer, advantages)

            for sample in data_generator:
                (
                    value_loss,
                    critic_grad_norm,
                    policy_loss,
                    dist_entropy,
                    actor_grad_norm,
                    ratio,
                ) = self.ppo_update(sample, turn_on)

                if self.world_size > 1:
                    train_info["reduced_value_loss"] += reduce_tensor(
                        value_loss.data, self.world_size
                    )
                    train_info["reduced_policy_loss"] += reduce_tensor(
                        policy_loss.data, self.world_size
                    )

                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()

                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["critic_grad_norm"] += critic_grad_norm
                train_info["ratio"] += ratio.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        keys_to_average = list(train_info.keys())
        for k in keys_to_average:
            train_info[k] /= num_updates

        cmdp_metrics = {}
        if self.cmdp_cost_count > 0:
            cmdp_metrics = self.cmdp_update_duals(buffer)

        if cmdp_metrics:
            train_info.update(cmdp_metrics)

        return train_info

    def train(self, buffer, turn_on=True):
        train_info = {}
        for train_func in self.train_list:
            train_info.update(train_func(buffer, turn_on))

        for optimizer in self.algo_module.optimizers.values():
            if hasattr(optimizer, "sync_lookahead"):
                optimizer.sync_lookahead()

        return train_info
