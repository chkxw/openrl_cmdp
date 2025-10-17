# OpenRL-CMDP

[![Original OpenRL](https://img.shields.io/badge/Based%20on-OpenRL-blue)](https://github.com/OpenRL-Lab/openrl)
[![License](https://img.shields.io/github/license/chkxw/openrl_cmdp)](https://github.com/chkxw/openrl_cmdp/blob/main/LICENSE)

**This repository is a tailored version of [OpenRL](https://github.com/OpenRL-Lab/openrl) with extensions for Constrained Markov Decision Process (CMDP) training.**

For the original OpenRL framework, documentation, and features, please visit:
- **Original Repository**: [OpenRL-Lab/openrl](https://github.com/OpenRL-Lab/openrl)
- **Documentation**: [OpenRL Docs](https://openrl-docs.readthedocs.io/)

---

## OpenRL CMDP Extensions

This document summarizes the substantive changes made inside the `openrl_cmdp/` tree (excluding the external training scripts) to support objective-based CMDP training with multi-stream costs and dual updates.

### 1. Configuration Surface (`openrl_cmdp/configs/config.py`)
- Added command-line / config parser options for CMDP use cases:
  - `cmdp_cost_keys`, `cmdp_constraint_budgets`
  - Dual-update hyperparameters (`cmdp_dual_step_size`, `cmdp_lambda_init`, `cmdp_lambda_max`)
  - Behavioural toggles (`cmdp_use_lambda_tanh_param`, `cmdp_adv_norm_per_stream`, `cmdp_match_discount_for_cost`, `cmdp_ema_momentum`)
- These options propagate through `cfg` objects so every component (buffers, drivers, algorithms, networks) can query the same cost metadata.

### 2. Replay Buffer & Advantage Handling (`openrl_cmdp/buffers/replay_data.py`)
- Buffer now tracks `cmdp_cost_keys` and sizes value tensors as `1 + cost_count` to store reward plus each cost head.
- Added `self.cmdp_costs` array and ensured `insert()` writes cost samples when provided.
- Generalised return/advantage computation to iterate over reward + each cost stream, sharing GAE/discount logic while optionally using PopArt on the reward head only.
- Advantage normalisation helpers respect `cmdp_adv_norm_per_stream` and mask out inactive samples.

### 3. Driver Pipeline (`openrl_cmdp/drivers/rl_driver.py`, `openrl_cmdp/drivers/onpolicy_driver.py`)
- Drivers cache the configured `cmdp_cost_keys` and, during rollouts, extract matching entries from `info` dicts, assembling a `[env, agent, cost]` tensor that is handed to the replay buffer.
- `compute_returns()` requests multi-head value predictions and passes them to the updated buffer logic.

### 4. Network Heads (`openrl_cmdp/modules/networks/value_network.py`, `openrl_cmdp/modules/networks/policy_value_network.py`)
- Critics (standalone and shared) now instantiate:
  - Primary reward head (PopArt aware).
  - One linear head per cost stream.
- Forward passes concatenate reward and cost estimates so downstream losses receive `[..., 1 + cost_count]`.

### 5. PPO Algorithm (`openrl_cmdp/algorithms/ppo.py`)
- Initialises CMDP metadata (budgets, λ inits, tanh parametrisation state, EMA buffers).
- Computes per-stream advantages, normalises according to settings, and forms the objective advantage `A = A_reward - Σ λ_i · A_cost_i`.
- Applies value losses to every head (reward + each cost).
- Implements dual updates (`cmdp_update_duals`), with optional tanh reparameterisation (`lambda = lambda_max · (1 + tanh ν)/2`) to keep λ ∈ [0, λ_max].
- Logs cost/budget/λ/violation metrics under `cmdp/*`.

### 6. Auxiliary Support
- `openrl_cmdp/modules/ppo_module.py`, `openrl_cmdp/modules/common/ppo_net.py` already accept the multi-head critic outputs without further changes; no action required beyond consuming the larger value tensors.
- Added a targeted regression test (`tests/test_cmdp/cmdp_mappo_demo.py`) that spins up a toy two-agent CMDP environment, verifying cost logging, multi-head critic behaviour, and dual ascent dynamics.

### 7. Expected Environment Contract
- Environments supplying CMDP data must emit scalar or per-agent tensors in `info[cost_key]`; the driver handles both dict-per-agent and broadcast scalar formats.
- Reward remains unshaped—constraints are handled purely through the CMDP loss and dual updates.

These changes collectively enable OpenRL's MAPPO stack to train under CMDP objectives with multiple cost streams, while keeping the integration configurable through standard OpenRL arguments.

---

## Installation

Install from this git repository:

```bash
pip install git+https://github.com/chkxw/openrl_cmdp.git
```

Or for local development:

```bash
git clone https://github.com/chkxw/openrl_cmdp.git
cd openrl_cmdp
pip install -e .
```

## Usage

Import the package as `openrl_cmdp`:

```python
import openrl_cmdp
from openrl_cmdp.envs.common import make
from openrl_cmdp.modules.common import PPONet as Net
from openrl_cmdp.runners.common import PPOAgent as Agent
```

For CMDP-specific configuration, refer to the configuration options added in `openrl_cmdp/configs/config.py`.

---

## License

This project inherits the Apache 2.0 license from the original OpenRL framework.

## Acknowledgments

This work builds upon [OpenRL](https://github.com/OpenRL-Lab/openrl):

```bibtex
@article{huang2023openrl,
  title={OpenRL: A Unified Reinforcement Learning Framework},
  author={Huang, Shiyu and Chen, Wentse and Sun, Yiwen and Bie, Fuqing and Tu, Wei-Wei},
  journal={arXiv preprint arXiv:2312.16189},
  year={2023}
}
```

For the complete list of OpenRL's acknowledgments and original features, please visit the [original repository](https://github.com/OpenRL-Lab/openrl).
