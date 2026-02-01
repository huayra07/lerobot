#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import concurrent.futures as cf
import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)
from examples.training.train_pusht_lang_diffusion_STEP_C_FIXED3 import (
    CLIPLanguageEncoder, HybridCLIPDiffusionPolicy
)
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy



def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done) and step < max_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)
        

        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        observation = env_preprocessor(observation)

        observation = preprocessor(observation)
        try:
            print("[DEBUG] env.action_space:", env.action_space)
            print("[DEBUG] env.unwrapped.action_space:", env.unwrapped.action_space)
        except Exception as e:
            print("[DEBUG] could not print action_space:", e)

        def pstats(tag, a):
            if torch.is_tensor(a):
                print(f"[{tag}] shape={tuple(a.shape)} min/max={float(a.min()):.3f}/{float(a.max()):.3f} device={a.device} dtype={a.dtype}")
            else:
                # import numpy as np
                a = np.asarray(a)
                print(f"[{tag}] shape={a.shape} min/max={a.min():.3f}/{a.max():.3f} type={type(a)}")
        # with torch.inference_mode():
        #     action = policy.select_action(observation)
        # pstats("RAW policy.select_action", action)
        # if torch.is_tensor(action):
        #     sat = (action.abs() > 0.98).float().mean().item()
        #     print(f"[DEBUG] fraction |action|>0.98: {sat:.3f}")



        # ---- UNNORMALIZE / POSTPROCESS ACTIONS BEFORE env.step ----
        # if hasattr(policy, "postprocess_action"):
        #     action = policy.postprocess_action(action)
        #     pstats("AFTER policy.postprocess_action", action)

        # elif hasattr(policy, "action_postprocessor") and policy.action_postprocessor is not None:
        #     # common in LeRobot: a Processor object
        #     action = policy.action_postprocessor(action)
        #     pstats("AFTER policy.action_postprocessor", action)

        # elif hasattr(policy, "postprocessors") and isinstance(policy.postprocessors, dict):
        #     # sometimes stored as dict or list
        #     pp = policy.postprocessors.get("action", None) or policy.postprocessors.get("actions", None)
        #     if pp is not None:
        #         action = pp(action)

        # (optional safety) ensure cpu + numpy if env expects numpy
        # if torch.is_tensor(action):
        #     action_numpy = action.detach().cpu().numpy()
        # else:
        #     action_numpy = action

        # -----------------------------------------------------------
        # action = postprocessor(action)

        # action_transition = {ACTION: action}
        # print("ACTION constant =", ACTION)
        # print("env_postprocessor type =", type(env_postprocessor))
        # print("env_postprocessor =", env_postprocessor)
        # action_transition = {ACTION: action}


        # --- 1) Torch path: what keys + ranges come out? ---


        # if hasattr(policy, "postprocess_action"):
        #     try:
        #         action = policy.postprocess_action(action)
        #         pstats("AFTER policy.postprocess_action", action)
        #     except Exception as e:
        #         print("policy.postprocess_action failed:", e)

        # if hasattr(policy, "action_postprocessor") and policy.action_postprocessor is not None:
        #     try:
        #         action = policy.action_postprocessor(action)
        #         pstats("AFTER policy.action_postprocessor", action)
        #     except Exception as e:
        #         print("policy.action_postprocessor failed:", e)
        # out = env_postprocessor({ACTION: action})
        # print("postprocessor out keys:", out.keys())
        # for k, v in out.items():
        #     if torch.is_tensor(v):
        #         print(" key", k, "min/max", float(v.min()), float(v.max()), "device", v.device, "dtype", v.dtype)
        #     else:
        #         # import numpy as np
        #         vv = np.asarray(v)
        #         print(" key", k, "min/max", vv.min(), vv.max(), "type", type(v))

        # # --- 2) Key mismatch brute force (still torch) ---
        # for key in ["action", "actions", "ACTION", ACTION]:
        #     try:
        #         outk = env_postprocessor({key: action})
        #         v = outk.get(key, None)
        #         print("key", repr(key), "returned same key?", v is not None)
        #     except Exception as e:
        #         print("key", repr(key), "error:", e)

        # # --- 3) Numpy/CPU path: does it only work on numpy? ---
        # # import numpy as np
        # # action = out[ACTION]
        # # action_np = action.detach().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
        # # out_np = env_postprocessor({ACTION: action_np})
        # # v_np = np.asarray(out_np.get(ACTION))
        # # print("[numpy test] out min/max", v_np.min(), v_np.max(), "dtype", v_np.dtype)
        # # out = env_postprocessor({ACTION: action})  # action is torch.Tensor

        # # print("ACTION constant =", ACTION, "type=", type(ACTION))

        # print("postprocessor out keys:", out.keys())
        # for k,v in out.items():
        #     if torch.is_tensor(v):
        #         print(" key", k, "shape", tuple(v.shape), "min/max", float(v.min()), float(v.max()), "device", v.device)
        #     else:
        #         # import numpy as np
        #         vv = np.asarray(v)
        #         print(" key", k, "min/max", vv.min(), vv.max(), "type", type(v))
        # # dump pipeline steps (common attribute names)
        # for attr in ["steps", "_steps", "processor_steps", "_processor_steps"]:
        #     if hasattr(env_postprocessor, attr):
        #         steps = getattr(env_postprocessor, attr)
        #         try:
        #             print("env_postprocessor.", attr, "len =", len(steps))
        #             print([type(s).__name__ for s in steps])
        #         except Exception as e:
        #             print("couldn't print steps via", attr, "error:", e)

        # print("has policy.action_postprocessor?", hasattr(policy, "action_postprocessor"), type(getattr(policy, "action_postprocessor", None)))
        # print("has policy.postprocess_action?", hasattr(policy, "postprocess_action"))
        # print("has policy.postprocessors?", hasattr(policy, "postprocessors"), type(getattr(policy, "postprocessors", None)))
        # print("policy attrs containing 'post':", [x for x in dir(policy) if "post" in x.lower()])
        # print("policy attrs containing 'processor':", [x for x in dir(policy) if "processor" in x.lower()])


        # print("env_postprocessor dir has:", [x for x in dir(env_postprocessor) if "step" in x or "processor" in x])

        # action = out[ACTION]
        # pstats("AFTER env_postprocessor", action)

        # # action_np = action.detach().cpu().numpy() # only after postprocessing


        # # --- pick ONE output to actually step with (use what works) ---
        # # If numpy test changes range (e.g., to 0..512), use that.
        # # Otherwise use torch output (likely unchanged).
        # # if v_np is not None and (v_np.min() < -1.01 or v_np.max() > 1.01):
        # #     action_to_env = v_np
        # # else:
        # #     action_out = out.get(ACTION, action)
        # #     action_to_env = action_out.detach().cpu().numpy() if torch.is_tensor(action_out) else np.asarray(action_out)
        # # action_np = action_to_env.detach().cpu().numpy() if torch.is_tensor(action_to_env) else np.asarray(action_to_env)
        # pstats("AFTER env_postprocessor", action)

        # # --- if still in [-1,1], do MANUAL map using env.action_space as a debug fallback ---
        # action_to_env = action
        # if torch.is_tensor(action_to_env):
        #     a_min = float(action_to_env.min())
        #     a_max = float(action_to_env.max())
        # else:
        #     a_np = np.asarray(action_to_env)
        #     a_min = float(a_np.min())
        #     a_max = float(a_np.max())

        # if a_min >= -1.01 and a_max <= 1.01 and hasattr(env, "action_space"):
        #     asp = env.action_space
        #     if hasattr(asp, "low") and hasattr(asp, "high"):
        #         low = torch.as_tensor(asp.low, device=action.device, dtype=action.dtype)
        #         high = torch.as_tensor(asp.high, device=action.device, dtype=action.dtype)
        #         # broadcast low/high to (batch, action_dim)
        #         action_to_env = low + (action + 1.0) * 0.5 * (high - low)
        #         pstats("AFTER manual [-1,1]->[low,high]", action_to_env)

        # # finally step with numpy
        # action_np = action_to_env.detach().cpu().numpy() if torch.is_tensor(action_to_env) else np.asarray(action_to_env)
        # # print("[EVAL] action to env:", action_np.shape, action_np.min(), action_np.max())
        # if torch.is_tensor(action_to_env):
        #     print("[EVAL] action_to_env torch min/max:", float(action_to_env.min()), float(action_to_env.max()), "device", action_to_env.device)


        # # print("[EVAL] action to env:", action_to_env.shape, action_to_env.min(), action_to_env.max())
        # # observation, reward, terminated, truncated, info = env.step(action_to_env)
        # print("sample action[0]:", action_np[0])

        # action_np = action_to_env.detach().cpu().numpy() if torch.is_tensor(action_to_env) else np.asarray(action_to_env)


        # with torch.inference_mode():
        #     action = policy.select_action(observation)  # torch tensor, likely on cuda, in [-1, 1]
        # if isinstance(observation, dict):
        #     print("[DEBUG] observation keys:", sorted(observation.keys())[:50])
        #     # also print any key containing 'image' or 'rgb'
        #     print("[DEBUG] image-like keys:", [k for k in observation.keys() if "image" in k or "rgb" in k or "pixel" in k])
        # action_pp = action
        # did_postprocess = False

        # if hasattr(policy, "postprocess_action"):
        #     try:
        #         action_pp = policy.postprocess_action(action_pp)
        #         did_postprocess = True
        #         pstats("AFTER policy.postprocess_action", action_pp)
        #     except Exception as e:
        #         print("[DEBUG] policy.postprocess_action failed:", e)

        # if (not did_postprocess) and hasattr(policy, "action_postprocessor") and (policy.action_postprocessor is not None):
        #     try:
        #         action_pp = policy.action_postprocessor(action_pp)
        #         did_postprocess = True
        #         pstats("AFTER policy.action_postprocessor", action_pp)
        #     except Exception as e:
        #         print("[DEBUG] policy.action_postprocessor failed:", e)

        # if (not did_postprocess) and hasattr(policy, "postprocessors") and isinstance(policy.postprocessors, dict):
        #     # Common naming variations
        #     for key in ("action", "actions"):
        #         pp = policy.postprocessors.get(key, None)
        #         if pp is not None:
        #             try:
        #                 action_pp = pp(action_pp)
        #                 did_postprocess = True
        #                 pstats(f"AFTER policy.postprocessors['{key}']", action_pp)
        #                 break
        #             except Exception as e:
        #                 print(f"[DEBUG] policy.postprocessors['{key}'] failed:", e)





                
        # if "env_postprocessor" in locals() and env_postprocessor is not None:
        #     try:
        #         # Try the canonical key first
        #         out = env_postprocessor({ACTION: action_pp})
        #         if isinstance(out, dict) and (ACTION in out):
        #             action_pp = out[ACTION]
        #             pstats("AFTER env_postprocessor", action_pp)
        #         else:
        #             # If it returns something unexpected, log it rather than silently using it
        #             print("[DEBUG] env_postprocessor returned keys:", list(out.keys()) if isinstance(out, dict) else type(out))
        #     except Exception as e:
        #         print("[DEBUG] env_postprocessor failed:", e)

        # action_to_env = action_pp

        # def looks_normalized(a):
        #     if torch.is_tensor(a):
        #         a_min, a_max = float(a.min()), float(a.max())
        #     else:
        #         aa = np.asarray(a)
        #         a_min, a_max = float(aa.min()), float(aa.max())
        #     return (a_min >= -1.01) and (a_max <= 1.01)

        # if looks_normalized(action_to_env):
        #     print("[WARN] Action still looks normalized in [-1,1]. Applying MANUAL scaling fallback.")
        #     asp = env.action_space  # may be wrapped; that's why we printed unwrapped too
        #     if hasattr(asp, "low") and hasattr(asp, "high"):
        #         if not torch.is_tensor(action_to_env):
        #             action_to_env = torch.as_tensor(action_to_env)
        #         action_to_env = action_to_env.to(device=action.device, dtype=action.dtype)

        #         low = torch.as_tensor(asp.low, device=action_to_env.device, dtype=action_to_env.dtype)
        #         high = torch.as_tensor(asp.high, device=action_to_env.device, dtype=action_to_env.dtype)

        #         # Broadcast to batch if needed
        #         if low.ndim == 1 and action_to_env.ndim == 2:
        #             low = low.unsqueeze(0)
        #             high = high.unsqueeze(0)

        #         action_to_env = low + (action_to_env + 1.0) * 0.5 * (high - low)
        #         pstats("AFTER manual [-1,1]->[low,high]", action_to_env)
        #     else:
        #         print("[WARN] env.action_space has no low/high; cannot manual-scale.")

        # # Optional: keep ONE print for sanity
        # # pstats("RAW policy.select_action", action)

        # # # Map [-1,1] -> [low,high] using env.action_space (PushT usually wants pixel coords)
        # # asp = env.action_space
        # # low = torch.as_tensor(asp.low, device=action.device, dtype=action.dtype)
        # # high = torch.as_tensor(asp.high, device=action.device, dtype=action.dtype)

        # # action_to_env = low + (action + 1.0) * 0.5 * (high - low)

        # # # Optional: keep ONE print for sanity
        # # pstats("action_to_env", action_to_env)
        # # print("sample action[0]:", action_to_env[0].detach().cpu().numpy())

        # # IMPORTANT: step with numpy on CPU (prevents render crash)
        # if torch.is_tensor(action_to_env):
        #     action_np = action_to_env.detach().cpu().numpy()
        # else:
        #     action_np = np.asarray(action_to_env)
        # print("[STEP] RAW min/max:", float(action_pp.min()), float(action_pp.max()))
        # print("[STEP] ENV min/max:", float(action_np.min()), float(action_np.max()))
        # print("[STEP] sending:", action_np[0])
        # observation, reward, terminated, truncated, info = env.step(action_np)



        # action_to_env = low + (action + 1.0) * 0.5 * (high-low) # torch on cuda, [0,512]
        # action_np = action_to_env.detach().cpu().numpy()        # numpy on cpu

        # observation, reward, terminated, truncated, info = env.step(action_np)

        # observation, reward, terminated, truncated, info = env.step(action_np)


        # action_transition = env_postprocessor(action_transition)
        # action = action_transition[ACTION]
        # pstats("AFTER env_postprocessor", action)




        # # Convert to CPU / numpy.
        # action_numpy: np.ndarray = action.to("cpu").numpy()
        # assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # # Apply the next action.
        # a = action
        # if torch.is_tensor(a):
        #     print("[EVAL] action to env:",
        #         "shape", tuple(a.shape),
        #         "min/max", float(a.min()), float(a.max()),
        #         "dtype", a.dtype, "device", a.device)
        # if hasattr(env, "action_space"):
        #     asp = env.action_space
        #     if hasattr(asp, "low") and hasattr(asp, "high"):
        #         low, high = asp.low, asp.high  # numpy arrays
        #         # map [-1, 1] -> [low, high]
        #         action_numpy = low + (action_numpy + 1.0) * 0.5 * (high - low)
        # print("[EVAL] action to env:", action_numpy.shape, action_numpy.min(), action_numpy.max())



        # action_numpy = action.detach().cpu().numpy() if torch.is_tensor(action) else action
        # print("[EVAL] action to env:", action_numpy.shape, action_numpy.min(), action_numpy.max())

        # observation, reward, terminated, truncated, info = env.step(action_numpy)
        print("[CHECK] has language_embedding:", isinstance(observation, dict) and "language_embedding" in observation)
        if isinstance(observation, dict) and "language_embedding" in observation:
            # x = observation["language_embedding"]
            # print("[CHECK] lang shape:", getattr(x, "shape", None), "min/max:", float(x.min()), float(x.max()))
            observation["language_embedding"] = torch.zeros_like(observation["language_embedding"])


        with torch.inference_mode():
            action = policy.select_action(observation)
        
        # # TEMP TEST: remove injected language embedding and see if policy can still run
        # if isinstance(observation, dict) and "language_embedding" in observation:
        #     observation = dict(observation)
        #     observation.pop("language_embedding")
        #     print("[TEST] removed language_embedding from observation")
        # #temp
        raw = action.detach().cpu()
        sat = ((raw.abs() > 0.98).float().mean()).item()
        print("raw saturation frac:", sat)
        #until here


        
        pstats("RAW policy.select_action", action)

        # 1) ALWAYS apply the checkpoint postprocessor (unnormalizer etc.)
        action = postprocessor(action)
        pstats("AFTER postprocessor(action)", action)

        # 2) Apply env-specific postprocessing (expects a transition dict)
        transition = {ACTION: action}
        transition = env_postprocessor(transition)
        action = transition[ACTION]
        pstats("AFTER env_postprocessor", action)

        # 3) Step the env with numpy on CPU
        # action_np = action.detach().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
        # assert action_np.ndim == 2 and action_np.shape[1] == 2, f"bad action shape {action_np.shape}"

        # print("[STEP] sending:", action_np[0])
        action_np = action.detach().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)

        print("action_np shape:", action_np.shape, "dtype:", action_np.dtype,
            "min/max:", float(action_np.min()), float(action_np.max()))
        print("env.action_space:", env.action_space)

        assert action_np.shape == env.action_space.shape, f"{action_np.shape} vs {env.action_space.shape}"
        assert env.action_space.contains(action_np), "env.action_space.contains failed"

        observation, reward, terminated, truncated, info = env.step(action_np)

        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_np)) # used to be action_numpy
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = (
            einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
        )
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    # if not isinstance(policy, PreTrainedPolicy):
    #     exc = ValueError(
    #         f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
    #     )
    #     try:
    #         from peft import PeftModel

    #         if not isinstance(policy, PeftModel):
    #             raise exc
    #     except ImportError:
    #         raise exc from None
    if not isinstance(policy, PreTrainedPolicy):
        exc = ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )

        # Allow wrapper policies that are plain torch.nn.Module (e.g., HybridCLIPDiffusionPolicy)
        if isinstance(policy, torch.nn.Module):
            pass
        else:
            try:
                from peft import PeftModel
                if not isinstance(policy, PeftModel):
                    raise exc
            except ImportError:
                raise exc from None

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            ep_frames.append(np.stack([env.envs[i].render() for i in range(n_to_render_now)]))  # noqa: B023
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            # Here we must render all frames and discard any we don't need.
            ep_frames.append(np.stack(env.call("render")[:n_to_render_now]))

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        # trust_remote_code=cfg.trust_remote_code,
        trust_remote_code=getattr(cfg, "trust_remote_code", False),

    )

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )
    print("\n[DEBUG] policy class:", type(policy))
    # sd = torch.load(Path(cfg.policy.pretrained_path) / "model.safetensors", map_location="cpu")
    # print("[DEBUG] loaded safetensors keys:", len(sd.keys()))

    device = torch.device(cfg.policy.device)  # robust
    policy = policy.to(device)
    # p = Path(cfg.policy.pretrained_path)  # whatever variable you already have
    # device = torch.device(cfg.policy.device)

    # # Step-C hybrid checkpoint typically has base_policy/ and clip_info.json
    # is_hybrid = (p / "base_policy" / "config.json").exists() or (p / "clip_info.json").exists()

    # # Wrap Step-C only when it expects CLIP embeddings injected
    # # if getattr(cfg.policy, "use_language_cond", False) and getattr(cfg.policy, "language_embedding_source", "") == "clip":
    # #     clip_encoder = CLIPLanguageEncoder(getattr(cfg.policy, "text_encoder_name", "openai/clip-vit-base-patch32")).to(device)
    # #     policy = HybridCLIPDiffusionPolicy(policy, clip_encoder).to(device)
    # #     policy.eval()
    # if is_hybrid:
    #     # IMPORTANT: point to the HYBRID ROOT (the folder that contains base_policy/)
    #     hybrid_root = p if (p / "base_policy").exists() else p.parent
    #     policy = HybridCLIPDiffusionPolicy.from_pretrained(str(hybrid_root), device=device)
    # else:
    #     policy = DiffusionPolicy.from_pretrained(str(p)).to(device)
    p = Path(cfg.policy.pretrained_path)
    device = torch.device(cfg.policy.device)

    # 1) Load the model from the migrated folder (so processors exist)
    policy = DiffusionPolicy.from_pretrained(str(p)).to(device)
    policy.eval()

    # 2) If this is Step-C (language_cond + clip source), wrap it so we inject embeddings
    if getattr(cfg.policy, "use_language_cond", False) and getattr(cfg.policy, "language_embedding_source", "") == "clip":
        clip_name = getattr(cfg.policy, "text_encoder_name", "openai/clip-vit-base-patch32")
        clip_encoder = CLIPLanguageEncoder(clip_name).to(device)
        policy = HybridCLIPDiffusionPolicy(policy, clip_encoder).to(device)
        policy.eval()
        print("[EVAL] Wrapped policy with HybridCLIPDiffusionPolicy (CLIP injection enabled).")
    else:
        print("[EVAL] Using plain DiffusionPolicy (no hybrid wrapper).")
    print("[DEBUG] has policy.postprocess_action:", hasattr(policy, "postprocess_action"))
    print("[DEBUG] has policy.action_postprocessor:", hasattr(policy, "action_postprocessor"),
        "value type:", type(getattr(policy, "action_postprocessor", None)))
    print("[DEBUG] has policy.postprocessors:", hasattr(policy, "postprocessors"),
        "value type:", type(getattr(policy, "postprocessors", None)))
    print("[DEBUG] policy attrs containing 'post':", [x for x in dir(policy) if "post" in x.lower()])



    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    )


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    metrics = eval_one(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
    )

    if max_parallel_tasks <= 1:
        # sequential path (single accumulator path on the main thread)
        # NOTE: keeping a single-threaded accumulator avoids concurrent list appends or locks
        for task_group, task_id, env in tasks:
            tg, tid, metrics = task_runner(task_group, task_id, env)
            _accumulate_to(tg, metrics)
            per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
    else:
        # threaded path: submit all tasks, consume completions on main thread and accumulate there
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id)
            for fut in cf.as_completed(fut2meta):
                tg, tid, metrics = fut.result()
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()