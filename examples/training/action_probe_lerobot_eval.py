#!/usr/bin/env python3
import os
import sys
import json
import time
import atexit
import numpy as np

import gymnasium as gym
import gym_pusht  # registers envs

OUT = os.environ.get("ACTION_LOG_JSON", "action_probe.json")


class LogActionWrapper(gym.Wrapper):
    """
    Logs the action that reaches env.step().

    Works for:
      - action shape (act_dim,)
      - action shape (n_envs, act_dim)  [vectorized]
    Tracks min/max over time per action dimension (last axis).
    """
    def __init__(self, env):
        super().__init__(env)
        self.amin = None
        self.amax = None
        self.n = 0
        self.samples = []
        self.first_shape = None

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)

        if self.first_shape is None:
            self.first_shape = tuple(a.shape)

        # Compute per-dim min/max over any batch dims (keep last axis)
        if a.ndim == 1:
            cur_min = a
            cur_max = a
        else:
            batch_axes = tuple(range(a.ndim - 1))
            cur_min = np.min(a, axis=batch_axes)
            cur_max = np.max(a, axis=batch_axes)

        # Track only if action has at least 2 dims
        if cur_min.size >= 2:
            if self.amin is None:
                self.amin = cur_min.copy()
                self.amax = cur_max.copy()
            else:
                self.amin = np.minimum(self.amin, cur_min)
                self.amax = np.maximum(self.amax, cur_max)

            if len(self.samples) < 30:
                # store first env if batched; else store the vector
                if a.ndim == 1:
                    self.samples.append(a[:2].tolist())
                else:
                    self.samples.append(a.reshape(-1, a.shape[-1])[0, :2].tolist())

        self.n += 1
        return self.env.step(action)

    def close(self):
        dump(self)
        return super().close()


_WRAPPED = []


def dump(w=None):
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_steps_seen": int(w.n) if w is not None else None,
        "first_action_shape": list(w.first_shape) if (w is not None and w.first_shape is not None) else None,
        "action_min": w.amin.tolist() if (w is not None and w.amin is not None) else None,
        "action_max": w.amax.tolist() if (w is not None and w.amax is not None) else None,
        "action_samples_first30_xy": w.samples if w is not None else None,
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[action_probe] wrote {OUT}")
    print(json.dumps(payload, indent=2))


@atexit.register
def _dump_on_exit():
    if _WRAPPED:
        dump(_WRAPPED[-1])


def install_patch():
    # Patch gymnasium.make
    real_make = gym.make

    def patched_make(id, *args, **kwargs):
        env = real_make(id, *args, **kwargs)
        if "pusht" in str(id).lower():
            w = LogActionWrapper(env)
            _WRAPPED.append(w)
            return w
        return env

    gym.make = patched_make

    # Also patch legacy gym if present
    try:
        import gym as legacy_gym  # type: ignore
        legacy_real_make = legacy_gym.make

        def legacy_patched_make(id, *args, **kwargs):
            env = legacy_real_make(id, *args, **kwargs)
            if "pusht" in str(id).lower():
                w = LogActionWrapper(env)
                _WRAPPED.append(w)
                return w
            return env

        legacy_gym.make = legacy_patched_make
    except Exception:
        pass


def main():
    install_patch()

    # Pass-through: python action_probe_lerobot_eval.py -- <lerobot-eval args...>
    if "--" in sys.argv:
        i = sys.argv.index("--")
        eval_args = sys.argv[i + 1 :]
    else:
        eval_args = sys.argv[1 :]

    # Run lerobot-eval entrypoint in-process so our patch applies
    from lerobot.scripts.lerobot_eval import main as lerobot_eval_main  # type: ignore
    sys.argv = ["lerobot-eval"] + eval_args

    try:
        lerobot_eval_main()
    finally:
        if _WRAPPED:
            dump(_WRAPPED[-1])


if __name__ == "__main__":
    main()
