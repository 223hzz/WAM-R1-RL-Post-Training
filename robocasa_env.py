"""RoboCasa-GR1 environment adapter for the dedicated online-RL entry point."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


class RobocasaEnvWrapper:
    ACTION_KEYS = (
        ("action.left_arm", 7),
        ("action.right_arm", 7),
        ("action.left_hand", 6),
        ("action.right_hand", 6),
        ("action.waist", 3),
    )
    STATE_KEYS = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")

    def __init__(self, env_names: str, seed: int = 42, max_episode_steps: int = 720,
                 render_gpu_device_id: int = 0) -> None:
        import gymnasium as gym
        import robocasa  # noqa: F401
        import robosuite  # noqa: F401
        import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
        from examples.Robocasa_tabletop.eval_files.wrappers.multistep_wrapper import MultiStepWrapper

        self.env_names = [name.strip() for name in str(env_names).split(",") if name.strip()]
        if not self.env_names:
            raise ValueError("No RoboCasa tasks were provided")
        missing = [name for name in self.env_names if name not in gym.registry]
        if missing:
            raise ValueError(f"Unregistered RoboCasa tasks: {missing}")
        self.task_ids = list(range(len(self.env_names)))
        self.task_id = 0
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self._gym = gym
        self._MultiStepWrapper = MultiStepWrapper
        self.env = None
        self._episode = 0
        self._switch_task(0)

    def _switch_task(self, task_id: int) -> None:
        task_id = int(task_id)
        if self.env is not None:
            self.env.close()
        self.task_id = task_id
        base = self._gym.make(self.env_names[task_id], enable_render=True)
        self.env = self._MultiStepWrapper(
            base,
            video_delta_indices=np.array([0]),
            state_delta_indices=np.array([0]),
            n_action_steps=1,
            max_episode_steps=self.max_episode_steps,
        )

    @staticmethod
    def _first(value: Any) -> np.ndarray:
        array = np.asarray(value)
        while array.ndim > 1 and array.shape[0] == 1:
            array = array[0]
        return array

    def _convert_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        image_key = "video.ego_view"
        if image_key not in observation:
            image_key = "video.ego_view_bg_crop_pad_res256_freq20"
        rgb = self._first(observation[image_key]).astype(np.uint8)
        parts = []
        for key in self.STATE_KEYS:
            values = self._first(observation[f"state.{key}"]).reshape(-1).astype(np.float32)
            parts.append(np.concatenate((np.sin(values), np.cos(values))))
        state = np.concatenate(parts).astype(np.float32)
        if state.size < 64:
            state = np.pad(state, (0, 64 - state.size))
        else:
            state = state[:64]
        language = observation.get("annotation.human.coarse_action", "")
        if isinstance(language, (list, tuple, np.ndarray)):
            language = language[0] if len(language) else ""
        return {"rgb": rgb, "state": state, "lang": str(language)}

    def reset_for_task(self, task_id: int, episode_index: int = 0) -> Dict[str, Any]:
        if not 0 <= int(task_id) < len(self.env_names):
            raise ValueError(f"task_id={task_id} outside [0, {len(self.env_names)})")
        self._switch_task(int(task_id))
        self._episode = int(episode_index)
        observation, _ = self.env.reset(seed=self.seed + self._episode)
        return self._convert_observation(observation)

    def reset(self) -> Dict[str, Any]:
        return self.reset_for_task(self.task_id, self._episode + 1)

    @property
    def current_episode_key(self) -> Tuple[int, int]:
        return self.task_id, self._episode

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict]:
        flat = np.asarray(action, dtype=np.float32).reshape(-1)
        if flat.size < 29:
            raise ValueError(f"RoboCasa-GR1 expects 29 action values, got {flat.size}")
        cursor = 0
        action_dict = {}
        for key, width in self.ACTION_KEYS:
            action_dict[key] = flat[cursor:cursor + width][None, :]
            cursor += width
        observation, reward, terminated, truncated, info = self.env.step(action_dict)
        info = dict(info or {})
        success_value = info.get("success", False)
        success = bool(np.asarray(success_value).reshape(-1)[0])
        done = bool(terminated or truncated)
        return self._convert_observation(observation), float(np.asarray(reward).max()), done, {
            **info, "success": success, "task_id": self.task_id,
            "init_state_id": self._episode,
        }

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
