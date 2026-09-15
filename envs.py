# -*- coding: utf-8 -*-
"""Environment adapters used by the online Action-GRPO trainer."""

from __future__ import annotations

import gc
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


def _parse_task_ids(spec: Optional[str], default_task_id: int, num_tasks: int) -> List[int]:
    if spec is None or not str(spec).strip():
        ids = [int(default_task_id)]
    elif str(spec).strip().lower() == "all":
        ids = list(range(num_tasks))
    else:
        ids = []
        for token in str(spec).split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start, end = (int(value) for value in token.split("-", 1))
                ids.extend(range(start, end + 1))
            else:
                ids.append(int(token))
        ids = list(dict.fromkeys(ids))
    if not ids:
        raise ValueError("At least one LIBERO task id is required")
    invalid = [task_id for task_id in ids if not 0 <= task_id < num_tasks]
    if invalid:
        raise ValueError(f"LIBERO task ids outside [0, {num_tasks}): {invalid}")
    return ids


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert LIBERO/robosuite xyzw quaternion to an axis-angle vector."""
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - float(quat[3]) ** 2))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(float(quat[3]))) / denominator).astype(
        np.float32
    )


class LiberoEnvWrapper:
    """Single-task LIBERO adapter with deterministic GRPO group resets."""

    def __init__(
        self,
        benchmark_name: str = "libero_10",
        task_id: int = 0,
        task_ids: Optional[str] = None,
        task_schedule: str = "round_robin",
        task_block_size: int = 1,
        resolution: int = 224,
        seed: int = 42,
        init_state_id: Optional[int] = None,
        num_steps_wait: int = 10,
        camera_name: str = "agentview_image",
        wrist_camera_name: Optional[str] = "robot0_eye_in_hand_image",
        env_action_dim: int = 7,
        render_gpu_device_id: int = -1,
    ) -> None:
        try:
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as exc:
            raise ImportError(
                "LIBERO is not importable. Add <project>/LIBERO to PYTHONPATH "
                "or install the repository's LIBERO package."
            ) from exc

        benchmark_dict = benchmark.get_benchmark_dict()
        if benchmark_name not in benchmark_dict:
            choices = ", ".join(sorted(benchmark_dict))
            raise ValueError(
                f"Unknown LIBERO benchmark {benchmark_name!r}. Available: {choices}"
            )
        self.task_suite = benchmark_dict[benchmark_name]()
        self.task_ids = _parse_task_ids(
            task_ids, task_id, self.task_suite.get_num_tasks()
        )
        if task_schedule not in {"round_robin", "random"}:
            raise ValueError("task_schedule must be 'round_robin' or 'random'")
        self.task_schedule = task_schedule
        self.task_block_size = max(1, int(task_block_size))
        self.benchmark_name = benchmark_name
        self.seed = seed
        self.init_state_id = init_state_id
        self.num_steps_wait = max(0, int(num_steps_wait))
        self.camera_name = camera_name
        self.wrist_camera_name = wrist_camera_name
        self.env_action_dim = int(env_action_dim)
        self.render_gpu_device_id = int(render_gpu_device_id)
        self.resolution = int(resolution)
        self._OffScreenRenderEnv = OffScreenRenderEnv
        self._bddl_root = Path(get_libero_path("bddl_files"))
        self._rng = np.random.default_rng(seed)
        self._training_group_index = 0
        self._random_task_id = self.task_ids[0]
        self._episode_indices = {selected_id: 0 for selected_id in self.task_ids}
        self._current_init_state_index: Optional[int] = None
        self.task_id = -1
        self.task = None
        self.language = ""
        self.initial_states = []
        self.env = None
        self._switch_task(self.task_ids[0])

    def _switch_task(self, task_id: int) -> None:
        if task_id == self.task_id and self.env is not None:
            return
        self._dispose_env()
        self.task_id = int(task_id)
        self.task = self.task_suite.get_task(self.task_id)
        self.language = str(self.task.language)
        self.initial_states = self.task_suite.get_task_init_states(self.task_id)
        if len(self.initial_states) == 0:
            raise RuntimeError(f"LIBERO task {self.task_id} has no benchmark initial states")
        bddl_file = self._bddl_root / self.task.problem_folder / self.task.bddl_file
        new_env = None
        try:
            new_env = self._OffScreenRenderEnv(
                bddl_file_name=bddl_file,
                camera_heights=self.resolution,
                camera_widths=self.resolution,
                render_gpu_device_id=self.render_gpu_device_id,
            )
            new_env.seed(self.seed + self.task_id)
        except Exception:
            if new_env is not None:
                try:
                    new_env.close()
                except Exception:
                    logger.debug("Failed to close a partially initialized LIBERO env", exc_info=True)
            gc.collect()
            raise
        self.env = new_env
        self._current_init_state_index = None

    def _dispose_env(self) -> None:
        current_env, self.env = self.env, None
        if current_env is not None:
            try:
                current_env.close()
            except Exception:
                logger.warning("LIBERO environment cleanup failed", exc_info=True)
        gc.collect()

    def _select_training_task(self) -> int:
        block_index = self._training_group_index // self.task_block_size
        if self.task_schedule == "round_robin":
            selected = self.task_ids[block_index % len(self.task_ids)]
        else:
            if self._training_group_index % self.task_block_size == 0:
                self._random_task_id = int(self._rng.choice(self.task_ids))
            selected = self._random_task_id
        self._training_group_index += 1
        return selected

    def _select_init_state(self) -> int:
        if self.init_state_id is not None:
            return int(self.init_state_id) % len(self.initial_states)
        episode_index = self._episode_indices.setdefault(self.task_id, 0)
        index = episode_index % len(self.initial_states)
        self._episode_indices[self.task_id] = episode_index + 1
        return index

    def _restore_current_state(self) -> Dict[str, Any]:
        if self._current_init_state_index is None:
            raise RuntimeError("reset() must be called before reset_to_group_start()")
        self.env.reset()
        raw_obs = self.env.set_init_state(
            self.initial_states[self._current_init_state_index]
        )
        if self.num_steps_wait:
            dummy_action = np.zeros(self.env_action_dim, dtype=np.float32)
            for _ in range(self.num_steps_wait):
                raw_obs, _, done, _ = self.env.step(dummy_action.tolist())
                if done:
                    break
        return self._convert_observation(raw_obs)

    def reset(self) -> Dict[str, Any]:
        self._switch_task(self._select_training_task())
        self._current_init_state_index = self._select_init_state()
        return self._restore_current_state()

    def reset_for_task(
        self, task_id: int, episode_index: int = 0
    ) -> Dict[str, Any]:
        """Reset a specific task/init-state without advancing training schedule."""
        if task_id not in self.task_ids:
            raise ValueError(f"task_id={task_id} is not in configured task_ids={self.task_ids}")
        self._switch_task(task_id)
        selected_index = self.init_state_id if self.init_state_id is not None else episode_index
        self._current_init_state_index = int(selected_index) % len(self.initial_states)
        return self._restore_current_state()

    @property
    def current_episode_key(self) -> Tuple[int, int]:
        if self._current_init_state_index is None:
            raise RuntimeError("The environment has not been reset")
        return self.task_id, self._current_init_state_index

    def training_state_dict(self) -> Dict[str, Any]:
        return {
            "training_group_index": self._training_group_index,
            "episode_indices": dict(self._episode_indices),
            "random_task_id": self._random_task_id,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_training_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self._training_group_index = int(state.get("training_group_index", 0))
        self._episode_indices.update(
            {int(key): int(value) for key, value in state.get("episode_indices", {}).items()}
        )
        self._random_task_id = int(state.get("random_task_id", self.task_ids[0]))
        if "rng_state" in state:
            self._rng.bit_generator.state = state["rng_state"]

    def reset_to_group_start(self) -> Dict[str, Any]:
        """Restore the exact benchmark initial state used by the current group."""
        return self._restore_current_state()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < self.env_action_dim:
            raise ValueError(
                f"LIBERO needs {self.env_action_dim} action values, got {action.size}"
            )
        action = action[: self.env_action_dim]
        if not np.isfinite(action).all():
            raise ValueError("Action contains NaN or infinity")
        if self.env_action_dim >= 7:
            # Match examples/LIBERO/eval_files/eval_libero.py.
            action[6] = 1.0 - 2.0 * float(action[6] > 0.5)

        raw_obs, reward, done, info = self.env.step(action.tolist())
        success = bool(done)
        if hasattr(self.env, "check_success"):
            success = bool(self.env.check_success())
        info = dict(info or {})
        info["success"] = success
        info["init_state_id"] = self._current_init_state_index
        info["task_id"] = self.task_id
        return self._convert_observation(raw_obs), float(reward), bool(done), info

    def _convert_observation(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        if self.camera_name not in raw_obs:
            raise KeyError(
                f"Camera key {self.camera_name!r} missing from LIBERO observation"
            )
        # Matches examples/LIBERO/eval_files/eval_libero.py training alignment.
        rgb = np.ascontiguousarray(raw_obs[self.camera_name][::-1, ::-1])
        wrist_rgb = None
        if self.wrist_camera_name:
            if self.wrist_camera_name not in raw_obs:
                raise KeyError(
                    f"Wrist camera key {self.wrist_camera_name!r} missing from "
                    "LIBERO observation"
                )
            wrist_rgb = np.ascontiguousarray(
                raw_obs[self.wrist_camera_name][::-1, ::-1]
            )
        state = np.concatenate(
            (
                np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32),
                _quat2axisangle(raw_obs["robot0_eef_quat"]),
                np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32),
            )
        )
        return {
            "rgb": rgb,
            "wrist_rgb": wrist_rgb,
            "state": state.astype(np.float32, copy=False),
            "lang": self.language,
            "contact_points": None,
        }

    def close(self) -> None:
        self._dispose_env()

