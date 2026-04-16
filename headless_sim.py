"""Headless simulator wrapper for RL training and evaluation."""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

# Physical scales mapping normalized policy action [-1,1] to physical units.
# Must match sim_core.Robot.apply_normalized_action.
_PHYSICAL_SCALES: Dict[str, np.ndarray] = {
    "heading_drive":   np.array([40.0, 0.40], dtype=np.float32),
    "xy_strafe":       np.array([0.40, 0.40], dtype=np.float32),
    "heading_strafe":  np.array([40.0, 0.40, 0.40], dtype=np.float32),
}
# Number of action outputs per control mode.
_N_ACTION: Dict[str, int] = {
    "heading_drive":  2,
    "xy_strafe":      2,
    "heading_strafe": 3,
}

# IL training feature order for action history terms.
# The normalized action in heading_drive is [rotation_rate_norm, drive_speed_norm], but IL
# logs/features are [drive_speed_mps, rotation_rate_dps]. We reorder accordingly.
_ACTION_FEATURE_ORDER: Dict[str, np.ndarray] = {
    "heading_drive": np.array([1, 0], dtype=np.int64),
    "xy_strafe": np.array([0, 1], dtype=np.int64),
    "heading_strafe": np.array([0, 1, 2], dtype=np.int64),
}
# State features per timestep: 11 whiskers + 11 target-bbox whiskers + 1 heading.
_STATE_DIM = 23

from sim_core import (
    ROOM_HEIGHT,
    ROOM_WIDTH,
    TARGET_REACH_RADIUS,
    Robot,
    generate_obstacles,
    generate_target,
    generate_target_bbox_pose,
)


class HeadlessSimulator:
    """A fully headless simulation environment with fixed-step dynamics."""

    def __init__(
        self,
        seed: int,
        control_mode: str,
        history_window: int = 5,
        n_obstacles: int = 0,
        distance_reduction_scale: float = 1.0,
        collision_penalty: float = -0.1,
        collision_fail_penalty: float = -5.0,
        timestep_penalty: float = -0.01,
        target_reached_bonus: float = 10.0,
        timeout_penalty: float = -1.0,
        terminate_on_collision: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.dt = 0.05
        # Stuck watchdog: each window must improve distance-to-target by this amount.
        self.min_progress_m = 1.0
        self.progress_window_s = 60.0

        self.control_mode = control_mode
        self.history_window = int(history_window)
        # n_obstacles: exact count per episode. 0 = random 3-10.
        self.n_obstacles_count = int(n_obstacles)

        self.distance_reduction_scale = float(distance_reduction_scale)
        self.collision_penalty = float(collision_penalty)
        self.collision_fail_penalty = float(collision_fail_penalty)
        self.timestep_penalty = float(timestep_penalty)
        self.target_reached_bonus = float(target_reached_bonus)
        self.timeout_penalty = float(timeout_penalty)
        self.terminate_on_collision = bool(terminate_on_collision)

        self.robot = Robot()
        self.robot.set_control_mode(control_mode)

        self.obstacles: List[Tuple[float, float, float]] = []
        self.target: Tuple[float, float, float] = (ROOM_WIDTH * 0.5, ROOM_HEIGHT * 0.5, 0.075)
        self.target_bbox_pose: Dict[str, float] = {"rotation_rad": 0.0, "offset_x": 0.0, "offset_y": 0.0}
        self.whisker_segments = []

        self.episode_steps = 0
        self.simulated_time = 0.0
        self.collision_count = 0
        self.trajectory_history: List[Tuple[float, float]] = []
        self._window_start_time = 0.0
        self._window_start_distance = 0.0

        self._n_action = _N_ACTION.get(control_mode, 2)
        self._phys_scales = _PHYSICAL_SCALES.get(control_mode, np.ones(2, dtype=np.float32))
        self._action_feature_order = _ACTION_FEATURE_ORDER.get(control_mode, np.arange(self._n_action))
        # Per-step state history (raw whiskers + raw heading).
        self._state_history: deque = deque(maxlen=self.history_window)
        # Per-transition action history between consecutive states (physical units).
        # maxlen = history_window-1; for window=1 this is 0 (no inter-step actions).
        self._action_history: deque = deque(maxlen=max(1, self.history_window - 1))

    def _reset_episode_state(self) -> None:
        self.robot.collision_flag = False
        self.robot.target_contact_flag = False
        self.robot.collision_source = None
        self.whisker_segments = self.robot.compute_whiskers(
            self.obstacles, target=self.target, target_bbox_pose=self.target_bbox_pose
        )
        self.robot.update_heading_to_target(self.target)

        self.episode_steps = 0
        self.simulated_time = 0.0
        self.collision_count = 0
        self.trajectory_history = [(self.robot.x, self.robot.y)]
        self._window_start_time = 0.0
        self._window_start_distance = self._distance_to_target()

        # Prime history buffers with copies of the initial real observation
        # for state slots, and zero-pad the action slots (no command issued
        # yet). Avoids the 'whiskers=0 = obstacles everywhere' artifact that
        # literal zero-padding would produce in the first N-1 ticks.
        zero_action = np.zeros(self._n_action, dtype=np.float32)
        initial_state = self._encode_state()
        self._state_history.clear()
        self._action_history.clear()
        for _ in range(self.history_window - 1):
            self._state_history.append(initial_state.copy())
        for _ in range(self.history_window - 1):
            self._action_history.append(zero_action.copy())
        self._state_history.append(initial_state)

    def reset(self, scenario: Optional[dict] = None) -> np.ndarray:
        """Reset map/robot state and return initial observation.

        Args:
            scenario: Optional deterministic episode description with keys:
                - robot_pose: {x, y, heading}
                - obstacles: [{x, y, radius}, ...]
                - target: {x, y, radius}
        """
        self.robot.set_control_mode(self.control_mode)

        if scenario is None:
            self.robot.reset_random_pose(self.rng)

            obs_dicts = generate_obstacles(
                rng=self.rng,
                room_w=ROOM_WIDTH,
                room_h=ROOM_HEIGHT,
                robot_pose={"x": self.robot.x, "y": self.robot.y, "heading": self.robot.heading_deg},
                n_obs=self.n_obstacles_count if self.n_obstacles_count > 0 else int(self.rng.integers(3, 11)),
                obstacle_radius=0.05,
                robot_collision_radius=self.robot.collision_radius,
                robot_body_length=self.robot.BODY_LENGTH_M,
                robot_body_width=self.robot.BODY_WIDTH_M,
            )
            self.obstacles = [(float(o["x"]), float(o["y"]), float(o["radius"])) for o in obs_dicts]

            tgt = generate_target(
                rng=self.rng,
                room_w=ROOM_WIDTH,
                room_h=ROOM_HEIGHT,
                obstacles=obs_dicts,
                robot_pose={"x": self.robot.x, "y": self.robot.y, "heading": self.robot.heading_deg},
                target_r=0.075,
            )
            self.target = (float(tgt["x"]), float(tgt["y"]), float(tgt["radius"]))
            self.target_bbox_pose = generate_target_bbox_pose(self.rng)
        else:
            pose = scenario.get("robot_pose", {})
            tgt = scenario.get("target", {})
            self.robot.x = float(pose.get("x", self.robot.x))
            self.robot.y = float(pose.get("y", self.robot.y))
            self.robot.heading_deg = float(pose.get("heading", self.robot.heading_deg))
            self.robot.collision_flag = False
            self.robot.collision_flash_timer = 0.0

            self.obstacles = [
                (
                    float(o.get("x", 0.0)),
                    float(o.get("y", 0.0)),
                    float(o.get("radius", 0.05)),
                )
                for o in scenario.get("obstacles", [])
            ]
            self.target = (
                float(tgt.get("x", ROOM_WIDTH * 0.5)),
                float(tgt.get("y", ROOM_HEIGHT * 0.5)),
                float(tgt.get("radius", 0.075)),
            )
            bbox = scenario.get("target_bbox")
            if isinstance(bbox, dict):
                self.target_bbox_pose = {
                    "rotation_rad": float(bbox.get("rotation_rad", 0.0)),
                    "offset_x": float(bbox.get("offset_x", 0.0)),
                    "offset_y": float(bbox.get("offset_y", 0.0)),
                }
            else:
                self.target_bbox_pose = generate_target_bbox_pose(self.rng)

        self._reset_episode_state()
        return self._build_observation()

    def _encode_state(self) -> np.ndarray:
        """Raw 23-element state: 11 whisker lengths (m) + 11 target-bbox lengths (m)
        + heading-to-target (deg). Order matches train_mlp._flatten_history_features
        so IL weights transfer directly after dividing by the per-feature saturation
        scale vector (train_mlp.build_input_scale).
        """
        whiskers = np.asarray(self.robot.whisker_lengths, dtype=np.float32)
        bbox_lens = np.asarray(self.robot.target_bbox_lengths, dtype=np.float32)
        heading  = np.array([float(self.robot.heading_to_target_deg)], dtype=np.float32)
        return np.concatenate([whiskers, bbox_lens, heading], axis=0)

    def _encode_action_physical(self, action_normalized: np.ndarray) -> np.ndarray:
        """Convert a normalized [-1, 1] action to physical units for history storage."""
        physical = (np.asarray(action_normalized, dtype=np.float32) * self._phys_scales).astype(np.float32)
        return physical[self._action_feature_order]

    def _build_observation(self) -> np.ndarray:
        """Build IL-format flat obs: [s0, a0, s1, a1, ..., a_{N-2}, s_{N-1}].

        State entries are raw (un-normalised); action entries are in physical units.
        This layout is identical to the feature vectors produced by train_mlp.py,
        so IL model x_scale applies directly.
        """
        states  = list(self._state_history)
        actions = list(self._action_history)[:self.history_window - 1]
        parts: List[np.ndarray] = []
        for i, s in enumerate(states):
            parts.append(s)
            if i < len(actions):
                parts.append(actions[i])
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _distance_to_target(self) -> float:
        tx, ty, _ = self.target
        return float(math.hypot(self.robot.x - tx, self.robot.y - ty))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one fixed timestep and return gym-style transition tuple."""
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape[0] != self._n_action:
            raise ValueError(f"action must be shape ({self._n_action},)")
        action_arr = np.clip(action_arr, -1.0, 1.0)

        prev_distance = self._distance_to_target()

        self.robot.collision_flag = False
        self.robot.target_contact_flag = False
        self.robot.collision_source = None
        self.robot.apply_normalized_action(action_arr)
        self.whisker_segments = self.robot.step(
            self.dt, self.obstacles, target=self.target, target_bbox_pose=self.target_bbox_pose
        )
        self.robot.update_heading_to_target(self.target)

        self.simulated_time += self.dt
        self.episode_steps += 1
        self.trajectory_history.append((self.robot.x, self.robot.y))

        # Target contact (body overlap with target circle) preempts collision:
        # if the robot reached the goal on the same tick it grazed an obstacle,
        # the episode is a success.
        target_contact = bool(self.robot.target_contact_flag)
        collided_this_step = bool(self.robot.collision_flag) and not target_contact
        if collided_this_step:
            self.collision_count += 1

        curr_distance = self._distance_to_target()
        reward = (prev_distance - curr_distance) * self.distance_reduction_scale
        if collided_this_step:
            reward += self.collision_penalty
        reward += self.timestep_penalty

        reached_target = (curr_distance <= TARGET_REACH_RADIUS) or target_contact
        terminated = bool(reached_target)
        truncated = False
        stuck_failed = False
        collision_failed = False

        # Collision termination: treat collision as an immediate failure.
        if collided_this_step and self.terminate_on_collision and not target_contact:
            terminated = True
            collision_failed = True
            reward += self.collision_fail_penalty

        # Stuck termination: require >=1m progress per 60s window.
        if (not terminated) and (self.simulated_time - self._window_start_time) >= self.progress_window_s:
            window_progress = self._window_start_distance - curr_distance
            if window_progress < self.min_progress_m:
                stuck_failed = True
                truncated = True
            else:
                # Agent is making sufficient progress; start a new watchdog window.
                self._window_start_time = self.simulated_time
                self._window_start_distance = curr_distance

        if reached_target:
            reward += self.target_reached_bonus
        if stuck_failed:
            reward += self.timeout_penalty

        # Store physical action taken and resulting state for obs history.
        self._action_history.append(self._encode_action_physical(action_arr))
        self._state_history.append(self._encode_state())

        info = {
            "episode_steps": int(self.episode_steps),
            "collision_count": int(self.collision_count),
            "reached_target": bool(reached_target),
            "simulated_time": float(self.simulated_time),
            "stuck_failed": bool(stuck_failed),
            "collision_failed": bool(collision_failed),
        }
        return self._build_observation(), float(reward), terminated, truncated, info

    def get_render_state(self) -> dict:
        """Return episode state for external rendering."""
        return {
            "robot_pose": {
                "x": float(self.robot.x),
                "y": float(self.robot.y),
                "heading": float(self.robot.heading_deg),
            },
            "obstacles": [{"x": ox, "y": oy, "radius": r} for ox, oy, r in self.obstacles],
            "target": {"x": self.target[0], "y": self.target[1], "radius": self.target[2]},
            "target_bbox_pose": dict(self.target_bbox_pose),
            "whisker_lengths": [float(v) for v in self.robot.whisker_lengths],
            "target_bbox_lengths": [float(v) for v in self.robot.target_bbox_lengths],
            "trajectory_history": [(float(x), float(y)) for x, y in self.trajectory_history],
        }
