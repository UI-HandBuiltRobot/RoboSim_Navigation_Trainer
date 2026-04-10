import json
import math
import random
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None
from sim_core import (
    ROOM_HEIGHT,
    ROOM_WIDTH,
    TARGET_REACH_RADIUS,
    WHISKER_ANGLES,
    WHISKER_MAX_LENGTH,
    Robot,
    generate_obstacles,
    generate_target,
)

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None
    filedialog = None
    messagebox = None



    # Refactor verification: simulator runtime/UI behavior is intentionally preserved;
    # only shared simulation logic was moved into sim_core.py.

Vec2 = np.ndarray


class SimulatorApp:
    BG_COLOR = (235, 236, 240)
    PANEL_COLOR = (248, 248, 250)
    ROOM_BG = (250, 250, 250)
    ROOM_BORDER = (40, 40, 40)

    ROBOT_COLOR = (50, 90, 180)
    ROBOT_COLLIDE_COLOR = (230, 60, 60)
    WHEEL_COLOR = (20, 20, 20)

    WHISKER_HIT_COLOR = (220, 40, 40)
    WHISKER_FREE_COLOR = (90, 90, 90)

    TARGET_COLOR = (50, 180, 70)
    OBSTACLE_COLOR = (120, 120, 130)

    FPS = 30
    GOAL_REACH_RADIUS_M = TARGET_REACH_RADIUS

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("RoboSim Navigation Trainer")

        pygame.joystick.init()
        self.joystick: Optional[pygame.joystick.Joystick] = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

        self.input_mode = "keyboard"

        self.min_window_w = 980
        self.min_window_h = 680
        self.window_w = 1280
        self.window_h = 760
        self.screen = pygame.display.set_mode((self.window_w, self.window_h), pygame.RESIZABLE)
        self.robot_aligned_view = True

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 20)
        self.small_font = pygame.font.SysFont("consolas", 16)
        self.tooltip_font = pygame.font.SysFont("consolas", 14)
        self.mouse_pos = (0, 0)

        self.margin_px = 28
        self.toolbar_h = 152
        self.panel_w = 380

        self.robot = Robot()
        self.rng = np.random.default_rng()
        self.prev_control_mode = self.robot.control_mode
        self.obstacles: List[Tuple[float, float, float]] = []
        self.target: Optional[Tuple[float, float, float]] = None
        self.whisker_segments: List[Tuple[Vec2, Vec2, bool]] = []
        self.collision_display_timer = 0.0

        self.obstacle_count = random.randint(3, 10)
        self.logging_enabled = False
        self.active_log_rate_hz = 10.0
        self.in_memory_log: List[Dict] = []
        self.log_timer = 0.0
        self.history_len = 1
        self.history_buffer = deque(maxlen=self.history_len)
        self.log_dir = Path.cwd() / "logs"
        self.log_dir.mkdir(exist_ok=True)
        existing = [
            int(m.group(1))
            for f in self.log_dir.glob("run_*.jsonl")
            if (m := __import__("re").match(r"run_(\d+)_", f.name))
        ]
        self.run_counter = (max(existing) + 1) if existing else 0

        self.model_dir = Path.cwd() / "trained_models"
        self.loaded_model: Optional[Dict[str, np.ndarray]] = None
        self.loaded_model_ppo: Optional[PPO] = None
        self.loaded_model_type: Optional[str] = None  # "il" or "ppo"
        self.loaded_model_mode: Optional[str] = None
        self.selected_model_paths: Dict[str, Path] = {}
        self.model_status = "No model loaded"
        self.model_inference_buffer: deque = deque(maxlen=50)

        self.snapshot_dir = Path.cwd() / "dagger_snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)
        self.last_snapshot_path: Optional[Path] = None
        self.current_model_snapshot_path: Optional[Path] = None
        self.train_me_queue: List[Tuple[int, Path]] = []
        self.train_me_enqueue_seq = 1
        self.last_train_me_loaded_seq: Optional[int] = None
        self.active_train_me_case_seq: Optional[int] = None
        self.human_takeover_prompt_timer = 0.0

        self.prefs_open = False
        self.vis_mode = "action_radius"  # "all" | "action_radius" | "detected" | "sparse_sensing"
        self.obstacle_detected_timers: Dict[int, float] = {}
        self.sparse_detection_points: List[Dict[str, float]] = []

        self._recompute_layout()

        self.new_episode()

    def _recompute_layout(self) -> None:
        top = self.margin_px
        left = self.margin_px
        width = self.window_w - self.margin_px * 2

        self.toolbar_rect = pygame.Rect(left, top, width, self.toolbar_h)

        avail_w = self.window_w - self.margin_px * 3
        self.panel_w = min(420, max(320, int(avail_w * 0.36)))
        sim_w = max(420, avail_w - self.panel_w)
        panel_w = avail_w - sim_w
        self.panel_w = max(280, panel_w)

        sim_top = self.toolbar_rect.bottom + self.margin_px
        sim_h = self.window_h - sim_top - self.margin_px
        self.sim_rect = pygame.Rect(left, sim_top, sim_w, sim_h)
        self.panel_rect = pygame.Rect(self.sim_rect.right + self.margin_px, sim_top, self.panel_w, sim_h)

        self.room_rect_px, self.px_per_meter = self._compute_room_rect()

        tx, ty = self.toolbar_rect.left, self.toolbar_rect.top
        row1_y = ty + 10
        row2_y = ty + 56
        row3_y = ty + 102

        self.run_button_rect = pygame.Rect(tx + 12, row1_y, 110, 36)
        self.fail_button_rect = pygame.Rect(tx + 132, row1_y, 110, 36)

        obs_x = self.fail_button_rect.right + 50
        self.obs_minus_rect = pygame.Rect(obs_x + 80, row1_y + 6, 24, 24)
        self.obs_plus_rect = pygame.Rect(obs_x + 150, row1_y + 6, 24, 24)

        self.prefs_button_rect = pygame.Rect(self.toolbar_rect.right - 90, row2_y + 6, 80, 24)
        self.checkbox_logging_rect = pygame.Rect(self.toolbar_rect.right - 330, row1_y + 9, 18, 18)
        self.checkbox_robot_view_rect = pygame.Rect(self.toolbar_rect.right - 170, row1_y + 9, 18, 18)

        self.radio_keyboard_rect = pygame.Rect(tx + 120, row3_y, 18, 18)
        self.radio_gamepad_rect = pygame.Rect(tx + 245, row3_y, 18, 18)
        self.radio_model_rect = pygame.Rect(tx + 360, row3_y, 18, 18)

        model_center_x = self.toolbar_rect.centerx
        self.model_pick_rect = pygame.Rect(model_center_x - 36, row3_y + 7, 68, 22)
        self.model_reload_rect = pygame.Rect(model_center_x + 42, row3_y + 7, 72, 22)

        self.radio_heading_rect = pygame.Rect(tx + 120, row2_y, 18, 18)
        self.radio_xy_rect = pygame.Rect(tx + 280, row2_y, 18, 18)

        self.radio_heading_strafe_rect = pygame.Rect(tx + 415, row2_y, 18, 18)

        self._recompute_prefs_layout()

    def _recompute_prefs_layout(self) -> None:
        pw, ph = 460, 370
        px = (self.window_w - pw) // 2
        py = (self.window_h - ph) // 2
        self.prefs_rect = pygame.Rect(px, py, pw, ph)
        lx = px + 24
        self.prefs_close_rect = pygame.Rect(px + pw - 36, py + 8, 28, 28)
        row1 = py + 60    # History
        row2 = py + 110   # Rate Hz
        row4 = py + 210   # Radio: All
        row5 = py + 245   # Radio: Action radius
        row6 = py + 280   # Radio: Detected
        row7 = py + 315   # Radio: Sparse sensing
        self.prefs_hist_minus_rect = pygame.Rect(lx + 180, row1, 24, 24)
        self.prefs_hist_plus_rect  = pygame.Rect(lx + 250, row1, 24, 24)
        self.prefs_rate_minus_rect = pygame.Rect(lx + 180, row2, 24, 24)
        self.prefs_rate_plus_rect  = pygame.Rect(lx + 250, row2, 24, 24)
        self.prefs_vis_all_rect    = pygame.Rect(lx + 10, row4, 18, 18)
        self.prefs_vis_radius_rect = pygame.Rect(lx + 10, row5, 18, 18)
        self.prefs_vis_detect_rect = pygame.Rect(lx + 10, row6, 18, 18)
        self.prefs_vis_sparse_rect = pygame.Rect(lx + 10, row7, 18, 18)

    def _mode_output_desc(self, mode: str) -> str:
        if mode == "heading_drive":
            return "Trains outputs: drive_speed + rotation_rate"
        if mode == "xy_strafe":
            return "Trains outputs: vx + vy"
        if mode == "heading_strafe":
            return "Trains outputs: rotation_rate + vx + vy"
        return "Trains outputs for the selected control mode"

    def _labeled_radio_rect(self, rect: pygame.Rect, label: str) -> pygame.Rect:
        text_w, text_h = self.small_font.size(label)
        x = rect.right + 8
        y = rect.top - 2
        return pygame.Rect(x, y, text_w, text_h).union(rect)

    def _tooltip_for_pos(self, pos: Tuple[int, int]) -> Optional[str]:
        # If prefs panel is open, show prefs tips only
        if self.prefs_open:
            hist_field_rect = self.prefs_hist_minus_rect.union(self.prefs_hist_plus_rect).inflate(120, 10)
            rate_field_rect = self.prefs_rate_minus_rect.union(self.prefs_rate_plus_rect).inflate(120, 10)
            prefs_tips: List[Tuple[pygame.Rect, str]] = [
                (self.prefs_close_rect, "Close preferences panel"),
                (hist_field_rect, "History: logged context window length; larger values let the trainer learn temporal behavior"),
                (self.prefs_hist_minus_rect, "History -: shorten logging history window"),
                (self.prefs_hist_plus_rect, "History +: lengthen logging history window"),
                (rate_field_rect, "Rate Hz: active logging frequency; higher rates capture denser trajectories"),
                (self.prefs_rate_minus_rect, "Rate Hz -: decrease active logging frequency"),
                (self.prefs_rate_plus_rect, "Rate Hz +: increase active logging frequency"),
                (self.prefs_vis_all_rect, "Show all objects: render every obstacle regardless of proximity"),
                (self.prefs_vis_radius_rect, "Show within action radius: render only obstacles within whisker reach"),
                (self.prefs_vis_detect_rect, "Show when detected: obstacle appears when whisker hits it; fades after history_len / rate_hz seconds"),
                (self.prefs_vis_sparse_rect, "Sparse Sensing: render only whisker-detected world points with fading memory"),
            ]
            for rect, text in prefs_tips:
                if rect.collidepoint(pos):
                    return text
            return None

        obs_field_rect = self.obs_minus_rect.union(self.obs_plus_rect).inflate(120, 10)

        tips: List[Tuple[pygame.Rect, str]] = [
            (self.run_button_rect, "Run: start a new map or load next queued train-me case"),
            (self.fail_button_rect, "FAIL: in Model mode, queue this case for human takeover training"),
            (obs_field_rect, "Obstacles: number of obstacles per episode; higher values create harder training data"),
            (self.obs_minus_rect, "Obstacles -: reduce obstacle count"),
            (self.obs_plus_rect, "Obstacles +: increase obstacle count"),
            (self.prefs_button_rect, "Prefs: open preferences panel for History, Rate Hz, and Visualization settings"),
            (
                self._labeled_radio_rect(self.radio_heading_rect, "Heading+Drive"),
                "Control Mode Heading+Drive: rotate and drive forward/back; trainer learns drive_speed and turn rate",
            ),
            (
                self._labeled_radio_rect(self.radio_xy_rect, "XY Strafe"),
                "Control Mode XY Strafe: translation only; trainer learns vx and vy",
            ),
            (
                self._labeled_radio_rect(self.radio_heading_strafe_rect, "Hdg+Strafe"),
                "Control Mode Hdg+Strafe: full planar motion; trainer learns turn rate, vx, and vy",
            ),
            (
                self._labeled_radio_rect(self.radio_keyboard_rect, "Keyboard"),
                "Input Keyboard: teleoperate manually to generate supervised logs",
            ),
            (
                self._labeled_radio_rect(self.radio_gamepad_rect, "Gamepad"),
                "Input Gamepad: joystick teleoperation with deadband and cubic stick response",
            ),
            (
                self._labeled_radio_rect(self.radio_model_rect, "Model"),
                f"Input Model: run the loaded policy for {self._mode_output_desc(self.robot.control_mode)}",
            ),
            (self.model_pick_rect, "Pick: choose a specific model JSON for the current control mode"),
            (self.model_reload_rect, "Reload: reload selected model or latest model for this mode"),
            (
                self._labeled_radio_rect(self.checkbox_logging_rect, "Enable Logging"),
                "Enable Logging: write successful episodes to training logs when the goal is reached",
            ),
            (
                self._labeled_radio_rect(self.checkbox_robot_view_rect, "Robot View"),
                "Robot View: camera follows robot heading (forward is up) for easier directional driving",
            ),
        ]

        for rect, text in tips:
            if rect.collidepoint(pos):
                return text
        return None

    def _draw_tooltip(self, pos: Tuple[int, int], text: str) -> None:
        if not text:
            return

        pad = 6
        surf = self.tooltip_font.render(text, True, (22, 22, 22))
        box = surf.get_rect()
        box.left = pos[0] + 14
        box.top = pos[1] + 16
        box.width += pad * 2
        box.height += pad * 2

        if box.right > self.window_w - 8:
            box.right = self.window_w - 8
        if box.bottom > self.window_h - 8:
            box.bottom = self.window_h - 8

        pygame.draw.rect(self.screen, (255, 252, 210), box, border_radius=4)
        pygame.draw.rect(self.screen, (90, 90, 90), box, 1, border_radius=4)
        self.screen.blit(surf, (box.left + pad, box.top + pad))

    def _compute_room_rect(self) -> Tuple[pygame.Rect, float]:
        room_aspect = Robot.ROOM_WIDTH_M / Robot.ROOM_HEIGHT_M
        avail_w = max(80, self.sim_rect.width - 2 * self.margin_px)
        avail_h = max(80, self.sim_rect.height - 2 * self.margin_px)
        avail_aspect = avail_w / avail_h

        if avail_aspect >= room_aspect:
            h = avail_h
            w = int(h * room_aspect)
        else:
            w = avail_w
            h = int(w / room_aspect)

        x = self.sim_rect.left + (self.sim_rect.width - w) // 2
        y = self.sim_rect.top + (self.sim_rect.height - h) // 2

        px_per_meter = w / Robot.ROOM_WIDTH_M

        return pygame.Rect(x, y, int(w), int(h)), px_per_meter

    def world_to_screen(self, x_m: float, y_m: float) -> Tuple[int, int]:
        if not self.robot_aligned_view:
            x_px = self.room_rect_px.left + x_m * self.px_per_meter
            y_px = self.room_rect_px.bottom - y_m * self.px_per_meter
            return int(x_px), int(y_px)

        # Camera follows robot pose: forward is up on screen, left is left on screen.
        local_x, local_y = self.robot._basis_vectors()
        rel = np.array([x_m - self.robot.x, y_m - self.robot.y], dtype=float)
        forward = float(np.dot(rel, local_x))
        left = float(np.dot(rel, local_y))

        cx = self.room_rect_px.centerx
        cy = self.room_rect_px.centery
        x_px = cx - left * self.px_per_meter
        y_px = cy - forward * self.px_per_meter
        return int(x_px), int(y_px)

    def _point_to_segment_distance(self, p: Vec2, a: Vec2, b: Vec2) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        closest = a + t * ab
        return float(np.linalg.norm(p - closest))

    def _robot_edge_segments(self) -> List[Tuple[Vec2, Vec2]]:
        corners = self.robot.body_corners_world()
        return [
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0]),
        ]

    def _spawn_obstacles(self, n_obs: int) -> List[Tuple[float, float, float]]:
        robot_pose = {"x": self.robot.x, "y": self.robot.y, "heading": self.robot.heading_deg}
        obs = generate_obstacles(
            rng=self.rng,
            room_w=ROOM_WIDTH,
            room_h=ROOM_HEIGHT,
            robot_pose=robot_pose,
            n_obs=n_obs,
            obstacle_radius=0.05,
            robot_collision_radius=self.robot.collision_radius,
            robot_body_length=self.robot.BODY_LENGTH_M,
            robot_body_width=self.robot.BODY_WIDTH_M,
        )
        return [(float(o["x"]), float(o["y"]), float(o["radius"])) for o in obs]

    def _spawn_target(self, obstacles: List[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
        obstacle_dicts = [{"x": ox, "y": oy, "radius": orad} for ox, oy, orad in obstacles]
        robot_pose = {"x": self.robot.x, "y": self.robot.y, "heading": self.robot.heading_deg}
        tgt = generate_target(
            rng=self.rng,
            room_w=ROOM_WIDTH,
            room_h=ROOM_HEIGHT,
            obstacles=obstacle_dicts,
            robot_pose=robot_pose,
            target_r=0.075,
        )
        return float(tgt["x"]), float(tgt["y"]), float(tgt["radius"])

    def new_episode(self) -> None:
        self.active_train_me_case_seq = None
        self.obstacle_detected_timers = {}
        self.robot.reset_random_pose(self.rng)
        self.collision_display_timer = 0.0
        self.in_memory_log = []
        self.log_timer = 0.0
        self.history_buffer.clear()
        self.model_inference_buffer.clear()
        n_obs = max(1, int(self.obstacle_count))
        
        # If switching to strafe mode, orient robot toward the target.
        if self.robot.control_mode == "xy_strafe" and self.prev_control_mode != "xy_strafe":
            self.obstacles = self._spawn_obstacles(n_obs)
            self.target = self._spawn_target(self.obstacles)
            
            if self.target is not None:
                tx, ty, _ = self.target
                target_angle = math.degrees(math.atan2(tx - self.robot.x, ty - self.robot.y))
                self.robot.heading_deg = target_angle
        else:
            self.obstacles = self._spawn_obstacles(n_obs)
            self.target = self._spawn_target(self.obstacles)
        
        self.whisker_segments = self.robot.compute_whiskers(self.obstacles)
        self.robot.update_heading_to_target(self.target)
        self.prev_control_mode = self.robot.control_mode

        if self.input_mode == "model":
            self.current_model_snapshot_path = self._save_dagger_snapshot("model_start")
        else:
            self.current_model_snapshot_path = None

    def update_robot_command(self) -> None:
        # Detect control mode change and auto-generate new episode.
        if self.robot.control_mode != self.prev_control_mode:
            self.new_episode()
            return
        
        if self.input_mode == "keyboard":
            keys = pygame.key.get_pressed()
            self.robot.update_command_from_keys(keys)
        elif self.input_mode == "gamepad":
            self.robot.update_command_from_gamepad(self.joystick)
        else:
            if self.loaded_model_mode != self.robot.control_mode:
                self._load_latest_model_for_mode(self.robot.control_mode)
            self._update_command_from_model()
    
    def check_goal_reached(self) -> bool:
        """Check if robot has reached the target."""
        if self.target is None:
            return False
        tx, ty, _ = self.target
        dist = math.hypot(self.robot.x - tx, self.robot.y - ty)
        return dist <= self.GOAL_REACH_RADIUS_M
    
    def check_whisker_collision(self) -> bool:
        """Check if any whisker is shorter than 100mm."""
        WHISKER_COLLISION_THRESHOLD_M = 0.10
        return any(length < WHISKER_COLLISION_THRESHOLD_M for length in self.robot.whisker_lengths)

    def _has_nonzero_input(self) -> bool:
        """Check if user has non-zero control input."""
        cmd = self.robot.drive_command
        if self.robot.control_mode == "heading_drive":
            return cmd.get("rotation_rate", 0.0) != 0.0 or cmd.get("drive_speed", 0.0) != 0.0
        elif self.robot.control_mode == "heading_strafe":
            return cmd.get("rotation_rate", 0.0) != 0.0 or cmd.get("vx", 0.0) != 0.0 or cmd.get("vy", 0.0) != 0.0
        else:
            return cmd.get("vx", 0.0) != 0.0 or cmd.get("vy", 0.0) != 0.0

    def _obstacle_within_action_radius(self) -> bool:
        """Check if any obstacle is within action radius (whisker max length)."""
        DETECTION_RADIUS_M = 0.50
        for ox, oy, or_ in self.obstacles:
            dist = math.hypot(self.robot.x - ox, self.robot.y - oy)
            if dist - or_ <= DETECTION_RADIUS_M:
                return True
        return False

    def _is_obstacle_in_action_radius(self, obstacle: Tuple[float, float, float]) -> bool:
        """Check if one obstacle is within action radius of the robot."""
        ox, oy, or_ = obstacle
        detection_radius_m = 0.50
        dist = math.hypot(self.robot.x - ox, self.robot.y - oy)
        return (dist - or_) <= detection_radius_m

    def _update_obstacle_detection(self, dt: float) -> None:
        if self.vis_mode != "detected":
            self.obstacle_detected_timers.clear()
            return
        persistence = self.history_len / max(0.1, self.active_log_rate_hz)
        for i, (ox, oy, radius) in enumerate(self.obstacles):
            for _start, endpoint, hit in self.whisker_segments:
                if not hit:
                    continue
                if math.hypot(float(endpoint[0]) - ox, float(endpoint[1]) - oy) <= radius + 0.02:
                    self.obstacle_detected_timers[i] = persistence
                    break
        for idx in list(self.obstacle_detected_timers.keys()):
            self.obstacle_detected_timers[idx] -= dt
            if self.obstacle_detected_timers[idx] <= 0.0:
                del self.obstacle_detected_timers[idx]

    def _current_model_memory_len(self) -> int:
        mode = self.robot.control_mode
        n_action = len(self._command_keys_for_mode(mode))
        state_dim = 12  # 11 whiskers + heading_to_target
        if self.loaded_model_ppo is not None:
            obs_dim = int(self.loaded_model_ppo.observation_space.shape[0])
            denom = state_dim + n_action
            if obs_dim >= state_dim and denom > 0 and (obs_dim + n_action) % denom == 0:
                return max(1, (obs_dim + n_action) // denom)
        if self.loaded_model is not None:
            input_dim = int(self.loaded_model.get("input_dim", 0))
            denom = state_dim + n_action
            if input_dim >= state_dim and denom > 0 and (input_dim + n_action) % denom == 0:
                return max(1, (input_dim + n_action) // denom)
        return max(1, int(self.history_len))

    def _update_sparse_detection_points(self) -> None:
        n = self._current_model_memory_len()

        for p in self.sparse_detection_points:
            p["age"] += 1
        self.sparse_detection_points = [p for p in self.sparse_detection_points if p["age"] <= n]

        # Use exact whisker ray endpoints from physics to avoid frame/origin mismatch.
        # compute_whiskers() uses a forward-offset ray origin, not the robot center.
        for _start, endpoint, hit in self.whisker_segments:
            if not hit:
                continue
            self.sparse_detection_points.append(
                {"x": float(endpoint[0]), "y": float(endpoint[1]), "age": 0}
            )

    def _should_log(self) -> bool:
        """Determine if logging should occur at all this frame."""
        return self._current_log_interval() is not None

    def _current_log_interval(self) -> Optional[float]:
        """Return current logging interval in seconds based on scene context.

        - active_log_rate_hz: obstacle in action radius and non-zero control input.
        - 10% of active_log_rate_hz: no obstacle in action radius.
        - None: logging disabled, or obstacle nearby with zero control input.
        """
        if not self.logging_enabled:
            return None

        obstacle_near = self._obstacle_within_action_radius()
        if obstacle_near and self._has_nonzero_input():
            return 1.0 / max(0.1, self.active_log_rate_hz)
        if not obstacle_near:
            return 10.0 / max(0.1, self.active_log_rate_hz)
        return None

    def _collect_log_entry(self) -> Dict:
        """Collect a structured log row containing a fixed-length history window."""
        mode = self.robot.control_mode
        self.history_buffer.append(self._capture_timestep_record(mode))

        history = list(self.history_buffer)
        while len(history) < self.history_len:
            history.insert(0, self._zero_timestep_record(mode))

        return {
            "schema_version": 2,
            "timestamp": time.time(),
            "mode": mode,
            "history_len": self.history_len,
            "active_log_rate_hz": float(self.active_log_rate_hz),
            "history": history,
        }

    def _mode_suffix(self, mode: str) -> str:
        mode_map = {"heading_drive": "heading", "xy_strafe": "strafe", "heading_strafe": "heading_strafe"}
        return mode_map.get(mode, "other")

    def _command_keys_for_mode(self, mode: str) -> List[str]:
        if mode == "heading_drive":
            return ["drive_speed", "rotation_rate"]
        if mode == "heading_strafe":
            return ["rotation_rate", "vx", "vy"]
        return ["vx", "vy"]

    def _capture_timestep_record(self, mode: str) -> Dict:
        action_keys = self._command_keys_for_mode(mode)
        action = {k: float(self.robot.drive_command.get(k, 0.0)) for k in action_keys}
        return {
            "mode": mode,
            "whisker_lengths": [float(v) for v in self.robot.whisker_lengths],
            "heading_to_target": float(self.robot.heading_to_target_deg),
            "action": action,
        }

    def _zero_timestep_record(self, mode: str) -> Dict:
        action_keys = self._command_keys_for_mode(mode)
        return {
            "mode": mode,
            "whisker_lengths": [0.0 for _ in Robot.WHISKER_ANGLES_DEG],
            "heading_to_target": 0.0,
            "action": {k: 0.0 for k in action_keys},
        }

    def _write_log_to_file(self) -> None:
        """Write the in-memory log to a file."""
        if not self.in_memory_log:
            return

        mode_suffix = self._mode_suffix(self.robot.control_mode)
        bucket_dir = self.log_dir / f"{mode_suffix}_mem{self.history_len}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(bucket_dir.glob(f"run_*_{mode_suffix}_mem{self.history_len}.jsonl"))
        if existing:
            try:
                run_idx = int(existing[-1].stem.split("_")[1]) + 1
            except Exception:
                run_idx = len(existing)
        else:
            run_idx = 0
        filename = f"run_{run_idx:03d}_{mode_suffix}_mem{self.history_len}.jsonl"
        filepath = bucket_dir / filename
        
        try:
            with open(filepath, "w") as f:
                for entry in self.in_memory_log:
                    f.write(json.dumps(entry) + "\n")
            self.run_counter += 1
        except Exception as e:
            print(f"Error writing log file {filepath}: {e}")

    def _load_latest_model_for_mode(self, mode: str) -> bool:
        """Load the latest trained model artifact for the requested control mode."""
        suffix_map = {
            "heading_drive": "heading",
            "xy_strafe": "strafe",
            "heading_strafe": "heading_strafe",
        }
        suffix = suffix_map.get(mode, "other")
        model_files = sorted(self.model_dir.glob(f"model_{suffix}_*.json"))
        if not model_files:
            self.loaded_model = None
            self.loaded_model_mode = None
            self.model_status = f"No {suffix} model"
            return False

        model_path = model_files[-1]
        return self._load_model_from_file(model_path, mode)

    def _load_model_from_file(self, model_path: Path, expected_mode: str) -> bool:
        """Load IL JSON or PPO ZIP model and validate control mode match."""
        try:
            if str(model_path).lower().endswith(".zip"):
                # Load PPO ZIP model
                if PPO is None:
                    self.model_status = "SB3 not installed; cannot load ZIP"
                    return False
                try:
                    ppo_model = PPO.load(str(model_path.with_suffix("")))
                    self.loaded_model_ppo = ppo_model
                    self.loaded_model = None
                    self.loaded_model_type = "ppo"
                except Exception as e:
                    self.model_status = f"PPO load error: {str(e)[:50]}"
                    return False
            else:
                # Load IL JSON model
                with open(model_path, "r", encoding="utf-8") as f:
                    blob = json.load(f)

                file_mode = blob.get("mode")
                if file_mode is not None and file_mode != expected_mode:
                    self.loaded_model = None
                    self.loaded_model_mode = None
                    self.model_status = f"Mode mismatch in {model_path.name}"
                    return False

                weights = [np.asarray(w, dtype=np.float64) for w in blob["weights"]]
                biases = [np.asarray(b, dtype=np.float64) for b in blob["biases"]]

                self.loaded_model = {
                    "weights": weights,
                    "biases": biases,
                    "x_mean": np.asarray(blob["x_mean"], dtype=np.float64),
                    "x_std": np.asarray(blob["x_std"], dtype=np.float64),
                    "y_mean": np.asarray(blob["y_mean"], dtype=np.float64),
                    "y_std": np.asarray(blob["y_std"], dtype=np.float64),
                    "input_dim": int(blob.get("input_dim", weights[0].shape[0])),
                }
                self.loaded_model_ppo = None
                self.loaded_model_type = "il"

            self.model_inference_buffer.clear()
            self.loaded_model_mode = expected_mode
            self.selected_model_paths[expected_mode] = model_path
            inferred_len = self._current_model_memory_len()
            self.history_len = int(inferred_len)
            self.history_buffer = deque(list(self.history_buffer)[-self.history_len:], maxlen=self.history_len)
            self.model_status = f"Loaded {model_path.name} (mem={self.history_len})"
            return True
        except Exception as e:
            self.loaded_model = None
            self.loaded_model_ppo = None
            self.loaded_model_type = None
            self.loaded_model_mode = None
            self.model_status = f"Model load failed: {str(e)[:40]}"
            return False

    def _pick_model_for_current_mode(self) -> bool:
        """Open a file picker and load a selected model for active control mode."""
        if filedialog is None or tk is None:
            self.model_status = "File picker unavailable"
            return False

        model_path_str = self._safe_askopenfilename(
            title="Select trained model",
            initialdir=str(self.model_dir),
            filetypes=[("Model files", "*.json *.zip"), ("IL JSON", "*.json"), ("PPO ZIP", "*.zip"), ("All files", "*.*")],
        )
        if model_path_str is None:
            self.model_status = "File picker failed"
            return False

        if not model_path_str:
            return False

        return self._load_model_from_file(Path(model_path_str), self.robot.control_mode)

    def _reload_model_for_current_mode(self) -> bool:
        """Reload currently selected mode model; if none selected, load latest."""
        selected = self.selected_model_paths.get(self.robot.control_mode)
        if selected is not None and selected.exists():
            return self._load_model_from_file(selected, self.robot.control_mode)
        return self._load_latest_model_for_mode(self.robot.control_mode)

    def _save_dagger_snapshot(self, tag: str = "snapshot") -> Optional[Path]:
        """Persist a snapshot of the current map and robot state for DAGGER handoff."""
        snapshot = {
            "timestamp": time.time(),
            "control_mode": self.robot.control_mode,
            "input_mode": self.input_mode,
            "robot_pose": {
                "x": self.robot.x,
                "y": self.robot.y,
                "heading": self.robot.heading_deg,
            },
            "target": self.target,
            "obstacles": self.obstacles,
            "obstacle_count": self.obstacle_count,
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.snapshot_dir / f"{tag}_{ts}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f)
            self.last_snapshot_path = path
            return path
        except Exception:
            self.last_snapshot_path = None
            return None

    def _load_dagger_snapshot(self, snapshot_path: Path) -> bool:
        """Restore map and robot state from a saved DAGGER snapshot."""
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                snap = json.load(f)

            robot_pose = snap["robot_pose"]
            self.robot.set_control_mode(snap.get("control_mode", self.robot.control_mode))
            self.robot.x = float(robot_pose["x"])
            self.robot.y = float(robot_pose["y"])
            self.robot.heading_deg = float(robot_pose["heading"])
            self.robot.collision_flag = False
            self.robot.collision_flash_timer = 0.0

            self.obstacles = [tuple(ob) for ob in snap.get("obstacles", [])]
            tgt = snap.get("target")
            self.target = tuple(tgt) if tgt is not None else None
            self.obstacle_count = int(snap.get("obstacle_count", self.obstacle_count))

            self.whisker_segments = self.robot.compute_whiskers(self.obstacles)
            self.robot.update_heading_to_target(self.target)
            self.prev_control_mode = self.robot.control_mode
            return True
        except Exception:
            return False

    def _enqueue_train_me_case(self, reason: str) -> None:
        """Add current model run snapshot to the train-me queue."""
        snap = self.current_model_snapshot_path
        if snap is None or not snap.exists():
            snap = self._save_dagger_snapshot("model_start")
            self.current_model_snapshot_path = snap

        if snap is None:
            self.model_status = f"{reason}: snapshot save failed"
            return

        seq = self.train_me_enqueue_seq
        self.train_me_enqueue_seq += 1
        self.train_me_queue.append((seq, snap))
        self.model_status = f"Queued case #{seq} ({reason}) | {len(self.train_me_queue)} pending"

    def _prompt_run_train_me(self) -> bool:
        """Ask whether to run queued train-me cases."""
        if not self.train_me_queue:
            return False

        if messagebox is None or tk is None:
            return True

        res = self._safe_askyesno(
            title="Train-Me Queue",
            message=(
                f"There are {len(self.train_me_queue)} queued train-me cases.\n"
                "Do you want to go through the next case now?"
            ),
            default_on_error=True,
        )
        return True if res is None else bool(res)

    def _safe_askopenfilename(self, title: str, initialdir: str, filetypes: List[Tuple[str, str]]) -> Optional[str]:
        """Open a file picker without freezing the pygame loop on Linux.

        On Linux, Tk + SDL windows can deadlock when both run in one process.
        To avoid this, run the file dialog in a short-lived Python subprocess.
        """
        if filedialog is None or tk is None:
            return None

        if sys.platform.startswith("linux"):
            payload = {
                "title": title,
                "initialdir": initialdir,
                "filetypes": filetypes,
            }
            script = (
                "import json,sys\n"
                "import tkinter as tk\n"
                "from tkinter import filedialog\n"
                "cfg=json.loads(sys.argv[1])\n"
                "root=tk.Tk()\n"
                "root.withdraw()\n"
                "root.update_idletasks()\n"
                "path=filedialog.askopenfilename(title=cfg['title'], initialdir=cfg['initialdir'], filetypes=cfg['filetypes'])\n"
                "print(path or '')\n"
                "root.destroy()\n"
            )
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", script, json.dumps(payload)],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                )
                if proc.returncode != 0:
                    return None
                return proc.stdout.strip()
            except Exception:
                return None

        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.update_idletasks()
            path = filedialog.askopenfilename(
                title=title,
                initialdir=initialdir,
                filetypes=filetypes,
            )
            root.destroy()
            return path
        except Exception:
            return None

    def _safe_askyesno(self, title: str, message: str, default_on_error: bool) -> Optional[bool]:
        """Open a yes/no dialog safely; Linux uses subprocess isolation."""
        if messagebox is None or tk is None:
            return None

        if sys.platform.startswith("linux"):
            payload = {
                "title": title,
                "message": message,
            }
            script = (
                "import json,sys\n"
                "import tkinter as tk\n"
                "from tkinter import messagebox\n"
                "cfg=json.loads(sys.argv[1])\n"
                "root=tk.Tk()\n"
                "root.withdraw()\n"
                "root.update_idletasks()\n"
                "ans=messagebox.askyesno(cfg['title'], cfg['message'], parent=root)\n"
                "print('1' if ans else '0')\n"
                "root.destroy()\n"
            )
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", script, json.dumps(payload)],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                )
                if proc.returncode != 0:
                    return default_on_error
                return proc.stdout.strip() == "1"
            except Exception:
                return default_on_error

        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.update_idletasks()
            go = messagebox.askyesno(title, message, parent=root)
            root.destroy()
            return bool(go)
        except Exception:
            return default_on_error

    def _run_next_train_me_case(self) -> bool:
        """Load the next queued train-me map and prompt for human navigation."""
        if not self.train_me_queue:
            return False

        seq, snapshot_path = self.train_me_queue.pop(0)
        if not self._load_dagger_snapshot(snapshot_path):
            self.model_status = f"Failed to load case #{seq}: {snapshot_path.name}"
            return False

        self.human_takeover_prompt_timer = 4.0
        self.active_train_me_case_seq = seq
        self.last_train_me_loaded_seq = seq
        self.model_status = f"Loaded case #{seq} ({len(self.train_me_queue)} remaining)"
        return True

    def _trigger_dagger_human_takeover(self) -> None:
        """Queue current model-run map for later human takeover training."""
        self._enqueue_train_me_case("FAIL")

    def _predict_model_output(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Run forward pass with either IL (NumPy MLP) or PPO (SB3) model."""
        if self.loaded_model_type == "ppo":
            if self.loaded_model_ppo is None:
                return None
            # PPO.predict returns (action, _); action is already in [-1, 1] normalized space.
            action, _ = self.loaded_model_ppo.predict(features, deterministic=True)
            return action.reshape(1, -1)
        else:
            # IL (NumPy MLP) inference
            if self.loaded_model is None:
                return None
            x_mean = self.loaded_model["x_mean"]
            x_std = np.where(np.abs(self.loaded_model["x_std"]) < 1e-6, 1.0, self.loaded_model["x_std"])
            y_mean = self.loaded_model["y_mean"]
            y_std = np.where(np.abs(self.loaded_model["y_std"]) < 1e-6, 1.0, self.loaded_model["y_std"])
            weights = self.loaded_model["weights"]
            biases = self.loaded_model["biases"]

            a = (features - x_mean) / x_std
            for i in range(len(weights) - 1):
                a = np.maximum(0.0, a @ weights[i] + biases[i])
            out_norm = a @ weights[-1] + biases[-1]
            return out_norm * y_std + y_mean

    def _update_command_from_model(self) -> None:
        """Compute drive commands from loaded model and current sensor state."""
        if self.loaded_model is None and self.loaded_model_ppo is None:
            if self.robot.control_mode == "heading_drive":
                self.robot.drive_command = {"rotation_rate": 0.0, "drive_speed": 0.0}
            elif self.robot.control_mode == "heading_strafe":
                self.robot.drive_command = {"rotation_rate": 0.0, "vx": 0.0, "vy": 0.0}
            else:
                self.robot.drive_command = {"vx": 0.0, "vy": 0.0}
            return

        mode = self.robot.control_mode
        action_keys = self._command_keys_for_mode(mode)
        n_action = len(action_keys)
        state_dim = 12  # 11 whiskers + heading_to_target
        
        # Derive history_len from model type
        if self.loaded_model_type == "ppo":
            # PPO: obs_dim = 12*N + n_action*(N-1) = (12 + n_action)*N - n_action
            # => N = (obs_dim + n_action) / (12 + n_action)
            obs_dim = self.loaded_model_ppo.observation_space.shape[0]
            history_len = (obs_dim + n_action) // (state_dim + n_action)
        else:
            # IL: use input_dim from model metadata
            input_dim = self.loaded_model["input_dim"]
            history_len = (input_dim + n_action) // (state_dim + n_action)

        # Capture current state (action field = last issued drive_command)
        self.model_inference_buffer.append(self._capture_timestep_record(mode))

        # Build history window of length history_len, zero-padding if needed
        history = list(self.model_inference_buffer)[-history_len:]
        while len(history) < history_len:
            history.insert(0, self._zero_timestep_record(mode))

        # Flatten: each step contributes state; all but the last also contribute action
        feat: List[float] = []
        for i, step in enumerate(history):
            feat.extend([float(v) for v in step["whisker_lengths"]] + [float(step["heading_to_target"])])
            if i < history_len - 1:
                action_map = step.get("action", {})
                for k in action_keys:
                    feat.append(float(action_map.get(k, 0.0)))

        features = np.asarray(feat, dtype=np.float32 if self.loaded_model_type == "ppo" else np.float64).reshape(1, -1)
        pred = self._predict_model_output(features)
        if pred is None:
            return

        out = pred[0]
        if self.robot.control_mode == "heading_drive":
            # For IL: out = [drive_speed, rotation_rate]; for PPO: out = [rotation_rate, drive_speed]
            if self.loaded_model_type == "ppo":
                rotation_rate = float(np.clip(out[0], -1.0, 1.0)) * Robot.GAMEPAD_MAX_ROTATE_RATE_DPS
                drive_speed = float(np.clip(out[1], -1.0, 1.0)) * Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS
            else:
                drive_speed = float(np.clip(out[0], -Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS, Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS))
                rotation_rate = float(np.clip(out[1], -Robot.GAMEPAD_MAX_ROTATE_RATE_DPS, Robot.GAMEPAD_MAX_ROTATE_RATE_DPS))
            self.robot.drive_command = {"rotation_rate": rotation_rate, "drive_speed": drive_speed}
        elif self.robot.control_mode == "heading_strafe":
            if self.loaded_model_type == "ppo":
                rotation_rate = float(np.clip(out[0], -1.0, 1.0)) * Robot.GAMEPAD_MAX_ROTATE_RATE_DPS
                vx = float(np.clip(out[1], -1.0, 1.0)) * Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS
                vy = float(np.clip(out[2], -1.0, 1.0)) * Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS if len(out) > 2 else 0.0
            else:
                rotation_rate = float(np.clip(out[0], -Robot.GAMEPAD_MAX_ROTATE_RATE_DPS, Robot.GAMEPAD_MAX_ROTATE_RATE_DPS))
                vx = float(np.clip(out[1], -Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS, Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS))
                vy = float(np.clip(out[2], -Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS, Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS)) if len(out) > 2 else 0.0
            self.robot.drive_command = {"rotation_rate": rotation_rate, "vx": vx, "vy": vy}
        else:
            if self.loaded_model_type == "ppo":
                vx = float(np.clip(out[0], -1.0, 1.0)) * Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS
                vy = float(np.clip(out[1], -1.0, 1.0)) * Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS
            else:
                vx = float(np.clip(out[0], -Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS, Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS))
                vy = float(np.clip(out[1], -Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS, Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS))
            self.robot.drive_command = {"vx": vx, "vy": vy}

    def _handle_click(self, pos: Tuple[int, int]) -> None:
        # If prefs panel is open, handle its clicks first
        if self.prefs_open:
            if self.prefs_close_rect.collidepoint(pos):
                self.prefs_open = False
                return
            if not self.prefs_rect.collidepoint(pos):
                self.prefs_open = False
                return
            if self.prefs_hist_minus_rect.collidepoint(pos):
                self.history_len = max(1, self.history_len - 1)
                self.history_buffer = deque(list(self.history_buffer)[-self.history_len:], maxlen=self.history_len)
                return
            if self.prefs_hist_plus_rect.collidepoint(pos):
                self.history_len = min(10, self.history_len + 1)
                self.history_buffer = deque(list(self.history_buffer)[-self.history_len:], maxlen=self.history_len)
                return
            if self.prefs_rate_minus_rect.collidepoint(pos):
                self.active_log_rate_hz = max(1.0, self.active_log_rate_hz - 1.0)
                return
            if self.prefs_rate_plus_rect.collidepoint(pos):
                self.active_log_rate_hz = min(30.0, self.active_log_rate_hz + 1.0)
                return
            if self.prefs_vis_all_rect.collidepoint(pos):
                self.vis_mode = "all"
                return
            if self.prefs_vis_radius_rect.collidepoint(pos):
                self.vis_mode = "action_radius"
                return
            if self.prefs_vis_detect_rect.collidepoint(pos):
                self.vis_mode = "detected"
                return
            if self.prefs_vis_sparse_rect.collidepoint(pos):
                self.vis_mode = "sparse_sensing"
                return
            return  # Swallow remaining clicks inside panel

        if self.prefs_button_rect.collidepoint(pos):
            self.prefs_open = True
            return

        if self.run_button_rect.collidepoint(pos):
            self.sparse_detection_points.clear()
            if self.train_me_queue and self._prompt_run_train_me():
                self._run_next_train_me_case()
            else:
                self.new_episode()
            return

        if self.fail_button_rect.collidepoint(pos):
            if self.input_mode == "model":
                self._trigger_dagger_human_takeover()
                self.new_episode()
            else:
                self.model_status = "FAIL button requires Model input mode"
            return

        if self.obs_minus_rect.collidepoint(pos):
            self.obstacle_count = max(1, self.obstacle_count - 1)
            return

        if self.obs_plus_rect.collidepoint(pos):
            self.obstacle_count = min(50, self.obstacle_count + 1)
            return

        if self.radio_heading_rect.collidepoint(pos):
            self.robot.set_control_mode("heading_drive")
            return

        if self.radio_xy_rect.collidepoint(pos):
            self.robot.set_control_mode("xy_strafe")
            return

        if self.radio_heading_strafe_rect.collidepoint(pos):
            self.robot.set_control_mode("heading_strafe")
            return

        if self.radio_keyboard_rect.collidepoint(pos):
            self.input_mode = "keyboard"
            return

        if self.radio_gamepad_rect.collidepoint(pos):
            self.input_mode = "gamepad"
            return

        if self.radio_model_rect.collidepoint(pos):
            self.input_mode = "model"
            self._reload_model_for_current_mode()
            return

        if self.model_pick_rect.collidepoint(pos):
            self._pick_model_for_current_mode()
            return

        if self.model_reload_rect.collidepoint(pos):
            self._reload_model_for_current_mode()
            return

        if self.checkbox_logging_rect.collidepoint(pos):
            self.logging_enabled = not self.logging_enabled
            return
        if self.checkbox_robot_view_rect.collidepoint(pos):
            self.robot_aligned_view = not self.robot_aligned_view
            return
    def _draw_arrow(self, start_m: Vec2, vec_m: Vec2, color: Tuple[int, int, int], width: int = 3) -> None:
        start_px = self.world_to_screen(float(start_m[0]), float(start_m[1]))
        end = start_m + vec_m
        end_px = self.world_to_screen(float(end[0]), float(end[1]))
        pygame.draw.line(self.screen, color, start_px, end_px, width)

        dx = end_px[0] - start_px[0]
        dy = end_px[1] - start_px[1]
        angle = math.atan2(dy, dx)
        ah = 10

        left = (
            int(end_px[0] - ah * math.cos(angle - math.pi / 6)),
            int(end_px[1] - ah * math.sin(angle - math.pi / 6)),
        )
        right = (
            int(end_px[0] - ah * math.cos(angle + math.pi / 6)),
            int(end_px[1] - ah * math.sin(angle + math.pi / 6)),
        )
        pygame.draw.polygon(self.screen, color, [end_px, left, right])

    def _draw_goal_compass_arrow(self) -> None:
        """Draw a compass-style arrow (0.6m) from robot origin toward current goal."""
        if self.target is None:
            return

        tx, ty, _ = self.target
        dx = tx - self.robot.x
        dy = ty - self.robot.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return

        arrow_vec = np.array([dx / dist, dy / dist], dtype=float) * 0.6
        start = np.array([self.robot.x, self.robot.y], dtype=float)
        self._draw_arrow(start, arrow_vec, (245, 190, 35), 3)

    def _draw_simulation(self) -> None:
        if self.vis_mode == "sparse_sensing":
            pygame.draw.rect(self.screen, (0, 0, 0), self.sim_rect)
        else:
            pygame.draw.rect(self.screen, self.ROOM_BG, self.sim_rect)

        self.screen.set_clip(self.sim_rect)
        if self.robot_aligned_view:
            room_corners = [
                self.world_to_screen(0.0, 0.0),
                self.world_to_screen(Robot.ROOM_WIDTH_M, 0.0),
                self.world_to_screen(Robot.ROOM_WIDTH_M, Robot.ROOM_HEIGHT_M),
                self.world_to_screen(0.0, Robot.ROOM_HEIGHT_M),
            ]
            if self.vis_mode != "sparse_sensing":
                pygame.draw.polygon(self.screen, self.ROOM_BG, room_corners)
            pygame.draw.polygon(self.screen, self.ROOM_BORDER, room_corners, 2)
        else:
            pygame.draw.rect(self.screen, self.ROOM_BORDER, self.room_rect_px, 2)

        if self.vis_mode != "sparse_sensing":
            for i, obstacle in enumerate(self.obstacles):
                if self.vis_mode == "all":
                    visible = True
                elif self.vis_mode == "action_radius":
                    visible = self._is_obstacle_in_action_radius(obstacle)
                else:
                    visible = i in self.obstacle_detected_timers
                if not visible:
                    continue
                ox, oy, radius = obstacle
                center_px = self.world_to_screen(ox, oy)
                pygame.draw.circle(self.screen, self.OBSTACLE_COLOR, center_px, max(2, int(radius * self.px_per_meter)))
        else:
            point_layer = pygame.Surface((self.sim_rect.width, self.sim_rect.height), pygame.SRCALPHA)
            n = max(1, self._current_model_memory_len())
            for p in self.sparse_detection_points:
                age = int(p["age"])
                alpha = int(max(0.0, min(255.0, 255.0 * (1.0 - (age / n)))))
                if alpha <= 0:
                    continue
                sx, sy = self.world_to_screen(float(p["x"]), float(p["y"]))
                local_pt = (sx - self.sim_rect.left, sy - self.sim_rect.top)
                pygame.draw.circle(point_layer, (255, 255, 255, alpha), local_pt, 3)
            self.screen.blit(point_layer, self.sim_rect.topleft)

        if self.target is not None:
            tx, ty, tr = self.target
            center_px = self.world_to_screen(tx, ty)
            pygame.draw.circle(self.screen, self.TARGET_COLOR, center_px, max(2, int(tr * self.px_per_meter)))

        for p0, p1, hit in self.whisker_segments:
            c = self.WHISKER_HIT_COLOR if hit else self.WHISKER_FREE_COLOR
            pygame.draw.line(
                self.screen,
                c,
                self.world_to_screen(float(p0[0]), float(p0[1])),
                self.world_to_screen(float(p1[0]), float(p1[1])),
                1,
            )

        corners = self.robot.body_corners_world()
        poly_px = [self.world_to_screen(float(p[0]), float(p[1])) for p in corners]
        body_color = self.ROBOT_COLLIDE_COLOR if self.robot.collision_flash_timer > 0.0 else self.ROBOT_COLOR
        pygame.draw.polygon(self.screen, body_color, poly_px)

        for corner in corners:
            pygame.draw.circle(
                self.screen,
                self.WHEEL_COLOR,
                self.world_to_screen(float(corner[0]), float(corner[1])),
                4,
            )

        center = np.array([self.robot.x, self.robot.y], dtype=float)
        local_x, local_y = self.robot._basis_vectors()

        self._draw_arrow(center, local_x * Robot.AXIS_ARROW_LEN_M, (220, 40, 40), 2)
        self._draw_arrow(center, local_y * Robot.AXIS_ARROW_LEN_M, (30, 170, 60), 2)
        self._draw_goal_compass_arrow()

        if self.collision_display_timer > 0.0:
            overlay_rect = self.sim_rect.inflate(-40, -40)
            pygame.draw.rect(self.screen, (0, 0, 0), overlay_rect)
            collision_text = pygame.font.SysFont("consolas", 60).render("COLLISION", True, (255, 0, 0))
            text_rect = collision_text.get_rect(center=overlay_rect.center)
            self.screen.blit(collision_text, text_rect)

        if self.human_takeover_prompt_timer > 0.0:
            prompt_bg = self.sim_rect.inflate(-70, -200)
            pygame.draw.rect(self.screen, (0, 0, 0), prompt_bg)
            prompt_font = pygame.font.SysFont("consolas", 30)
            prompt_text = prompt_font.render("DAGGER: HUMAN TAKEOVER - NAVIGATE THIS MAP", True, (255, 220, 80))
            prompt_rect = prompt_text.get_rect(center=prompt_bg.center)
            self.screen.blit(prompt_text, prompt_rect)

        self.screen.set_clip(None)

    def _render_text(self, text: str, x: int, y: int, color: Tuple[int, int, int] = (25, 25, 25), small: bool = False) -> None:
        surf = (self.small_font if small else self.font).render(text, True, color)
        self.screen.blit(surf, (x, y))

    def _draw_radio(self, rect: pygame.Rect, selected: bool, label: str) -> None:
        center = (rect.left + rect.width // 2, rect.top + rect.height // 2)
        pygame.draw.circle(self.screen, (20, 20, 20), center, 8, 2)
        if selected:
            pygame.draw.circle(self.screen, (30, 150, 50), center, 4)
        self._render_text(label, rect.right + 8, rect.top - 2, small=True)

    def _draw_checkbox(self, rect: pygame.Rect, checked: bool, label: str) -> None:
        pygame.draw.rect(self.screen, (20, 20, 20), rect, 2)
        if checked:
            pygame.draw.line(self.screen, (30, 150, 50), (rect.left + 2, rect.top + 8), (rect.left + 6, rect.bottom - 2), 2)
            pygame.draw.line(self.screen, (30, 150, 50), (rect.left + 6, rect.bottom - 2), (rect.right - 2, rect.top + 2), 2)
        self._render_text(label, rect.right + 8, rect.top - 2, small=True)

    def _draw_top_bar(self) -> None:
        pygame.draw.rect(self.screen, self.PANEL_COLOR, self.toolbar_rect)
        pygame.draw.rect(self.screen, (180, 180, 185), self.toolbar_rect, 1)

        row1_y = self.toolbar_rect.top + 10
        row2_y = self.toolbar_rect.top + 56
        row3_y = self.toolbar_rect.top + 102

        pygame.draw.rect(self.screen, (75, 120, 210), self.run_button_rect, border_radius=6)
        self._render_text("Run", self.run_button_rect.left + 33, self.run_button_rect.top + 6, (255, 255, 255))
        fail_color = (180, 60, 60) if self.input_mode == "model" else (160, 160, 160)
        pygame.draw.rect(self.screen, fail_color, self.fail_button_rect, border_radius=6)
        self._render_text("FAIL", self.fail_button_rect.left + 31, self.fail_button_rect.top + 6, (255, 255, 255))

        self._render_text("Obstacles", self.obs_minus_rect.left - 92, row1_y + 4, small=True)
        pygame.draw.rect(self.screen, (230, 230, 234), self.obs_minus_rect, border_radius=3)
        pygame.draw.rect(self.screen, (230, 230, 234), self.obs_plus_rect, border_radius=3)
        pygame.draw.rect(self.screen, (160, 160, 165), self.obs_minus_rect, 1, border_radius=3)
        pygame.draw.rect(self.screen, (160, 160, 165), self.obs_plus_rect, 1, border_radius=3)
        self._render_text("-", self.obs_minus_rect.left + 8, self.obs_minus_rect.top + 1)
        self._render_text("+", self.obs_plus_rect.left + 7, self.obs_plus_rect.top + 1)
        self._render_text(f"{self.obstacle_count}", self.obs_minus_rect.right + 10, self.obs_minus_rect.top + 1)

        # Prefs button
        pygame.draw.rect(self.screen, (230, 230, 234), self.prefs_button_rect, border_radius=3)
        pygame.draw.rect(self.screen, (160, 160, 165), self.prefs_button_rect, 1, border_radius=3)
        self._render_text("Prefs", self.prefs_button_rect.left + 18, self.prefs_button_rect.top + 2, small=True)

        self._render_text("Control Mode", self.toolbar_rect.left + 12, row2_y + 4, small=True)
        self._draw_radio(self.radio_heading_rect, self.robot.control_mode == "heading_drive", "Heading+Drive")
        self._draw_radio(self.radio_xy_rect, self.robot.control_mode == "xy_strafe", "XY Strafe")

        self._draw_radio(self.radio_heading_strafe_rect, self.robot.control_mode == "heading_strafe", "Hdg+Strafe")

        self._render_text("Input Source", self.toolbar_rect.left + 12, row3_y + 4, small=True)
        self._draw_radio(self.radio_keyboard_rect, self.input_mode == "keyboard", "Keyboard")
        self._draw_radio(self.radio_gamepad_rect, self.input_mode == "gamepad", "Gamepad")
        self._draw_radio(self.radio_model_rect, self.input_mode == "model", "Model")

        self._render_text("Model", self.model_pick_rect.left - 64, row3_y + 5, small=True)

        pygame.draw.rect(self.screen, (230, 230, 234), self.model_pick_rect, border_radius=3)
        pygame.draw.rect(self.screen, (230, 230, 234), self.model_reload_rect, border_radius=3)
        pygame.draw.rect(self.screen, (160, 160, 165), self.model_pick_rect, 1, border_radius=3)
        pygame.draw.rect(self.screen, (160, 160, 165), self.model_reload_rect, 1, border_radius=3)
        self._render_text("Pick", self.model_pick_rect.left + 13, self.model_pick_rect.top + 1, small=True)
        self._render_text("Reload", self.model_reload_rect.left + 8, self.model_reload_rect.top + 1, small=True)

        self._draw_checkbox(self.checkbox_logging_rect, self.logging_enabled, "Enable Logging")
        self._draw_checkbox(self.checkbox_robot_view_rect, self.robot_aligned_view, "Robot View")

    def _draw_prefs_panel(self) -> None:
        # Dim overlay
        dim = pygame.Surface((self.window_w, self.window_h), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 120))
        self.screen.blit(dim, (0, 0))
        # Panel background
        pygame.draw.rect(self.screen, (245, 245, 248), self.prefs_rect, border_radius=8)
        pygame.draw.rect(self.screen, (140, 140, 150), self.prefs_rect, 2, border_radius=8)
        px, py = self.prefs_rect.left, self.prefs_rect.top
        lx = px + 24
        # Title
        title_surf = self.font.render("Preferences", True, (25, 25, 25))
        self.screen.blit(title_surf, (lx, py + 14))
        # Close button
        pygame.draw.rect(self.screen, (200, 80, 70), self.prefs_close_rect, border_radius=4)
        close_surf = self.small_font.render("X", True, (255, 255, 255))
        self.screen.blit(close_surf, (self.prefs_close_rect.left + 7, self.prefs_close_rect.top + 5))
        # ---- History spinner ----
        self._render_text("History (steps)", lx, self.prefs_hist_minus_rect.top + 2, small=True)
        pygame.draw.rect(self.screen, (230, 230, 234), self.prefs_hist_minus_rect, border_radius=3)
        pygame.draw.rect(self.screen, (230, 230, 234), self.prefs_hist_plus_rect, border_radius=3)
        pygame.draw.rect(self.screen, (160, 160, 165), self.prefs_hist_minus_rect, 1, border_radius=3)
        pygame.draw.rect(self.screen, (160, 160, 165), self.prefs_hist_plus_rect, 1, border_radius=3)
        self._render_text("-", self.prefs_hist_minus_rect.left + 8, self.prefs_hist_minus_rect.top + 1)
        self._render_text("+", self.prefs_hist_plus_rect.left + 7, self.prefs_hist_plus_rect.top + 1)
        self._render_text(f"{self.history_len}", self.prefs_hist_minus_rect.right + 10, self.prefs_hist_minus_rect.top + 1)
        # ---- Rate Hz spinner ----
        self._render_text("Rate Hz", lx, self.prefs_rate_minus_rect.top + 2, small=True)
        pygame.draw.rect(self.screen, (230, 230, 234), self.prefs_rate_minus_rect, border_radius=3)
        pygame.draw.rect(self.screen, (230, 230, 234), self.prefs_rate_plus_rect, border_radius=3)
        pygame.draw.rect(self.screen, (160, 160, 165), self.prefs_rate_minus_rect, 1, border_radius=3)
        pygame.draw.rect(self.screen, (160, 160, 165), self.prefs_rate_plus_rect, 1, border_radius=3)
        self._render_text("-", self.prefs_rate_minus_rect.left + 8, self.prefs_rate_minus_rect.top + 1)
        self._render_text("+", self.prefs_rate_plus_rect.left + 7, self.prefs_rate_plus_rect.top + 1)
        self._render_text(f"{self.active_log_rate_hz:.0f}", self.prefs_rate_minus_rect.right + 8, self.prefs_rate_minus_rect.top + 1)
        # ---- Visualization section ----
        vis_label_y = self.prefs_vis_all_rect.top - 35
        self._render_text("Visualization Mode", lx, vis_label_y, small=True)
        pygame.draw.line(self.screen, (180, 180, 185), (lx, vis_label_y + 20), (self.prefs_rect.right - 24, vis_label_y + 20), 1)
        self._draw_radio(self.prefs_vis_all_rect, self.vis_mode == "all", "Show all objects")
        self._draw_radio(self.prefs_vis_radius_rect, self.vis_mode == "action_radius", "Show within action radius")
        self._draw_radio(self.prefs_vis_detect_rect, self.vis_mode == "detected", "Show when detected (memory)")
        self._draw_radio(self.prefs_vis_sparse_rect, self.vis_mode == "sparse_sensing", "Sparse Sensing")

    def _draw_panel(self) -> None:
        pygame.draw.rect(self.screen, self.PANEL_COLOR, self.panel_rect)
        pygame.draw.rect(self.screen, (180, 180, 185), self.panel_rect, 1)

        model_color = (30, 140, 60) if self.loaded_model is not None else (180, 70, 40)
        self._render_text("Model Status", self.panel_rect.left + 22, self.panel_rect.top + 16, small=True)
        self._render_text(self.model_status[:36], self.panel_rect.left + 22, self.panel_rect.top + 34, model_color, small=True)
        next_seq = str(self.train_me_queue[0][0]) if self.train_me_queue else "-"
        last_seq = str(self.last_train_me_loaded_seq) if self.last_train_me_loaded_seq is not None else "-"
        self._render_text(f"Q:{len(self.train_me_queue)} N:{next_seq} L:{last_seq}", self.panel_rect.left + 22, self.panel_rect.top + 54, small=True)

        self._render_text("Drive Command", self.panel_rect.left + 22, self.panel_rect.top + 88, small=True)
        if self.robot.control_mode == "heading_drive":
            self._render_text(
                f"rotation_rate: {self.robot.drive_command['rotation_rate']:+.1f} deg/s",
                self.panel_rect.left + 22,
                self.panel_rect.top + 110,
                small=True,
            )
            self._render_text(
                f"drive_speed:   {self.robot.drive_command['drive_speed']:+.2f} m/s",
                self.panel_rect.left + 22,
                self.panel_rect.top + 132,
                small=True,
            )
        elif self.robot.control_mode == "heading_strafe":
            self._render_text(
                f"rot: {self.robot.drive_command['rotation_rate']:+.1f} deg/s",
                self.panel_rect.left + 22,
                self.panel_rect.top + 110,
                small=True,
            )
            self._render_text(
                f"vx:{self.robot.drive_command['vx']:+.2f} vy:{self.robot.drive_command['vy']:+.2f} m/s",
                self.panel_rect.left + 22,
                self.panel_rect.top + 132,
                small=True,
            )
        else:
            self._render_text(
                f"vx: {self.robot.drive_command['vx']:+.2f} m/s",
                self.panel_rect.left + 22,
                self.panel_rect.top + 110,
                small=True,
            )
            self._render_text(
                f"vy: {self.robot.drive_command['vy']:+.2f} m/s",
                self.panel_rect.left + 22,
                self.panel_rect.top + 132,
                small=True,
            )

        self._render_text("Whisker Lengths (m)", self.panel_rect.left + 22, self.panel_rect.top + 164, small=True)
        y = self.panel_rect.top + 186
        for i, length in enumerate(self.robot.whisker_lengths):
            angle = Robot.WHISKER_ANGLES_DEG[i]
            self._render_text(
                f"{angle:>4} deg: {length:.3f}",
                self.panel_rect.left + 22,
                y,
                small=True,
            )
            y += 20

        self._render_text(
            f"Heading to Target: {self.robot.heading_to_target_deg:+.1f}°",
            self.panel_rect.left + 22,
            self.panel_rect.bottom - 100,
            small=True,
        )

        coll_text = "YES" if self.robot.collision_flag else "NO"
        coll_color = (220, 40, 40) if self.robot.collision_flag else (30, 140, 60)
        self._render_text("Collision", self.panel_rect.left + 22, self.panel_rect.bottom - 68, small=True)
        self._render_text(coll_text, self.panel_rect.left + 22, self.panel_rect.bottom - 45, coll_color)

        self._render_text(
            f"Pose x={self.robot.robot_pose['x']:.2f}, y={self.robot.robot_pose['y']:.2f}, h={self.robot.robot_pose['heading']:.1f}",
            self.panel_rect.left + 22,
            self.panel_rect.bottom - 24,
            small=True,
        )

    def run(self) -> None:
        running = True

        while running:
            dt = self.clock.tick(self.FPS) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.window_w = max(self.min_window_w, event.w)
                    self.window_h = max(self.min_window_h, event.h)
                    self.screen = pygame.display.set_mode((self.window_w, self.window_h), pygame.RESIZABLE)
                    self._recompute_layout()
                elif event.type == pygame.MOUSEMOTION:
                    self.mouse_pos = event.pos
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            self.update_robot_command()
            self.whisker_segments = self.robot.step(dt, self.obstacles)
            self.robot.update_heading_to_target(self.target)
            self._update_sparse_detection_points()
            self._update_obstacle_detection(dt)
            
            # Logging: dynamic cadence by context
            self.log_timer += dt
            log_interval = self._current_log_interval()
            if log_interval is not None and self.log_timer >= log_interval:
                self.in_memory_log.append(self._collect_log_entry())
                self.log_timer = 0.0
            
            if self.check_goal_reached():
                # Goal reached: write log if present
                if self.logging_enabled and self.in_memory_log:
                    self._write_log_to_file()
                self.in_memory_log = []
                if self.active_train_me_case_seq is not None and self.train_me_queue:
                    self._run_next_train_me_case()
                else:
                    self.new_episode()
            elif self.robot.collision_flag or self.check_whisker_collision():
                print("COLLISION")
                self.sparse_detection_points.clear()
                if self.input_mode == "model":
                    self._enqueue_train_me_case("collision")
                # Collision: clear log without writing to disk
                self.in_memory_log = []
                self.collision_display_timer = 0.5
                self.new_episode()
            
            if self.collision_display_timer > 0.0:
                self.collision_display_timer = max(0.0, self.collision_display_timer - dt)
            if self.human_takeover_prompt_timer > 0.0:
                self.human_takeover_prompt_timer = max(0.0, self.human_takeover_prompt_timer - dt)

            self.screen.fill(self.BG_COLOR)
            self._draw_top_bar()
            self._draw_simulation()
            self._draw_panel()
            if self.prefs_open:
                self._draw_prefs_panel()
            tip = self._tooltip_for_pos(self.mouse_pos)
            if tip is not None:
                self._draw_tooltip(self.mouse_pos, tip)

            pygame.display.flip()

        pygame.quit()


def main() -> None:
    app = SimulatorApp()
    app.run()


if __name__ == "__main__":
    main()
