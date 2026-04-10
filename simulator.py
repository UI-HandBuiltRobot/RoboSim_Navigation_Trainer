import json
import math
import random
import shutil
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

Vec2 = np.ndarray


class Robot:
    ROOM_WIDTH_M = 5.0
    ROOM_HEIGHT_M = 3.0

    BODY_WIDTH_M = 0.20
    BODY_LENGTH_M = 0.25
    AXIS_ARROW_LEN_M = 0.10

    WHISKER_ANGLES_DEG = [-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90]
    WHISKER_MAX_LEN_M = 0.50

    KEYBOARD_DRIVE_SPEED_MPS = 0.80
    KEYBOARD_ROTATE_RATE_DPS = 60.0
    GAMEPAD_MAX_DRIVE_SPEED_MPS = 0.40
    GAMEPAD_MAX_ROTATE_RATE_DPS = 40.0

    CLEARANCE_FROM_WALLS_M = 0.30

    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.heading_deg = 0.0

        self.control_mode = "heading_drive"
        self.drive_command: Dict[str, float] = {"rotation_rate": 0.0, "drive_speed": 0.0}

        self.whisker_lengths: List[float] = [self.WHISKER_MAX_LEN_M for _ in self.WHISKER_ANGLES_DEG]
        self.robot_pose: Dict[str, float] = {"x": 0.0, "y": 0.0, "heading": 0.0}
        self.heading_to_target_deg = 0.0
        self.collision_flag = False

        self.collision_flash_timer = 0.0

        self.collision_radius = 0.5 * math.hypot(self.BODY_WIDTH_M, self.BODY_LENGTH_M)

        self.reset_random_pose()

    @staticmethod
    def _deg_to_rad(deg: float) -> float:
        return math.radians(deg)

    def _basis_vectors(self) -> Tuple[Vec2, Vec2]:
        theta = self._deg_to_rad(self.heading_deg)
        local_x = np.array([math.sin(theta), math.cos(theta)], dtype=float)
        local_y = np.array([-math.cos(theta), math.sin(theta)], dtype=float)
        return local_x, local_y

    def forward_vector(self) -> Vec2:
        local_x, _ = self._basis_vectors()
        return local_x

    @staticmethod
    def _apply_deadband(value: float, deadband: float = 0.05) -> float:
        if abs(value) < deadband:
            return 0.0
        if value > 0:
            return (value - deadband) / (1.0 - deadband)
        else:
            return (value + deadband) / (1.0 - deadband)

    @staticmethod
    def _shape_gamepad_axis(value: float) -> float:
        return value * value * value

    def reset_random_pose(self) -> None:
        self.x = random.uniform(self.CLEARANCE_FROM_WALLS_M, self.ROOM_WIDTH_M - self.CLEARANCE_FROM_WALLS_M)
        self.y = random.uniform(self.CLEARANCE_FROM_WALLS_M, self.ROOM_HEIGHT_M - self.CLEARANCE_FROM_WALLS_M)

        if self.control_mode in ("heading_drive", "heading_strafe"):
            self.heading_deg = random.uniform(-180.0, 180.0)
        else:
            to_center = np.array(
                [
                    0.5 * self.ROOM_WIDTH_M - self.x,
                    0.5 * self.ROOM_HEIGHT_M - self.y,
                ],
                dtype=float,
            )
            base_heading = math.degrees(math.atan2(to_center[0], to_center[1]))

            for _ in range(200):
                candidate = base_heading + random.uniform(-60.0, 60.0)
                self.heading_deg = candidate
                fwd = self.forward_vector()
                margin = 0.20
                projected = np.array([self.x, self.y], dtype=float) + fwd * margin
                if 0.0 < projected[0] < self.ROOM_WIDTH_M and 0.0 < projected[1] < self.ROOM_HEIGHT_M:
                    break

        if self.control_mode == "heading_drive":
            self.drive_command = {"rotation_rate": 0.0, "drive_speed": 0.0}
        elif self.control_mode == "heading_strafe":
            self.drive_command = {"rotation_rate": 0.0, "vx": 0.0, "vy": 0.0}
        else:
            self.drive_command = {"vx": 0.0, "vy": 0.0}
        self.whisker_lengths = [self.WHISKER_MAX_LEN_M for _ in self.WHISKER_ANGLES_DEG]
        self.robot_pose = {"x": self.x, "y": self.y, "heading": self.heading_deg}
        self.collision_flag = False
        self.collision_flash_timer = 0.0

    def compute_heading_to_target(self, target_x: float, target_y: float) -> float:
        dx = target_x - self.x
        dy = target_y - self.y
        world_angle_to_target_deg = math.degrees(math.atan2(dx, dy))
        heading_to_target = world_angle_to_target_deg - self.heading_deg
        heading_to_target = ((heading_to_target + 180.0) % 360.0) - 180.0
        return heading_to_target

    def set_control_mode(self, mode: str) -> None:
        if mode not in ("heading_drive", "xy_strafe", "heading_strafe"):
            return
        self.control_mode = mode
        if mode == "heading_drive":
            self.drive_command = {"rotation_rate": 0.0, "drive_speed": 0.0}
        elif mode == "heading_strafe":
            self.drive_command = {"rotation_rate": 0.0, "vx": 0.0, "vy": 0.0}
        else:
            self.drive_command = {"vx": 0.0, "vy": 0.0}

    def update_command_from_keys(self, keys: pygame.key.ScancodeWrapper) -> None:
        if self.control_mode == "heading_drive":
            drive_speed = 0.0
            rotation_rate = 0.0
            if keys[pygame.K_UP]:
                drive_speed += self.KEYBOARD_DRIVE_SPEED_MPS
            if keys[pygame.K_DOWN]:
                drive_speed -= self.KEYBOARD_DRIVE_SPEED_MPS
            if keys[pygame.K_LEFT]:
                rotation_rate -= self.KEYBOARD_ROTATE_RATE_DPS
            if keys[pygame.K_RIGHT]:
                rotation_rate += self.KEYBOARD_ROTATE_RATE_DPS
            self.drive_command = {
                "rotation_rate": rotation_rate,
                "drive_speed": drive_speed,
            }
        elif self.control_mode == "heading_strafe":
            rotation_rate = 0.0
            vx = 0.0
            vy = 0.0
            if keys[pygame.K_UP]:
                vx += self.KEYBOARD_DRIVE_SPEED_MPS
            if keys[pygame.K_DOWN]:
                vx -= self.KEYBOARD_DRIVE_SPEED_MPS
            if keys[pygame.K_LEFT]:
                rotation_rate -= self.KEYBOARD_ROTATE_RATE_DPS
            if keys[pygame.K_RIGHT]:
                rotation_rate += self.KEYBOARD_ROTATE_RATE_DPS
            if keys[pygame.K_a]:
                vy += self.KEYBOARD_DRIVE_SPEED_MPS
            if keys[pygame.K_d]:
                vy -= self.KEYBOARD_DRIVE_SPEED_MPS
            self.drive_command = {"rotation_rate": rotation_rate, "vx": vx, "vy": vy}
        else:
            vx = 0.0
            vy = 0.0
            if keys[pygame.K_LEFT]:
                vy += self.KEYBOARD_DRIVE_SPEED_MPS
            if keys[pygame.K_RIGHT]:
                vy -= self.KEYBOARD_DRIVE_SPEED_MPS
            if keys[pygame.K_UP]:
                vx += self.KEYBOARD_DRIVE_SPEED_MPS
            if keys[pygame.K_DOWN]:
                vx -= self.KEYBOARD_DRIVE_SPEED_MPS
            self.drive_command = {"vx": vx, "vy": vy}

    def update_command_from_gamepad(self, joystick: pygame.joystick.Joystick) -> None:
        if joystick is None:
            return

        axis_count = joystick.get_numaxes()

        if self.control_mode == "heading_drive":
            drive_speed = 0.0
            rotation_rate = 0.0

            if axis_count > 1:
                y_axis = self._shape_gamepad_axis(self._apply_deadband(joystick.get_axis(1)))
                drive_speed = -y_axis * self.GAMEPAD_MAX_DRIVE_SPEED_MPS

            if axis_count > 2:
                z_axis = self._shape_gamepad_axis(self._apply_deadband(joystick.get_axis(2)))
                rotation_rate = z_axis * self.GAMEPAD_MAX_ROTATE_RATE_DPS

            self.drive_command = {
                "rotation_rate": rotation_rate,
                "drive_speed": drive_speed,
            }
        elif self.control_mode == "heading_strafe":
            rotation_rate = 0.0
            vx = 0.0
            vy = 0.0

            if axis_count > 0:
                x_axis = self._shape_gamepad_axis(self._apply_deadband(joystick.get_axis(0)))
                vy = -x_axis * self.GAMEPAD_MAX_DRIVE_SPEED_MPS

            if axis_count > 1:
                y_axis = self._shape_gamepad_axis(self._apply_deadband(joystick.get_axis(1)))
                vx = -y_axis * self.GAMEPAD_MAX_DRIVE_SPEED_MPS

            if axis_count > 2:
                z_axis = self._shape_gamepad_axis(self._apply_deadband(joystick.get_axis(2)))
                rotation_rate = z_axis * self.GAMEPAD_MAX_ROTATE_RATE_DPS

            self.drive_command = {"rotation_rate": rotation_rate, "vx": vx, "vy": vy}
        else:
            vx = 0.0
            vy = 0.0

            if axis_count > 2:
                z_axis = self._shape_gamepad_axis(self._apply_deadband(joystick.get_axis(2)))
                vy = z_axis * self.GAMEPAD_MAX_DRIVE_SPEED_MPS

            if axis_count > 1:
                y_axis = self._shape_gamepad_axis(self._apply_deadband(joystick.get_axis(1)))
                vx = -y_axis * self.GAMEPAD_MAX_DRIVE_SPEED_MPS

            self.drive_command = {"vx": vx, "vy": vy}

    def integrate(self, dt: float) -> None:
        local_x, local_y = self._basis_vectors()

        if self.control_mode == "heading_drive":
            rot_dps = self.drive_command.get("rotation_rate", 0.0)
            drive_speed = self.drive_command.get("drive_speed", 0.0)
            self.heading_deg += rot_dps * dt
            self.heading_deg = ((self.heading_deg + 180.0) % 360.0) - 180.0
            displacement = local_x * drive_speed * dt
        elif self.control_mode == "heading_strafe":
            rot_dps = self.drive_command.get("rotation_rate", 0.0)
            vx = self.drive_command.get("vx", 0.0)
            vy = self.drive_command.get("vy", 0.0)
            self.heading_deg += rot_dps * dt
            self.heading_deg = ((self.heading_deg + 180.0) % 360.0) - 180.0
            local_x, local_y = self._basis_vectors()
            displacement = (local_x * vx + local_y * vy) * dt
        else:
            vx = self.drive_command.get("vx", 0.0)
            vy = self.drive_command.get("vy", 0.0)
            displacement = (local_x * vx + local_y * vy) * dt

        self.x += float(displacement[0])
        self.y += float(displacement[1])

    @staticmethod
    def _ray_circle_intersection(origin: Vec2, direction: Vec2, center: Vec2, radius: float) -> Optional[float]:
        oc = origin - center
        a = float(np.dot(direction, direction))
        b = 2.0 * float(np.dot(direction, oc))
        c = float(np.dot(oc, oc) - radius * radius)

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None

        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        ts = [t for t in (t1, t2) if t >= 0.0]
        if not ts:
            return None
        return min(ts)

    @staticmethod
    def _ray_segment_intersection(origin: Vec2, direction: Vec2, p1: Vec2, p2: Vec2) -> Optional[float]:
        v1 = origin - p1
        v2 = p2 - p1
        denom = direction[0] * v2[1] - direction[1] * v2[0]

        if abs(denom) < 1e-10:
            return None

        t = (v2[0] * v1[1] - v2[1] * v1[0]) / denom
        u = (direction[0] * v1[1] - direction[1] * v1[0]) / denom

        if t >= 0.0 and 0.0 <= u <= 1.0:
            return t
        return None

    def compute_whiskers(self, obstacles: List[Tuple[float, float, float]]) -> List[Tuple[Vec2, Vec2, bool]]:
        local_x, local_y = self._basis_vectors()
        origin = np.array([self.x, self.y], dtype=float) + local_x * 0.12
        whisker_segments: List[Tuple[Vec2, Vec2, bool]] = []
        lengths: List[float] = []

        wall_segments = [
            (np.array([0.0, 0.0], dtype=float), np.array([self.ROOM_WIDTH_M, 0.0], dtype=float)),
            (np.array([self.ROOM_WIDTH_M, 0.0], dtype=float), np.array([self.ROOM_WIDTH_M, self.ROOM_HEIGHT_M], dtype=float)),
            (np.array([self.ROOM_WIDTH_M, self.ROOM_HEIGHT_M], dtype=float), np.array([0.0, self.ROOM_HEIGHT_M], dtype=float)),
            (np.array([0.0, self.ROOM_HEIGHT_M], dtype=float), np.array([0.0, 0.0], dtype=float)),
        ]

        for angle_deg in self.WHISKER_ANGLES_DEG:
            world_deg = self.heading_deg + angle_deg
            theta = self._deg_to_rad(world_deg)
            direction = np.array([math.sin(theta), math.cos(theta)], dtype=float)

            min_dist = self.WHISKER_MAX_LEN_M
            hit = False

            for ox, oy, radius in obstacles:
                center = np.array([ox, oy], dtype=float)
                d = self._ray_circle_intersection(origin, direction, center, radius)
                if d is not None and 0.0 <= d < min_dist:
                    min_dist = d
                    hit = True

            for p1, p2 in wall_segments:
                d = self._ray_segment_intersection(origin, direction, p1, p2)
                if d is not None and 0.0 <= d < min_dist:
                    min_dist = d
                    hit = True

            endpoint = origin + direction * min_dist
            whisker_segments.append((origin.copy(), endpoint, hit))
            lengths.append(float(min_dist))

        self.whisker_lengths = lengths
        return whisker_segments

    def check_collision(self, obstacles: List[Tuple[float, float, float]]) -> bool:
        collided = False

        if self.x - self.collision_radius < 0.0:
            collided = True
        if self.x + self.collision_radius > self.ROOM_WIDTH_M:
            collided = True
        if self.y - self.collision_radius < 0.0:
            collided = True
        if self.y + self.collision_radius > self.ROOM_HEIGHT_M:
            collided = True

        for ox, oy, radius in obstacles:
            dx = self.x - ox
            dy = self.y - oy
            if dx * dx + dy * dy <= (self.collision_radius + radius) ** 2:
                collided = True
                break

        if collided:
            self.collision_flag = True
            self.collision_flash_timer = 0.20
            if self.control_mode == "heading_drive":
                self.drive_command = {"rotation_rate": 0.0, "drive_speed": 0.0}
            elif self.control_mode == "heading_strafe":
                self.drive_command = {"rotation_rate": 0.0, "vx": 0.0, "vy": 0.0}
            else:
                self.drive_command = {"vx": 0.0, "vy": 0.0}

        return collided

    def step(self, dt: float, obstacles: List[Tuple[float, float, float]]) -> List[Tuple[Vec2, Vec2, bool]]:
        prev_x, prev_y, prev_h = self.x, self.y, self.heading_deg

        self.integrate(dt)

        if self.check_collision(obstacles):
            self.x, self.y, self.heading_deg = prev_x, prev_y, prev_h

        whiskers = self.compute_whiskers(obstacles)

        if self.collision_flash_timer > 0.0:
            self.collision_flash_timer = max(0.0, self.collision_flash_timer - dt)

        self.robot_pose = {"x": self.x, "y": self.y, "heading": self.heading_deg}

        return whiskers

    def update_heading_to_target(self, target: Optional[Tuple[float, float, float]]) -> None:
        if target is None:
            self.heading_to_target_deg = 0.0
        else:
            tx, ty, _ = target
            self.heading_to_target_deg = self.compute_heading_to_target(tx, ty)

    def body_corners_world(self) -> List[Vec2]:
        local_x, local_y = self._basis_vectors()
        center = np.array([self.x, self.y], dtype=float)
        hx = 0.5 * self.BODY_LENGTH_M
        hy = 0.5 * self.BODY_WIDTH_M

        return [
            center + (+hx * local_x) + (+hy * local_y),
            center + (-hx * local_x) + (+hy * local_y),
            center + (-hx * local_x) + (-hy * local_y),
            center + (+hx * local_x) + (-hy * local_y),
        ]


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
    GOAL_REACH_RADIUS_M = 0.25

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
        self.model_dir.mkdir(exist_ok=True)
        self.loaded_model: Optional[Dict[str, np.ndarray]] = None
        self.loaded_model_mode: Optional[str] = None
        self.selected_model_paths: Dict[str, Path] = {}
        self.model_status = "No model loaded (train first)"

        self.snapshot_dir = Path.cwd() / "dagger_snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)
        self.last_snapshot_path: Optional[Path] = None
        self.current_model_snapshot_path: Optional[Path] = None
        self.train_me_queue: List[Tuple[int, Path]] = []
        self.train_me_enqueue_seq = 1
        self.last_train_me_loaded_seq: Optional[int] = None
        self.active_train_me_case_seq: Optional[int] = None
        self.human_takeover_prompt_timer = 0.0

        # Dialog backend: only external tools (zenity/kdialog) or none
        self.dialog_backend = "none"
        if shutil.which("zenity") or shutil.which("kdialog"):
            self.dialog_backend = "external"

        self.model_pick_thread: Optional[threading.Thread] = None
        self.model_pick_result: Optional[str] = None
        self.model_pick_result_ready = False

        self.trainme_prompt_thread = None
        self.trainme_prompt_result = None
        self.trainme_prompt_ready = False

        self._recompute_layout()

        self.new_episode()

    def _ask_open_filename_external(self, title: str, initialdir: Path) -> Optional[str]:
        initial = str(initialdir)

        zenity = shutil.which("zenity")
        if zenity:
            try:
                proc = subprocess.run(
                    [
                        zenity,
                        "--file-selection",
                        f"--title={title}",
                        f"--filename={initial.rstrip('/')}/",
                        "--file-filter=Model JSON | *.json",
                        "--file-filter=All files | *",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if proc.returncode == 0:
                    picked = proc.stdout.strip()
                    return picked if picked else None
                if proc.returncode == 1:
                    return ""
            except Exception:
                pass

        kdialog = shutil.which("kdialog")
        if kdialog:
            try:
                proc = subprocess.run(
                    [kdialog, "--getopenfilename", initial, "*.json|Model JSON"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if proc.returncode == 0:
                    picked = proc.stdout.strip()
                    return picked if picked else None
                if proc.returncode == 1:
                    return ""
            except Exception:
                pass

        return None

    def _ask_yes_no_external(self, title: str, prompt: str) -> Optional[bool]:
        zenity = shutil.which("zenity")
        if zenity:
            try:
                proc = subprocess.run(
                    [zenity, "--question", f"--title={title}", f"--text={prompt}"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if proc.returncode == 0:
                    return True
                if proc.returncode == 1:
                    return False
            except Exception:
                pass

        kdialog = shutil.which("kdialog")
        if kdialog:
            try:
                proc = subprocess.run(
                    [kdialog, "--yesno", prompt, "--title", title],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if proc.returncode == 0:
                    return True
                if proc.returncode == 1:
                    return False
            except Exception:
                pass

        return None

    def _ask_open_filename(self, title: str, initialdir: Path) -> Optional[str]:
        if self.dialog_backend == "external":
            return self._ask_open_filename_external(title, initialdir)
        return None

    def _ask_yes_no(self, title: str, prompt: str) -> Optional[bool]:
        if self.dialog_backend == "external":
            return self._ask_yes_no_external(title, prompt)
        return None

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

        spinner_base = max(self.fail_button_rect.right + 50, tx + int(self.toolbar_rect.width * 0.34))
        spinner_band = self.toolbar_rect.right - spinner_base - 110
        obs_x = spinner_base + int(spinner_band * 0.00)
        hist_x = spinner_base + int(spinner_band * 0.45)
        rate_x = spinner_base + int(spinner_band * 0.90)

        self.obs_minus_rect = pygame.Rect(obs_x, row1_y + 6, 24, 24)
        self.obs_plus_rect = pygame.Rect(obs_x + 70, row1_y + 6, 24, 24)

        self.hist_minus_rect = pygame.Rect(hist_x, row1_y + 6, 24, 24)
        self.hist_plus_rect = pygame.Rect(hist_x + 70, row1_y + 6, 24, 24)

        self.rate_minus_rect = pygame.Rect(rate_x, row1_y + 6, 24, 24)
        self.rate_plus_rect = pygame.Rect(rate_x + 70, row1_y + 6, 24, 24)

        self.radio_keyboard_rect = pygame.Rect(tx + 120, row3_y, 18, 18)
        self.radio_gamepad_rect = pygame.Rect(tx + 245, row3_y, 18, 18)
        self.radio_model_rect = pygame.Rect(tx + 360, row3_y, 18, 18)

        model_center_x = self.toolbar_rect.centerx
        self.model_pick_rect = pygame.Rect(model_center_x - 36, row3_y + 7, 68, 22)
        self.model_reload_rect = pygame.Rect(model_center_x + 42, row3_y + 7, 72, 22)

        self.checkbox_logging_rect = pygame.Rect(self.toolbar_rect.right - 330, row3_y + 8, 18, 18)
        self.checkbox_robot_view_rect = pygame.Rect(self.toolbar_rect.right - 150, row3_y + 8, 18, 18)

        self.radio_heading_rect = pygame.Rect(tx + 120, row2_y, 18, 18)
        self.radio_xy_rect = pygame.Rect(tx + 280, row2_y, 18, 18)

        self.radio_heading_strafe_rect = pygame.Rect(tx + 415, row2_y, 18, 18)

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
        obs_field_rect = self.obs_minus_rect.union(self.obs_plus_rect).inflate(120, 10)
        hist_field_rect = self.hist_minus_rect.union(self.hist_plus_rect).inflate(120, 10)
        rate_field_rect = self.rate_minus_rect.union(self.rate_plus_rect).inflate(120, 10)

        tips: List[Tuple[pygame.Rect, str]] = [
            (self.run_button_rect, "Run: start a new map or load next queued train-me case"),
            (self.fail_button_rect, "FAIL: in Model mode, queue this case for human takeover training"),
            (obs_field_rect, "Obstacles: number of obstacles per episode; higher values create harder training data"),
            (self.obs_minus_rect, "Obstacles -: reduce obstacle count"),
            (self.obs_plus_rect, "Obstacles +: increase obstacle count"),
            (hist_field_rect, "History: logged context window length; larger values let the trainer learn temporal behavior"),
            (self.hist_minus_rect, "History -: shorten logging history window"),
            (self.hist_plus_rect, "History +: lengthen logging history window"),
            (rate_field_rect, "Rate Hz: active logging frequency; higher rates capture denser trajectories"),
            (self.rate_minus_rect, "Rate Hz -: decrease active logging frequency"),
            (self.rate_plus_rect, "Rate Hz +: increase active logging frequency"),
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
        obstacles: List[Tuple[float, float, float]] = []
        radius = 0.05

        robot_center = np.array([self.robot.x, self.robot.y], dtype=float)
        robot_edges = self._robot_edge_segments()

        for _ in range(2000):
            if len(obstacles) >= n_obs:
                break

            ox = random.uniform(radius, Robot.ROOM_WIDTH_M - radius)
            oy = random.uniform(radius, Robot.ROOM_HEIGHT_M - radius)
            center = np.array([ox, oy], dtype=float)

            valid = True

            for ex, ey, er in obstacles:
                d = math.hypot(ox - ex, oy - ey)
                if d < (radius + er):
                    valid = False
                    break
            if not valid:
                continue

            center_dist_to_robot = float(np.linalg.norm(center - robot_center))
            if center_dist_to_robot < self.robot.collision_radius + radius + 0.20:
                valid = False
            else:
                min_seg_dist = min(self._point_to_segment_distance(center, a, b) for a, b in robot_edges)
                if min_seg_dist < radius + 0.20:
                    valid = False

            if valid:
                obstacles.append((ox, oy, radius))

        return obstacles

    def _spawn_target(self, obstacles: List[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
        target_r = 0.075

        for _ in range(3000):
            tx = random.uniform(0.30 + target_r, Robot.ROOM_WIDTH_M - 0.30 - target_r)
            ty = random.uniform(0.30 + target_r, Robot.ROOM_HEIGHT_M - 0.30 - target_r)

            ok = True
            for ox, oy, orad in obstacles:
                d = math.hypot(tx - ox, ty - oy)
                if d < target_r + orad + 0.30:
                    ok = False
                    break
            if not ok:
                continue

            dr = math.hypot(tx - self.robot.x, ty - self.robot.y)
            if dr < 0.50 + target_r:
                continue

            return tx, ty, target_r

        return None

    def new_episode(self) -> None:
        self.active_train_me_case_seq = None
        self.robot.reset_random_pose()
        self.collision_display_timer = 0.0
        self.in_memory_log = []
        self.log_timer = 0.0
        self.history_buffer.clear()
        n_obs = max(1, int(self.obstacle_count))

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
        # If control mode changed, restart episode
        if self.robot.control_mode != self.prev_control_mode:
            self.new_episode()
            return

        # --- CRITICAL: BLOCK MODEL DURING TAKEOVER ---
        if self.active_train_me_case_seq is not None:
            keys = pygame.key.get_pressed()
            self.robot.update_command_from_keys(keys)
            return
        # ------------------------------------------------

        # Normal behavior
        if self.input_mode == "keyboard":
            keys = pygame.key.get_pressed()
            self.robot.update_command_from_keys(keys)

        elif self.input_mode == "gamepad":
            self.robot.update_command_from_gamepad(self.joystick)

        else:  # MODEL MODE
            if self.loaded_model_mode != self.robot.control_mode:
                self._load_latest_model_for_mode(self.robot.control_mode)
            self._update_command_from_model()


    def check_goal_reached(self) -> bool:
        if self.target is None:
            return False
        tx, ty, _ = self.target
        dist = math.hypot(self.robot.x - tx, self.robot.y - ty)
        return dist <= self.GOAL_REACH_RADIUS_M

    def check_whisker_collision(self) -> bool:
        WHISKER_COLLISION_THRESHOLD_M = 0.10
        return any(length < WHISKER_COLLISION_THRESHOLD_M for length in self.robot.whisker_lengths)

    def _has_nonzero_input(self) -> bool:
        cmd = self.robot.drive_command
        if self.robot.control_mode == "heading_drive":
            return cmd.get("rotation_rate", 0.0) != 0.0 or cmd.get("drive_speed", 0.0) != 0.0
        elif self.robot.control_mode == "heading_strafe":
            return (
                cmd.get("rotation_rate", 0.0) != 0.0
                or cmd.get("vx", 0.0) != 0.0
                or cmd.get("vy", 0.0) != 0.0
            )
        else:
            return cmd.get("vx", 0.0) != 0.0 or cmd.get("vy", 0.0) != 0.0

    def _obstacle_within_action_radius(self) -> bool:
        DETECTION_RADIUS_M = 0.50
        for ox, oy, or_ in self.obstacles:
            dist = math.hypot(self.robot.x - ox, self.robot.y - oy)
            if dist - or_ <= DETECTION_RADIUS_M:
                return True
        return False

    def _is_obstacle_in_action_radius(self, obstacle: Tuple[float, float, float]) -> bool:
        ox, oy, or_ = obstacle
        detection_radius_m = 0.50
        dist = math.hypot(self.robot.x - ox, self.robot.y - oy)
        return (dist - or_) <= detection_radius_m

    def _should_log(self) -> bool:
        return self._current_log_interval() is not None

    def _current_log_interval(self) -> Optional[float]:
        if not self.logging_enabled:
            return None

        obstacle_near = self._obstacle_within_action_radius()
        if obstacle_near and self._has_nonzero_input():
            return 1.0 / max(0.1, self.active_log_rate_hz)
        if not obstacle_near:
            return 10.0 / max(0.1, self.active_log_rate_hz)
        return None

    def _collect_log_entry(self) -> Dict:
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
            self.model_status = f"No {suffix} model (run train_mlp.py)"
            return False

        model_path = model_files[-1]
        return self._load_model_from_file(model_path, mode)

    def _load_model_from_file(self, model_path: Path, expected_mode: Optional[str]) -> bool:
        try:
            if not model_path.exists():
                self.loaded_model = None
                self.loaded_model_mode = None
                self.model_status = f"Model not found: {model_path.name}"
                return False

            with open(model_path, "r", encoding="utf-8") as f:
                blob = json.load(f)

            if not isinstance(blob, dict):
                self.loaded_model = None
                self.loaded_model_mode = None
                self.model_status = "Invalid model file: root must be object"
                return False

            required_keys = {"weights", "biases", "x_mean", "x_std", "y_mean", "y_std"}
            if not required_keys.issubset(blob.keys()):
                self.loaded_model = None
                self.loaded_model_mode = None
                snapshot_like = {"robot_pose", "obstacles", "target"}.issubset(blob.keys())
                if snapshot_like:
                    self.model_status = "Selected DAGGER snapshot, not model"
                else:
                    self.model_status = "Invalid model JSON (use train_mlp.py output)"
                return False

            file_mode = blob.get("mode")
            valid_modes = {"heading_drive", "xy_strafe", "heading_strafe"}
            mode_to_use = expected_mode
            if mode_to_use is None:
                mode_to_use = file_mode if isinstance(file_mode, str) and file_mode in valid_modes else self.robot.control_mode

            if file_mode is not None and file_mode != mode_to_use:
                self.loaded_model = None
                self.loaded_model_mode = None
                self.model_status = f"Mode mismatch: file={file_mode} ui={mode_to_use}"
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
            }
            self.loaded_model_mode = mode_to_use
            self.selected_model_paths[mode_to_use] = model_path
            if mode_to_use != self.robot.control_mode:
                self.robot.set_control_mode(mode_to_use)
                self.model_status = f"Loaded {model_path.name} ({mode_to_use})"
            else:
                self.model_status = f"Loaded {model_path.name}"
            return True
        except Exception as e:
            self.loaded_model = None
            self.loaded_model_mode = None
            self.model_status = f"Model load failed: {type(e).__name__}"
            return False

    def _pick_model_for_current_mode(self) -> bool:
        if self.model_pick_thread is not None and self.model_pick_thread.is_alive():
            self.model_status = "Model picker already open"
            return False

        def _worker() -> None:
            try:
                picked = self._ask_open_filename("Select trained model", self.model_dir)
            except Exception:
                picked = None
            self.model_pick_result = picked
            self.model_pick_result_ready = True

        self.model_pick_result = None
        self.model_pick_result_ready = False
        self.model_pick_thread = threading.Thread(target=_worker, daemon=True)
        self.model_pick_thread.start()
        self.model_status = "Opening model picker..."
        return True

    def _process_pending_model_pick(self) -> None:
        # Process pending Train-Me popup result
        if self.trainme_prompt_ready:
            self.trainme_prompt_ready = False
            ans = self.trainme_prompt_result
            self.trainme_prompt_result = None

            if ans is True:
                self._run_next_train_me_case()
            elif ans is False:
                self.new_episode()

        
        if not self.model_pick_result_ready:
            return

        self.model_pick_result_ready = False
        model_path_str = self.model_pick_result
        self.model_pick_result = None

        if model_path_str is None:
            self.model_status = "File picker unavailable"
            return
        if model_path_str == "":
            self.model_status = "Model pick cancelled"
            return

        self._load_model_from_file(Path(model_path_str), None)

    def _reload_model_for_current_mode(self) -> bool:
        selected = self.selected_model_paths.get(self.robot.control_mode)
        if selected is not None and selected.exists():
            return self._load_model_from_file(selected, self.robot.control_mode)
        return self._load_latest_model_for_mode(self.robot.control_mode)

    def _save_dagger_snapshot(self, tag: str = "snapshot") -> Optional[Path]:
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

    def _prompt_run_train_me(self):
        if self.trainme_prompt_thread is not None and self.trainme_prompt_thread.is_alive():
            return None  # still waiting

        # If a result is ready, return it
        if self.trainme_prompt_ready:
            self.trainme_prompt_ready = False
            return self.trainme_prompt_result

        # Otherwise start a new thread
        def worker():
            try:
                ans = self._ask_yes_no(
                    "Train-Me Queue",
                    f"There are {len(self.train_me_queue)} queued train-me cases.\n"
                    "Do you want to go through the next case now?"
                )
            except Exception:
                ans = True  # safe fallback
            self.trainme_prompt_result = ans
            self.trainme_prompt_ready = True

        self.trainme_prompt_thread = threading.Thread(target=worker, daemon=True)
        self.trainme_prompt_thread.start()
        return None


    def _run_next_train_me_case(self) -> bool:
        if not self.train_me_queue:
            return False

        seq, snapshot_path = self.train_me_queue.pop(0)
        if not self._load_dagger_snapshot(snapshot_path):
            self.model_status = f"Failed to load case #{seq}: {snapshot_path.name}"
            return False

        self.human_takeover_prompt_timer = 4.0
        self.active_train_me_case_seq = seq
        self.last_train_me_loaded_seq = seq

        # --- CRITICAL FIX: FORCE HUMAN CONTROL ---
        self.input_mode = "keyboard"
        self.robot.drive_command = {
            "rotation_rate": 0.0,
            "drive_speed": 0.0,
            "vx": 0.0,
            "vy": 0.0,
        }

        self.model_status = f"Loaded case #{seq} (HUMAN TAKEOVER)"
        return True

    def _trigger_dagger_human_takeover(self) -> None:
        self._enqueue_train_me_case("FAIL")

    def _predict_model_output(self, features: np.ndarray) -> Optional[np.ndarray]:
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
        if self.loaded_model is None:
            if self.robot.control_mode == "heading_drive":
                self.robot.drive_command = {"rotation_rate": 0.0, "drive_speed": 0.0}
            elif self.robot.control_mode == "heading_strafe":
                self.robot.drive_command = {"rotation_rate": 0.0, "vx": 0.0, "vy": 0.0}
            else:
                self.robot.drive_command = {"vx": 0.0, "vy": 0.0}
            return

        features = np.asarray(self.robot.whisker_lengths + [self.robot.heading_to_target_deg], dtype=np.float64).reshape(
            1, -1
        )
        pred = self._predict_model_output(features)
        if pred is None:
            return

        out = pred[0]
        if self.robot.control_mode == "heading_drive":
            drive_speed = float(np.clip(out[0], -Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS, Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS))
            rotation_rate = float(
                np.clip(out[1], -Robot.GAMEPAD_MAX_ROTATE_RATE_DPS, Robot.GAMEPAD_MAX_ROTATE_RATE_DPS)
            )
            self.robot.drive_command = {"rotation_rate": rotation_rate, "drive_speed": drive_speed}
        elif self.robot.control_mode == "heading_strafe":
            rotation_rate = float(
                np.clip(out[0], -Robot.GAMEPAD_MAX_ROTATE_RATE_DPS, Robot.GAMEPAD_MAX_ROTATE_RATE_DPS)
            )
            vx = float(np.clip(out[1], -Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS, Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS))
            vy = (
                float(np.clip(out[2], -Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS, Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS))
                if len(out) > 2
                else 0.0
            )
            self.robot.drive_command = {"rotation_rate": rotation_rate, "vx": vx, "vy": vy}
        else:
            vx = float(np.clip(out[0], -Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS, Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS))
            vy = float(np.clip(out[1], -Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS, Robot.GAMEPAD_MAX_DRIVE_SPEED_MPS))
            self.robot.drive_command = {"vx": vx, "vy": vy}

    def _handle_click(self, pos: Tuple[int, int]) -> None:

        # -----------------------------
        # RUN BUTTON — ONLY place where
        # queued cases are processed
        # -----------------------------
        if self.run_button_rect.collidepoint(pos):
            if self.train_me_queue:
                ans = self._prompt_run_train_me()
                if ans is True:
                    self._run_next_train_me_case()
                elif ans is False:
                    self.new_episode()
                # If ans is None, popup still loading — do nothing
            else:
                self.new_episode()
            return

        # -----------------------------
        # FAIL BUTTON — ONLY queues cases
        # NEVER processes queue
        # -----------------------------
        if self.fail_button_rect.collidepoint(pos):
            if self.input_mode == "model":
                self._trigger_dagger_human_takeover()
                self.new_episode()   # restart map, do NOT process queue
            else:
                self.model_status = "FAIL button requires Model input mode"
            return

        # -----------------------------
        # Everything below here is unchanged
        # -----------------------------
        if self.obs_minus_rect.collidepoint(pos):
            self.obstacle_count = max(1, self.obstacle_count - 1)
            return

        if self.obs_plus_rect.collidepoint(pos):
            self.obstacle_count = min(50, self.obstacle_count + 1)
            return

        if self.hist_minus_rect.collidepoint(pos):
            self.history_len = max(1, self.history_len - 1)
            self.history_buffer = deque(list(self.history_buffer)[-self.history_len:], maxlen=self.history_len)
            return

        if self.hist_plus_rect.collidepoint(pos):
            self.history_len = min(10, self.history_len + 1)
            self.history_buffer = deque(list(self.history_buffer)[-self.history_len:], maxlen=self.history_len)
            return

        if self.rate_minus_rect.collidepoint(pos):
            self.active_log_rate_hz = max(1.0, self.active_log_rate_hz - 1.0)
            return

        if self.rate_plus_rect.collidepoint(pos):
            self.active_log_rate_hz = min(30.0, self.active_log_rate_hz + 1.0)
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

            # If there are queued cases, prompt immediately
            if self.train_me_queue:
                ans = self._prompt_run_train_me()
                if ans is True:
                    self._run_next_train_me_case()
                elif ans is False:
                    self.new_episode()
                # If ans is None, popup still loading — do nothing
            return


        if self.radio_gamepad_rect.collidepoint(pos):
            self.input_mode = "gamepad"

            # If there are queued cases, prompt immediately
            if self.train_me_queue:
                ans = self._prompt_run_train_me()
                if ans is True:
                    self._run_next_train_me_case()
                elif ans is False:
                    self.new_episode()
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
        pygame.draw.rect(self.screen, self.ROOM_BG, self.sim_rect)

        self.screen.set_clip(self.sim_rect)
        if self.robot_aligned_view:
            room_corners = [
                self.world_to_screen(0.0, 0.0),
                self.world_to_screen(Robot.ROOM_WIDTH_M, 0.0),
                self.world_to_screen(Robot.ROOM_WIDTH_M, Robot.ROOM_HEIGHT_M),
                self.world_to_screen(0.0, Robot.ROOM_HEIGHT_M),
            ]
            pygame.draw.polygon(self.screen, self.ROOM_BG, room_corners)
            pygame.draw.polygon(self.screen, self.ROOM_BORDER, room_corners, 2)
        else:
            pygame.draw.rect(self.screen, self.ROOM_BORDER, self.room_rect_px, 2)

        for obstacle in self.obstacles:
            if not self._is_obstacle_in_action_radius(obstacle):
                continue
            ox, oy, radius = obstacle
            center_px = self.world_to_screen(ox, oy)
            pygame.draw.circle(self.screen, self.OBSTACLE_COLOR, center_px, max(2, int(radius * self.px_per_meter)))

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

        self._render_text("History", self.hist_minus_rect.left - 78, row1_y + 4, small=True)
        pygame.draw.rect(self.screen, (230, 230, 234), self.hist_minus_rect, border_radius=3)
        pygame.draw.rect(self.screen, (230, 230, 234), self.hist_plus_rect, border_radius=3)
        pygame.draw.rect(self.screen, (160, 160, 165), self.hist_minus_rect, 1, border_radius=3)
        pygame.draw.rect(self.screen, (160, 160, 165), self.hist_plus_rect, 1, border_radius=3)
        self._render_text("-", self.hist_minus_rect.left + 8, self.hist_minus_rect.top + 1)
        self._render_text("+", self.hist_plus_rect.left + 7, self.hist_plus_rect.top + 1)
        self._render_text(f"{self.history_len}", self.hist_minus_rect.right + 12, self.hist_minus_rect.top + 1)

        self._render_text("Rate Hz", self.rate_minus_rect.left - 78, row1_y + 4, small=True)
        pygame.draw.rect(self.screen, (230, 230, 234), self.rate_minus_rect, border_radius=3)
        pygame.draw.rect(self.screen, (230, 230, 234), self.rate_plus_rect, border_radius=3)
        pygame.draw.rect(self.screen, (160, 160, 165), self.rate_minus_rect, 1, border_radius=3)
        pygame.draw.rect(self.screen, (160, 160, 165), self.rate_plus_rect, 1, border_radius=3)
        self._render_text("-", self.rate_minus_rect.left + 8, self.rate_minus_rect.top + 1)
        self._render_text("+", self.rate_plus_rect.left + 7, self.rate_plus_rect.top + 1)
        self._render_text(f"{self.active_log_rate_hz:.0f}", self.rate_minus_rect.right + 8, self.rate_minus_rect.top + 1)

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

    def _draw_panel(self) -> None:
        pygame.draw.rect(self.screen, self.PANEL_COLOR, self.panel_rect)
        pygame.draw.rect(self.screen, (180, 180, 185), self.panel_rect, 1)

        model_color = (30, 140, 60) if self.loaded_model is not None else (180, 70, 40)
        self._render_text("Model Status", self.panel_rect.left + 22, self.panel_rect.top + 16, small=True)
        self._render_text(self.model_status[:36], self.panel_rect.left + 22, self.panel_rect.top + 34, model_color, small=True)
        next_seq = str(self.train_me_queue[0][0]) if self.train_me_queue else "-"
        last_seq = str(self.last_train_me_loaded_seq) if self.last_train_me_loaded_seq is not None else "-"
        self._render_text(
            f"Q:{len(self.train_me_queue)} N:{next_seq} L:{last_seq}",
            self.panel_rect.left + 22,
            self.panel_rect.top + 54,
            small=True,
        )

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

            # -------------------------
            # EVENT HANDLING
            # -------------------------
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

            # -------------------------
            # PROCESS MODEL PICK POPUP
            # -------------------------
            self._process_pending_model_pick()

            # -------------------------
            # PROCESS TRAIN-ME POPUP RESULT
            # (THIS IS THE FIX YOU NEEDED)
            # -------------------------
            if self.trainme_prompt_ready:
                self.trainme_prompt_ready = False
                ans = self.trainme_prompt_result
                self.trainme_prompt_result = None

                if ans is True:
                    self._run_next_train_me_case()
                elif ans is False:
                    self.new_episode()

            # -------------------------
            # SIMULATION UPDATE
            # -------------------------
            self.update_robot_command()
            self.whisker_segments = self.robot.step(dt, self.obstacles)
            self.robot.update_heading_to_target(self.target)

            # -------------------------
            # LOGGING
            # -------------------------
            self.log_timer += dt
            log_interval = self._current_log_interval()
            if log_interval is not None and self.log_timer >= log_interval:
                self.in_memory_log.append(self._collect_log_entry())
                self.log_timer = 0.0

            # -------------------------
            # GOAL / COLLISION HANDLING
            # -------------------------
            if self.check_goal_reached():
                if self.logging_enabled and self.in_memory_log:
                    self._write_log_to_file()
                self.in_memory_log = []
                if self.active_train_me_case_seq is not None and self.train_me_queue:
                    self._run_next_train_me_case()
                else:
                    self.new_episode()

            elif self.robot.collision_flag or self.check_whisker_collision():
                print("COLLISION")
                if self.input_mode == "model":
                    self._enqueue_train_me_case("collision")
                self.in_memory_log = []
                self.collision_display_timer = 0.5
                self.new_episode()

            # -------------------------
            # TIMERS
            # -------------------------
            if self.collision_display_timer > 0.0:
                self.collision_display_timer = max(0.0, self.collision_display_timer - dt)
            if self.human_takeover_prompt_timer > 0.0:
                self.human_takeover_prompt_timer = max(0.0, self.human_takeover_prompt_timer - dt)

            # -------------------------
            # DRAWING
            # -------------------------
            self.screen.fill(self.BG_COLOR)
            self._draw_top_bar()
            self._draw_simulation()
            self._draw_panel()
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
