"""Shared simulation core logic for navigation tasks.

This module intentionally contains no rendering or display calls.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None

Vec2 = np.ndarray

ROOM_WIDTH = 5.0
ROOM_HEIGHT = 3.0
WHISKER_ANGLES = [-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90]
WHISKER_MAX_LENGTH = 0.50
TARGET_REACH_RADIUS = 0.25

# Simulated CV-bbox offset envelope, per axis, in metres. Each episode the
# target's circumscribing square is translated by a uniform draw from
# [-TARGET_BBOX_OFFSET_MAX_M, +TARGET_BBOX_OFFSET_MAX_M] on x and y, and
# rotated by a uniform draw from [0, 2π). Models the CV pipeline's bbox slop
# so the policy learns to fuse "whisker range" with "whisker-vs-bbox range".
TARGET_BBOX_OFFSET_MAX_M = 0.025


def generate_target_bbox_pose(rng: np.random.Generator) -> Dict[str, float]:
    """Sample a per-episode bbox pose (rotation + xy offset) for the target."""
    return {
        "rotation_rad": float(rng.uniform(0.0, 2.0 * math.pi)),
        "offset_x": float(rng.uniform(-TARGET_BBOX_OFFSET_MAX_M, TARGET_BBOX_OFFSET_MAX_M)),
        "offset_y": float(rng.uniform(-TARGET_BBOX_OFFSET_MAX_M, TARGET_BBOX_OFFSET_MAX_M)),
    }


def randomize_robot_pose(
    rng: np.random.Generator,
    room_w: float,
    room_h: float,
    clearance_from_walls_m: float = 0.30,
    control_mode: str = "heading_drive",
) -> Dict[str, float]:
    """Sample a collision-safe starting pose with heading constrained away from walls."""
    x = float(rng.uniform(clearance_from_walls_m, room_w - clearance_from_walls_m))
    y = float(rng.uniform(clearance_from_walls_m, room_h - clearance_from_walls_m))

    if control_mode in ("heading_drive", "heading_strafe"):
        heading = float(rng.uniform(-180.0, 180.0))
        return {"x": x, "y": y, "heading": heading}

    to_center = np.array([0.5 * room_w - x, 0.5 * room_h - y], dtype=float)
    base_heading = math.degrees(math.atan2(float(to_center[0]), float(to_center[1])))

    heading = base_heading
    for _ in range(200):
        candidate = base_heading + float(rng.uniform(-60.0, 60.0))
        theta = math.radians(candidate)
        fwd = np.array([math.sin(theta), math.cos(theta)], dtype=float)
        projected = np.array([x, y], dtype=float) + fwd * 0.20
        if 0.0 < projected[0] < room_w and 0.0 < projected[1] < room_h:
            heading = candidate
            break

    return {"x": x, "y": y, "heading": float(heading)}


def _point_to_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def _robot_edge_segments(robot_pose: Dict[str, float], body_length: float, body_width: float) -> List[Tuple[Vec2, Vec2]]:
    x = float(robot_pose["x"])
    y = float(robot_pose["y"])
    heading = float(robot_pose["heading"])
    theta = math.radians(heading)
    local_x = np.array([math.sin(theta), math.cos(theta)], dtype=float)
    local_y = np.array([-math.cos(theta), math.sin(theta)], dtype=float)
    center = np.array([x, y], dtype=float)
    hx = 0.5 * body_length
    hy = 0.5 * body_width
    corners = [
        center + (+hx * local_x) + (+hy * local_y),
        center + (-hx * local_x) + (+hy * local_y),
        center + (-hx * local_x) + (-hy * local_y),
        center + (+hx * local_x) + (-hy * local_y),
    ]
    return [
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    ]


def generate_obstacles(
    rng: np.random.Generator,
    room_w: float,
    room_h: float,
    robot_pose: Dict[str, float],
    n_obs: int = 8,
    obstacle_radius: float = 0.05,
    robot_collision_radius: float = 0.16,
    robot_body_length: float = 0.25,
    robot_body_width: float = 0.20,
) -> List[Dict[str, float]]:
    """Generate non-overlapping obstacle circles with clearance around robot."""
    obstacles: List[Dict[str, float]] = []

    robot_center = np.array([float(robot_pose["x"]), float(robot_pose["y"])], dtype=float)
    robot_edges = _robot_edge_segments(robot_pose, robot_body_length, robot_body_width)

    for _ in range(2000):
        if len(obstacles) >= int(n_obs):
            break

        ox = float(rng.uniform(obstacle_radius, room_w - obstacle_radius))
        oy = float(rng.uniform(obstacle_radius, room_h - obstacle_radius))
        center = np.array([ox, oy], dtype=float)

        valid = True
        for ob in obstacles:
            d = math.hypot(ox - float(ob["x"]), oy - float(ob["y"]))
            if d < (obstacle_radius + float(ob["radius"])):
                valid = False
                break
        if not valid:
            continue

        center_dist_to_robot = float(np.linalg.norm(center - robot_center))
        if center_dist_to_robot < robot_collision_radius + obstacle_radius + 0.20:
            valid = False
        else:
            min_seg_dist = min(_point_to_segment_distance(center, a, b) for a, b in robot_edges)
            if min_seg_dist < obstacle_radius + 0.20:
                valid = False

        if valid:
            obstacles.append({"x": ox, "y": oy, "radius": obstacle_radius})

    return obstacles


def generate_target(
    rng: np.random.Generator,
    room_w: float,
    room_h: float,
    obstacles: List[Dict[str, float]],
    robot_pose: Dict[str, float],
    target_r: float = 0.075,
) -> Dict[str, float]:
    """Generate a target point with obstacle and robot clearance."""
    for _ in range(3000):
        tx = float(rng.uniform(0.30 + target_r, room_w - 0.30 - target_r))
        ty = float(rng.uniform(0.30 + target_r, room_h - 0.30 - target_r))

        ok = True
        for ob in obstacles:
            d = math.hypot(tx - float(ob["x"]), ty - float(ob["y"]))
            if d < target_r + float(ob["radius"]) + 0.30:
                ok = False
                break
        if not ok:
            continue

        dr = math.hypot(tx - float(robot_pose["x"]), ty - float(robot_pose["y"]))
        if dr < 0.50 + target_r:
            continue

        return {"x": tx, "y": ty, "radius": target_r}

    return {"x": 0.5 * room_w, "y": 0.5 * room_h, "radius": target_r}


def generate_target_against_wall(
    rng: np.random.Generator,
    room_w: float,
    room_h: float,
    obstacles: List[Dict[str, float]],
    robot_pose: Dict[str, float],
    target_r: float = 0.075,
    corner_clearance_m: float = 0.50,
) -> Dict[str, float]:
    """Place the target's center on a random wall plane, avoiding corners.

    Trains approaches where an object is up against a wall: all collision
    whiskers register an obstacle ahead, but one of them is actually the target.
    """
    for _ in range(3000):
        wall = int(rng.integers(0, 4))
        if wall == 0:  # bottom
            tx = float(rng.uniform(corner_clearance_m, room_w - corner_clearance_m))
            ty = 0.0
        elif wall == 1:  # top
            tx = float(rng.uniform(corner_clearance_m, room_w - corner_clearance_m))
            ty = float(room_h)
        elif wall == 2:  # left
            tx = 0.0
            ty = float(rng.uniform(corner_clearance_m, room_h - corner_clearance_m))
        else:  # right
            tx = float(room_w)
            ty = float(rng.uniform(corner_clearance_m, room_h - corner_clearance_m))

        ok = True
        for ob in obstacles:
            d = math.hypot(tx - float(ob["x"]), ty - float(ob["y"]))
            if d < target_r + float(ob["radius"]) + 0.30:
                ok = False
                break
        if not ok:
            continue

        dr = math.hypot(tx - float(robot_pose["x"]), ty - float(robot_pose["y"]))
        if dr < 0.50 + target_r:
            continue

        return {"x": tx, "y": ty, "radius": target_r}

    return {"x": 0.5 * room_w, "y": 0.0, "radius": target_r}


class Robot:
    """Shared robot dynamics and sensing model."""

    ROOM_WIDTH_M = ROOM_WIDTH
    ROOM_HEIGHT_M = ROOM_HEIGHT

    BODY_WIDTH_M = 0.20
    BODY_LENGTH_M = 0.25
    AXIS_ARROW_LEN_M = 0.10

    WHISKER_ANGLES_DEG = WHISKER_ANGLES
    WHISKER_MAX_LEN_M = WHISKER_MAX_LENGTH

    KEYBOARD_DRIVE_SPEED_MPS = 0.20
    KEYBOARD_ROTATE_RATE_DPS = 20.0
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
        self.target_bbox_lengths: List[float] = [self.WHISKER_MAX_LEN_M for _ in self.WHISKER_ANGLES_DEG]
        self.robot_pose: Dict[str, float] = {"x": 0.0, "y": 0.0, "heading": 0.0}
        self.heading_to_target_deg = 0.0
        self.collision_flag = False
        self.target_contact_flag = False
        # "wall" | "obstacle" | "target" | None. Set by check_collision alongside
        # collision_flag / target_contact_flag so callers can categorise by source.
        self.collision_source: Optional[str] = None

        self.collision_flash_timer = 0.0
        self.collision_radius = 0.5 * math.hypot(self.BODY_WIDTH_M, self.BODY_LENGTH_M)

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
        return (value + deadband) / (1.0 - deadband)

    @staticmethod
    def _shape_gamepad_axis(value: float) -> float:
        return value * value * value

    @staticmethod
    def apply_rotation_stiction(
        rate_dps: float,
        min_turn_rate_dps: float,
        inner_deadband_dps: float,
    ) -> float:
        if min_turn_rate_dps <= 0.0:
            return rate_dps
        mag = abs(rate_dps)
        if mag == 0.0 or mag < inner_deadband_dps:
            return 0.0
        if mag < min_turn_rate_dps:
            return math.copysign(min_turn_rate_dps, rate_dps)
        return rate_dps

    @staticmethod
    def apply_rotation_pipeline(
        rate_dps: float,
        min_turn_rate_dps: float,
        inner_deadband_dps: float,
        exit_dps: float,
        state: Dict[str, bool],
    ) -> float:
        """Combined dual-band stiction snap + hysteresis with feather band.

        Used at runtime (model playback in sim, real-robot ROS node) to give
        the steering channel a 'feathering' region below `min_turn_rate_dps`
        that is reachable only once a turn is already in progress. Removes
        chatter without losing stiction protection on entry.

        State machine:
          not_turning:
            |rate| <  inner_db                -> 0
            inner_db <= |rate| < min          -> sign(rate) * min   (snap up + enter turning)
            |rate| >= min                     -> rate               (passthrough + enter turning)
          turning:
            |rate| <  exit_dps                -> 0                  (drop out of turn)
            exit_dps <= |rate|                -> rate               (feather + passthrough)

        `exit_dps` should be < `min_turn_rate_dps`. If `min_turn_rate_dps`
        is 0 the whole pipeline is a no-op.

        `state` is a single-key dict {"was_turning": bool}, persisted across
        ticks by the caller. Reset to False at episode start.
        """
        if min_turn_rate_dps <= 0.0:
            return rate_dps
        mag = abs(rate_dps)
        if state.get("was_turning", False):
            if mag < exit_dps:
                state["was_turning"] = False
                return 0.0
            return rate_dps
        # not currently turning
        if mag < inner_deadband_dps or mag == 0.0:
            return 0.0
        state["was_turning"] = True
        if mag < min_turn_rate_dps:
            return math.copysign(min_turn_rate_dps, rate_dps)
        return rate_dps

    def reset_random_pose(self, rng: Optional[np.random.Generator] = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        pose = randomize_robot_pose(
            rng=rng,
            room_w=self.ROOM_WIDTH_M,
            room_h=self.ROOM_HEIGHT_M,
            clearance_from_walls_m=self.CLEARANCE_FROM_WALLS_M,
            control_mode=self.control_mode,
        )
        self.x = float(pose["x"])
        self.y = float(pose["y"])
        self.heading_deg = float(pose["heading"])

        if self.control_mode == "heading_drive":
            self.drive_command = {"rotation_rate": 0.0, "drive_speed": 0.0}
        elif self.control_mode == "heading_strafe":
            self.drive_command = {"rotation_rate": 0.0, "vx": 0.0, "vy": 0.0}
        else:
            self.drive_command = {"vx": 0.0, "vy": 0.0}

        self.whisker_lengths = [self.WHISKER_MAX_LEN_M for _ in self.WHISKER_ANGLES_DEG]
        self.target_bbox_lengths = [self.WHISKER_MAX_LEN_M for _ in self.WHISKER_ANGLES_DEG]
        self.robot_pose = {"x": self.x, "y": self.y, "heading": self.heading_deg}
        self.collision_flag = False
        self.target_contact_flag = False
        self.collision_source = None
        self.collision_flash_timer = 0.0

    def compute_heading_to_target(self, target_x: float, target_y: float) -> float:
        # Right-hand (z-up) positive yaw convention: target to the robot's LEFT
        # yields a positive angle; target to the RIGHT yields a negative angle.
        # A positive heading_to_target is therefore reduced by a positive
        # rotation_rate (CCW, turn-left) in the same right-hand convention.
        dx = target_x - self.x
        dy = target_y - self.y
        world_angle_to_target_deg = math.degrees(math.atan2(dx, dy))
        heading_to_target = self.heading_deg - world_angle_to_target_deg
        return ((heading_to_target + 180.0) % 360.0) - 180.0

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

    def update_command_from_keys(self, keys: object) -> None:
        if pygame is None:
            return
        # rotation_rate is right-hand positive (CCW / turn-left).
        if self.control_mode == "heading_drive":
            drive_speed = 0.0
            rotation_rate = 0.0
            if keys[pygame.K_UP]:
                drive_speed += self.KEYBOARD_DRIVE_SPEED_MPS
            if keys[pygame.K_DOWN]:
                drive_speed -= self.KEYBOARD_DRIVE_SPEED_MPS
            if keys[pygame.K_LEFT]:
                rotation_rate += self.KEYBOARD_ROTATE_RATE_DPS
            if keys[pygame.K_RIGHT]:
                rotation_rate -= self.KEYBOARD_ROTATE_RATE_DPS
            self.drive_command = {"rotation_rate": rotation_rate, "drive_speed": drive_speed}
        elif self.control_mode == "heading_strafe":
            rotation_rate = 0.0
            vx = 0.0
            vy = 0.0
            if keys[pygame.K_UP]:
                vx += self.KEYBOARD_DRIVE_SPEED_MPS
            if keys[pygame.K_DOWN]:
                vx -= self.KEYBOARD_DRIVE_SPEED_MPS
            if keys[pygame.K_LEFT]:
                rotation_rate += self.KEYBOARD_ROTATE_RATE_DPS
            if keys[pygame.K_RIGHT]:
                rotation_rate -= self.KEYBOARD_ROTATE_RATE_DPS
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

    def update_command_from_gamepad(self, joystick: object) -> None:
        if joystick is None:
            return
        axis_count = joystick.get_numaxes()

        # rotation_rate is right-hand positive (CCW / turn-left); stick-right
        # (positive z-axis) therefore produces a negative rotation_rate.
        if self.control_mode == "heading_drive":
            drive_speed = 0.0
            rotation_rate = 0.0
            if axis_count > 1:
                y_axis = self._shape_gamepad_axis(self._apply_deadband(joystick.get_axis(1)))
                drive_speed = -y_axis * self.GAMEPAD_MAX_DRIVE_SPEED_MPS
            if axis_count > 2:
                z_axis = self._shape_gamepad_axis(self._apply_deadband(joystick.get_axis(2)))
                rotation_rate = -z_axis * self.GAMEPAD_MAX_ROTATE_RATE_DPS
            self.drive_command = {"rotation_rate": rotation_rate, "drive_speed": drive_speed}
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
                rotation_rate = -z_axis * self.GAMEPAD_MAX_ROTATE_RATE_DPS
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

    def apply_normalized_action(self, action: np.ndarray) -> None:
        a0 = float(np.clip(action[0], -1.0, 1.0))
        a1 = float(np.clip(action[1], -1.0, 1.0))
        if self.control_mode == "heading_drive":
            self.drive_command = {
                "rotation_rate": a0 * self.GAMEPAD_MAX_ROTATE_RATE_DPS,
                "drive_speed": a1 * self.GAMEPAD_MAX_DRIVE_SPEED_MPS,
            }
        elif self.control_mode == "heading_strafe":
            a2 = float(np.clip(action[2], -1.0, 1.0))
            self.drive_command = {
                "rotation_rate": a0 * self.GAMEPAD_MAX_ROTATE_RATE_DPS,
                "vx": a1 * self.GAMEPAD_MAX_DRIVE_SPEED_MPS,
                "vy": a2 * self.GAMEPAD_MAX_DRIVE_SPEED_MPS,
            }
        else:
            self.drive_command = {
                "vx": a0 * self.GAMEPAD_MAX_DRIVE_SPEED_MPS,
                "vy": a1 * self.GAMEPAD_MAX_DRIVE_SPEED_MPS,
            }

    def integrate(self, dt: float) -> None:
        local_x, local_y = self._basis_vectors()

        # Note: the internal heading_deg is measured CW-from-north (a left-hand
        # convention), so a right-hand positive rotation_rate (CCW, turn-left)
        # is applied as a *decrease* in heading_deg.
        if self.control_mode == "heading_drive":
            rot_dps = self.drive_command.get("rotation_rate", 0.0)
            drive_speed = self.drive_command.get("drive_speed", 0.0)
            self.heading_deg = ((self.heading_deg - rot_dps * dt + 180.0) % 360.0) - 180.0
            displacement = local_x * drive_speed * dt
        elif self.control_mode == "heading_strafe":
            rot_dps = self.drive_command.get("rotation_rate", 0.0)
            vx = self.drive_command.get("vx", 0.0)
            vy = self.drive_command.get("vy", 0.0)
            self.heading_deg = ((self.heading_deg - rot_dps * dt + 180.0) % 360.0) - 180.0
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
        return min(ts) if ts else None

    @staticmethod
    def _ray_segment_intersection(origin: Vec2, direction: Vec2, p1: Vec2, p2: Vec2) -> Optional[float]:
        v1 = origin - p1
        v2 = p2 - p1
        denom = direction[0] * v2[1] - direction[1] * v2[0]
        if abs(float(denom)) < 1e-10:
            return None

        t = (v2[0] * v1[1] - v2[1] * v1[0]) / denom
        u = (direction[0] * v1[1] - direction[1] * v1[0]) / denom
        if t >= 0.0 and 0.0 <= u <= 1.0:
            return float(t)
        return None

    @staticmethod
    def _ray_obb_intersection(
        origin: Vec2,
        direction: Vec2,
        center: Vec2,
        half_extents: Tuple[float, float],
        rotation_rad: float,
    ) -> Optional[float]:
        """Ray vs. oriented 2D box (slab method in the box's local frame)."""
        cos_r = math.cos(rotation_rad)
        sin_r = math.sin(rotation_rad)
        # World-to-local rotation: transpose of [[cos, -sin], [sin, cos]].
        ox = origin[0] - center[0]
        oy = origin[1] - center[1]
        lox = cos_r * ox + sin_r * oy
        loy = -sin_r * ox + cos_r * oy
        ldx = cos_r * direction[0] + sin_r * direction[1]
        ldy = -sin_r * direction[0] + cos_r * direction[1]

        hx, hy = half_extents
        tmin = -math.inf
        tmax = math.inf
        for lo, ld, h in ((lox, ldx, hx), (loy, ldy, hy)):
            if abs(ld) < 1e-12:
                if lo < -h or lo > h:
                    return None
                continue
            t1 = (-h - lo) / ld
            t2 = (h - lo) / ld
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > tmin:
                tmin = t1
            if t2 < tmax:
                tmax = t2
            if tmin > tmax:
                return None
        if tmax < 0.0:
            return None
        return float(max(0.0, tmin))

    def compute_whiskers(
        self,
        obstacles: List[Tuple[float, float, float]],
        target: Optional[Tuple[float, float, float]] = None,
        target_bbox_pose: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Vec2, Vec2, bool]]:
        local_x, _ = self._basis_vectors()
        origin = np.array([self.x, self.y], dtype=float) + local_x * 0.12
        whisker_segments: List[Tuple[Vec2, Vec2, bool]] = []
        lengths: List[float] = []
        target_bbox_lengths: List[float] = []

        wall_segments = [
            (np.array([0.0, 0.0], dtype=float), np.array([self.ROOM_WIDTH_M, 0.0], dtype=float)),
            (np.array([self.ROOM_WIDTH_M, 0.0], dtype=float), np.array([self.ROOM_WIDTH_M, self.ROOM_HEIGHT_M], dtype=float)),
            (np.array([self.ROOM_WIDTH_M, self.ROOM_HEIGHT_M], dtype=float), np.array([0.0, self.ROOM_HEIGHT_M], dtype=float)),
            (np.array([0.0, self.ROOM_HEIGHT_M], dtype=float), np.array([0.0, 0.0], dtype=float)),
        ]

        # Primary whisker channel hits "anything that isn't empty space"; on
        # hardware this includes the physical target, which we model in sim as
        # a circle of radius `target[2]` centred at `target[:2]`.
        target_center: Optional[Vec2] = None
        target_radius = 0.0
        if target is not None:
            target_center = np.array([target[0], target[1]], dtype=float)
            target_radius = float(target[2])

        # Parallel target-bbox channel: CV-pipeline proxy. Tight square of side
        # 2*radius rotated by rotation_rad and translated by (offset_x,
        # offset_y). The policy fuses the two channels to learn that target
        # hits are attractive, not obstacles.
        bbox_center: Optional[Vec2] = None
        bbox_half_extents: Optional[Tuple[float, float]] = None
        bbox_rotation_rad = 0.0
        if target is not None and target_bbox_pose is not None:
            tx, ty, tr = target
            bbox_center = np.array(
                [tx + float(target_bbox_pose.get("offset_x", 0.0)),
                 ty + float(target_bbox_pose.get("offset_y", 0.0))],
                dtype=float,
            )
            bbox_half_extents = (float(tr), float(tr))
            bbox_rotation_rad = float(target_bbox_pose.get("rotation_rad", 0.0))

        for angle_deg in self.WHISKER_ANGLES_DEG:
            theta = self._deg_to_rad(self.heading_deg + angle_deg)
            direction = np.array([math.sin(theta), math.cos(theta)], dtype=float)

            min_dist = self.WHISKER_MAX_LEN_M
            hit = False

            for ox, oy, radius in obstacles:
                center = np.array([ox, oy], dtype=float)
                d = self._ray_circle_intersection(origin, direction, center, radius)
                if d is not None and 0.0 <= d < min_dist:
                    min_dist = float(d)
                    hit = True

            for p1, p2 in wall_segments:
                d = self._ray_segment_intersection(origin, direction, p1, p2)
                if d is not None and 0.0 <= d < min_dist:
                    min_dist = float(d)
                    hit = True

            if target_center is not None and target_radius > 0.0:
                d = self._ray_circle_intersection(origin, direction, target_center, target_radius)
                if d is not None and 0.0 <= d < min_dist:
                    min_dist = float(d)
                    hit = True

            if bbox_center is not None and bbox_half_extents is not None:
                bbox_d = self._ray_obb_intersection(
                    origin, direction, bbox_center, bbox_half_extents, bbox_rotation_rad
                )
            else:
                bbox_d = None

            if bbox_d is None or bbox_d >= self.WHISKER_MAX_LEN_M:
                target_bbox_lengths.append(self.WHISKER_MAX_LEN_M)
            else:
                target_bbox_lengths.append(float(max(0.0, bbox_d)))

            endpoint = origin + direction * min_dist
            whisker_segments.append((origin.copy(), endpoint, hit))
            lengths.append(min_dist)

        self.whisker_lengths = lengths
        self.target_bbox_lengths = target_bbox_lengths
        return whisker_segments

    def check_collision(
        self,
        obstacles: List[Tuple[float, float, float]],
        target: Optional[Tuple[float, float, float]] = None,
    ) -> bool:
        """Physical overlap check. Sets:
          - collision_flag + collision_source='wall'|'obstacle' on hard failures
          - target_contact_flag + collision_source='target' when robot body
            overlaps the target circle (a successful reach, not a failure)
        Returns True only for hard-failure overlap (wall/obstacle). A
        target-only overlap returns False so Robot.step does not revert the
        robot's position — the caller promotes target_contact_flag into a
        goal-reached event.
        """
        hard_collided = False
        source: Optional[str] = None

        if self.x - self.collision_radius < 0.0:
            hard_collided = True; source = "wall"
        if self.x + self.collision_radius > self.ROOM_WIDTH_M:
            hard_collided = True; source = "wall"
        if self.y - self.collision_radius < 0.0:
            hard_collided = True; source = "wall"
        if self.y + self.collision_radius > self.ROOM_HEIGHT_M:
            hard_collided = True; source = "wall"

        for ox, oy, radius in obstacles:
            dx = self.x - ox
            dy = self.y - oy
            if dx * dx + dy * dy <= (self.collision_radius + radius) ** 2:
                hard_collided = True
                source = "obstacle"
                break

        target_hit = False
        if target is not None:
            tx, ty, tr = target
            dx = self.x - tx
            dy = self.y - ty
            if dx * dx + dy * dy <= (self.collision_radius + float(tr)) ** 2:
                target_hit = True

        if target_hit:
            # Recorded independently of hard_collided so that simultaneous
            # target + obstacle overlap still surfaces the target contact.
            # Callers that treat target contact as success should check this
            # flag before acting on collision_flag.
            self.target_contact_flag = True

        if hard_collided:
            self.collision_flag = True
            self.collision_source = source
            self.collision_flash_timer = 0.20
            if self.control_mode == "heading_drive":
                self.drive_command = {"rotation_rate": 0.0, "drive_speed": 0.0}
            elif self.control_mode == "heading_strafe":
                self.drive_command = {"rotation_rate": 0.0, "vx": 0.0, "vy": 0.0}
            else:
                self.drive_command = {"vx": 0.0, "vy": 0.0}
        elif target_hit:
            self.collision_source = "target"

        return hard_collided

    def step(
        self,
        dt: float,
        obstacles: List[Tuple[float, float, float]],
        target: Optional[Tuple[float, float, float]] = None,
        target_bbox_pose: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Vec2, Vec2, bool]]:
        prev_x, prev_y, prev_h = self.x, self.y, self.heading_deg
        self.integrate(dt)
        if self.check_collision(obstacles, target=target):
            self.x, self.y, self.heading_deg = prev_x, prev_y, prev_h

        whiskers = self.compute_whiskers(obstacles, target=target, target_bbox_pose=target_bbox_pose)

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
