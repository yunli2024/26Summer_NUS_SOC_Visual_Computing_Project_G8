"""Three-lane runner game state and OpenCV renderer."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import cv2
import numpy as np

from . import config
from .runner_actions import Action


LEFT_LANE = 0
CENTER_LANE = 1
RIGHT_LANE = 2
LANE_NAMES = {LEFT_LANE: "LEFT", CENTER_LANE: "CENTER", RIGHT_LANE: "RIGHT"}
OBSTACLE_LOW = "low_barrier"
OBSTACLE_HIGH = "high_bar"
OBSTACLE_FULL = "full_block"


@dataclass
class Obstacle:
    lane: int
    kind: str
    y: float
    passed: bool = False


class RunnerGame:
    def __init__(self):
        self.rng = random.Random(7)
        self.reset()

    def reset(self):
        self.lane = config.RUNNER_START_LANE
        self.target_lane = config.RUNNER_START_LANE
        self.player_x = float(config.RUNNER_LANE_X[self.lane])
        self.jump_time = 0.0
        self.slide_time = 0.0
        self.obstacles: list[Obstacle] = []
        self.spawn_timer = 0.0
        self.score = 0
        self.alive_time = 0.0
        self.game_over = False
        self.message = "校准后自动开始，或使用键盘测试"

    @property
    def jumping(self) -> bool:
        return self.jump_time > 0.0

    @property
    def sliding(self) -> bool:
        return self.slide_time > 0.0

    @property
    def lane_name(self) -> str:
        return LANE_NAMES[self.target_lane]

    def handle_action(self, action: Action):
        if self.game_over or action is Action.NONE:
            return
        if action is Action.LEFT:
            self.target_lane = max(LEFT_LANE, self.target_lane - 1)
            self.message = "Lane left"
        elif action is Action.RIGHT:
            self.target_lane = min(RIGHT_LANE, self.target_lane + 1)
            self.message = "Lane right"
        elif action is Action.JUMP and not self.jumping:
            self.jump_time = config.RUNNER_JUMP_DURATION
            self.message = "Jump"
        elif action is Action.SLIDE and not self.sliding:
            self.slide_time = config.RUNNER_SLIDE_DURATION
            self.message = "Slide"

    def update(self, dt: float):
        if self.game_over:
            return
        self.alive_time += dt
        self.score = int(self.alive_time * 100)
        self._update_player(dt)
        self._update_obstacles(dt)
        self._spawn_obstacles(dt)
        self._check_collisions()

    def _update_player(self, dt: float):
        target_x = float(config.RUNNER_LANE_X[self.target_lane])
        step = config.RUNNER_LANE_MOVE_SPEED * dt
        if abs(target_x - self.player_x) <= step:
            self.player_x = target_x
            self.lane = self.target_lane
        else:
            self.player_x += step if target_x > self.player_x else -step
        self.jump_time = max(0.0, self.jump_time - dt)
        self.slide_time = max(0.0, self.slide_time - dt)

    def _update_obstacles(self, dt: float):
        speed = config.RUNNER_OBSTACLE_SPEED_START + self.alive_time * config.RUNNER_OBSTACLE_SPEED_GAIN
        for obstacle in self.obstacles:
            obstacle.y += speed * dt
            if not obstacle.passed and obstacle.y > config.RUNNER_PLAYER_Y + 55:
                obstacle.passed = True
                self.score += 80
        self.obstacles = [obs for obs in self.obstacles if obs.y < config.RUNNER_WINDOW_SIZE[1] + 80]

    def _spawn_obstacles(self, dt: float):
        self.spawn_timer -= dt
        if self.spawn_timer > 0:
            return
        self.spawn_timer = max(0.64, config.RUNNER_OBSTACLE_SPAWN_SECONDS - self.alive_time * 0.006)
        lane = self.rng.choice(config.RUNNER_LANES)
        kind = self.rng.choice([OBSTACLE_LOW, OBSTACLE_HIGH, OBSTACLE_FULL])
        if self.obstacles and abs(self.obstacles[-1].y) < 140 and self.obstacles[-1].lane == lane:
            lane = (lane + 1) % 3
        self.obstacles.append(Obstacle(lane=lane, kind=kind, y=-70.0))

    def _check_collisions(self):
        player_y = config.RUNNER_PLAYER_Y
        for obstacle in self.obstacles:
            obstacle_x = config.RUNNER_LANE_X[obstacle.lane]
            if abs(obstacle_x - self.player_x) > config.RUNNER_PLAYER_COLLISION_HALF_WIDTH:
                continue
            if abs(obstacle.y - player_y) > config.RUNNER_COLLISION_Y_WINDOW:
                continue
            if obstacle.kind == OBSTACLE_LOW and self.jump_height() < config.RUNNER_LOW_CLEAR_HEIGHT:
                self._crash("撞到低矮路障，需要跳跃")
            elif obstacle.kind == OBSTACLE_HIGH and not self.sliding:
                self._crash("撞到高处横杆，需要下滑")
            elif obstacle.kind == OBSTACLE_FULL:
                self._crash("撞到整道障碍，需要换道")

    def _crash(self, message: str):
        self.game_over = True
        self.message = message

    def jump_height(self) -> float:
        if not self.jumping:
            return 0.0
        progress = 1.0 - self.jump_time / config.RUNNER_JUMP_DURATION
        return math.sin(progress * math.pi) * config.RUNNER_JUMP_HEIGHT

    def render(self, camera_preview=None, action=Action.NONE, recognizer_status="", calibrated=False, warning=""):
        width, height = config.RUNNER_WINDOW_SIZE
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._draw_background(frame)
        self._draw_lanes(frame)
        for obstacle in self.obstacles:
            self._draw_obstacle(frame, obstacle)
        self._draw_player(frame)
        self._draw_hud(frame, action, recognizer_status, calibrated, warning)
        self._draw_preview(frame, camera_preview)
        if self.game_over:
            self._draw_game_over(frame)
        return frame

    def _draw_background(self, frame):
        h, w = frame.shape[:2]
        frame[:] = (22, 27, 34)
        cv2.rectangle(frame, (180, 80), (1000, h), (38, 45, 52), -1)
        for i in range(16):
            y = int(100 + i * 48 + (self.alive_time * 120) % 48)
            cv2.line(frame, (230, y), (950, y), (55, 64, 72), 1)

    def _draw_lanes(self, frame):
        top_y = 90
        bottom_y = config.RUNNER_WINDOW_SIZE[1]
        for x in config.RUNNER_LANE_X:
            cv2.line(frame, (x, top_y), (x, bottom_y), (78, 95, 110), 3)
        cv2.line(frame, (config.RUNNER_LANE_X[0] - 100, top_y), (config.RUNNER_LANE_X[0] - 100, bottom_y), (95, 110, 125), 3)
        cv2.line(frame, (config.RUNNER_LANE_X[2] + 100, top_y), (config.RUNNER_LANE_X[2] + 100, bottom_y), (95, 110, 125), 3)

    def _draw_player(self, frame):
        x = int(self.player_x)
        y = int(config.RUNNER_PLAYER_Y - self.jump_height())
        if self.sliding:
            cv2.rectangle(frame, (x - 34, y - 18), (x + 34, y + 22), (70, 220, 255), -1)
            cv2.circle(frame, (x + 28, y), 18, (245, 245, 245), -1)
        else:
            cv2.rectangle(frame, (x - 28, y - 78), (x + 28, y - 12), (70, 220, 255), -1)
            cv2.circle(frame, (x, y - 98), 24, (245, 245, 245), -1)
        cv2.circle(frame, (x, y), 46, (10, 14, 18), 2)

    def _draw_obstacle(self, frame, obstacle: Obstacle):
        x = config.RUNNER_LANE_X[obstacle.lane]
        y = int(obstacle.y)
        if obstacle.kind == OBSTACLE_LOW:
            cv2.rectangle(frame, (x - 55, y - 22), (x + 55, y + 22), (50, 90, 235), -1)
            label = "JUMP"
        elif obstacle.kind == OBSTACLE_HIGH:
            cv2.rectangle(frame, (x - 70, y - 75), (x + 70, y - 45), (235, 170, 55), -1)
            label = "SLIDE"
        else:
            cv2.rectangle(frame, (x - 60, y - 65), (x + 60, y + 45), (170, 70, 225), -1)
            label = "DODGE"
        cv2.putText(frame, label, (x - 38, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_hud(self, frame, action, recognizer_status, calibrated, warning):
        lane_text = f"Lane: {self.lane_name}"
        status = "GAME OVER" if self.game_over else "RUNNING"
        cv2.putText(frame, f"Action: {action.name}", (30, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(frame, lane_text, (250, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Calibrated: {calibrated}", (430, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Score: {self.score}", (720, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(frame, f"State: {status}", (930, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(frame, recognizer_status[:46], (30, 686), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 230, 120), 2, cv2.LINE_AA)
        if warning:
            cv2.putText(frame, warning[:50], (30, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "A/D or arrows: lane  W/up: jump  S/down: slide  C: calibrate  R: restart  Esc: quit", (300, 686), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 220, 230), 1, cv2.LINE_AA)

    def _draw_preview(self, frame, camera_preview):
        x, y = 900, 78
        w, h = config.RUNNER_PREVIEW_SIZE
        cv2.rectangle(frame, (x - 4, y - 28), (x + w + 4, y + h + 4), (18, 22, 28), -1)
        cv2.putText(frame, "Pose Preview", (x + 72, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        if camera_preview is None:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (35, 40, 45), -1)
            cv2.putText(frame, "No camera", (x + 72, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
        else:
            preview = cv2.resize(camera_preview, (w, h))
            frame[y : y + h, x : x + w] = preview

    def _draw_game_over(self, frame):
        cv2.rectangle(frame, (315, 240), (865, 420), (18, 22, 28), -1)
        cv2.rectangle(frame, (315, 240), (865, 420), (220, 220, 220), 2)
        cv2.putText(frame, "GAME OVER", (455, 305), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (70, 70, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, self.message, (370, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press R to restart", (455, 392), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
