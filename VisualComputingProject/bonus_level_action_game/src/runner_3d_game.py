"""Ursina 3D three-lane runner scene and game logic."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from PIL import Image
from ursina import AmbientLight, DirectionalLight, Entity, Text, Texture, Vec3, camera, color, destroy, lerp, scene, time

from . import config
from .runner_actions import Action


LEFT_LANE = 0
CENTER_LANE = 1
RIGHT_LANE = 2
LANE_NAMES = {LEFT_LANE: "LEFT", CENTER_LANE: "CENTER", RIGHT_LANE: "RIGHT"}
OBSTACLE_LOW = "low_barrier"
OBSTACLE_HIGH = "high_bar"
OBSTACLE_FULL = "full_block"


def _display_safe(text: str, max_len: int) -> str:
    safe = text.encode("ascii", errors="ignore").decode("ascii")
    return safe[:max_len]


@dataclass
class MovingItem:
    entity: Entity
    lane: int
    kind: str
    collected: bool = False


class Runner3DGame:
    def __init__(self):
        self.rng = random.Random(11)
        self.items: list[MovingItem] = []
        self.road_segments: list[Entity] = []
        self.target_lane = config.RUNNER_START_LANE
        self.lane = config.RUNNER_START_LANE
        self.score = 0
        self.coins = 0
        self.alive_time = 0.0
        self.spawn_timer = 0.0
        self.coin_timer = 0.0
        self.jump_timer = 0.0
        self.slide_timer = 0.0
        self.started = False
        self.game_over = False
        self.message = "Calibrate pose or use keyboard"
        self.last_action = Action.NONE
        self._build_scene()
        self.reset()

    def _build_scene(self):
        scene.fog_color = color.rgb32(12, 16, 22)
        scene.fog_density = 0.018
        AmbientLight(color=color.rgba32(115, 130, 150, 255))
        sun = DirectionalLight(rotation=(45, -35, 25), color=color.rgba32(255, 248, 224, 255))
        sun.look_at(Vec3(0, -1, 2))
        camera.clear_color = color.rgb32(12, 16, 22)
        camera.position = (0, 6.2, -12.5)
        camera.rotation_x = 20
        camera.fov = 72
        Entity(model="cube", color=color.rgb32(18, 24, 31), scale=(80, 38, 1), position=(0, 15, 68))
        self.player = Entity(position=(0, 0, config.RUNNER_3D_PLAYER_Z))
        self.player_body = Entity(parent=self.player, model="cube", color=color.azure, scale=(0.9, 1.35, 0.55), position=(0, 1.05, 0))
        self.player_head = Entity(parent=self.player, model="sphere", color=color.rgb32(246, 226, 196), scale=0.52, position=(0, 2.0, 0))
        self.player_shadow = Entity(model="circle", color=color.rgba32(0, 0, 0, 90), rotation_x=90, scale=(1.15, 1.15, 1), position=(0, 0.025, 0))
        for i in range(config.RUNNER_3D_ROAD_SEGMENTS):
            z = i * config.RUNNER_3D_ROAD_SEGMENT_LENGTH - 12
            seg_color = color.rgb32(45, 53, 63) if i % 2 == 0 else color.rgb32(36, 44, 54)
            seg = Entity(model="cube", color=seg_color, scale=(12.5, 0.18, config.RUNNER_3D_ROAD_SEGMENT_LENGTH), position=(0, -0.1, z))
            self.road_segments.append(seg)
        for lane_x in config.RUNNER_3D_LANE_X:
            Entity(model="cube", color=color.rgb32(240, 210, 90), scale=(0.08, 0.06, config.RUNNER_3D_ROAD_SEGMENT_LENGTH * 7), position=(lane_x, 0.04, 42))
        for edge_x in (-5.55, 5.55):
            Entity(model="cube", color=color.rgb32(88, 102, 120), scale=(0.18, 0.55, config.RUNNER_3D_ROAD_SEGMENT_LENGTH * 7), position=(edge_x, 0.22, 42))
        self.hud = Text(text="", position=(-0.86, 0.47), origin=(-0.5, 0.5), scale=0.78, color=color.white, background=True)
        self.status_text = Text(text="", position=(-0.86, 0.425), origin=(-0.5, 0.5), scale=0.56, color=color.rgb32(120, 255, 140), background=True)
        self.prompt_text = Text(
            text="",
            position=(0, -0.02),
            origin=(0, 0),
            scale=0.9,
            color=color.white,
            background=True,
        )
        self.preview_panel = Entity(
            parent=camera.ui,
            model="quad",
            color=color.rgb32(20, 24, 30),
            scale=(0.34, 0.19),
            position=(0.63, -0.36),
        )
        self.preview_label = Text(
            text="CAMERA",
            parent=camera.ui,
            position=(0.47, -0.24),
            origin=(-0.5, 0.5),
            scale=0.65,
            color=color.white,
            background=True,
        )
        self._preview_texture: Texture | None = None

    @property
    def jumping(self) -> bool:
        return self.jump_timer > 0.0

    @property
    def sliding(self) -> bool:
        return self.slide_timer > 0.0

    @property
    def lane_name(self) -> str:
        return LANE_NAMES[self.target_lane]

    def reset(self):
        for item in self.items:
            destroy(item.entity)
        self.items.clear()
        self.target_lane = config.RUNNER_START_LANE
        self.lane = config.RUNNER_START_LANE
        self.player.position = (config.RUNNER_3D_LANE_X[self.lane], 0, config.RUNNER_3D_PLAYER_Z)
        self.player.scale = (1, 1, 1)
        self.player_body.scale = (0.9, 1.35, 0.55)
        self.player_body.position = (0, 1.05, 0)
        self.player_head.position = (0, 2.0, 0)
        self.score = 0
        self.coins = 0
        self.alive_time = 0.0
        self.spawn_timer = 0.0
        self.coin_timer = 0.3
        self.jump_timer = 0.0
        self.slide_timer = 0.0
        self.started = False
        self.game_over = False
        self.message = "Stand centered for pose calibration"
        self.last_action = Action.NONE

    def handle_action(self, action: Action):
        if action is Action.NONE:
            return
        if self.game_over:
            if action in {Action.LEFT, Action.RIGHT, Action.JUMP, Action.SLIDE}:
                return
        self.started = True
        self.last_action = action
        if action is Action.LEFT:
            self.target_lane = max(LEFT_LANE, self.target_lane - 1)
            self.message = "Lane left"
        elif action is Action.RIGHT:
            self.target_lane = min(RIGHT_LANE, self.target_lane + 1)
            self.message = "Lane right"
        elif action is Action.JUMP and not self.jumping:
            self.jump_timer = config.RUNNER_3D_JUMP_DURATION
            self.message = "Jump"
        elif action is Action.SLIDE and not self.sliding:
            self.slide_timer = config.RUNNER_3D_SLIDE_DURATION
            self.message = "Slide"

    def update(self, pose_status: str = "", calibrated: bool = False, warning: str = ""):
        dt = time.dt
        if calibrated and not self.started:
            self.started = True
            self.message = "Pose calibrated - running"
        if self.started and not self.game_over:
            self.alive_time += dt
            self.score = int(self.alive_time * 100) + self.coins * 50
            self._update_player(dt)
            self._update_world(dt)
            self._spawn(dt)
            self._check_collisions()
        self._update_camera(dt)
        self._update_hud(pose_status, calibrated, warning)

    def update_preview(self, rgb_frame):
        if rgb_frame is None:
            self.preview_panel.texture = None
            self.preview_panel.color = color.rgb32(20, 24, 30)
            self.preview_label.text = "CAMERA: waiting"
            return
        image = Image.fromarray(rgb_frame).convert("RGBA")
        flipped_bytes = image.transpose(Image.FLIP_TOP_BOTTOM).tobytes()
        if self._preview_texture is None or self._preview_texture.width != image.width or self._preview_texture.height != image.height:
            self._preview_texture = Texture(image)
            self.preview_panel.texture = self._preview_texture
        else:
            self._preview_texture._texture.setRamImageAs(flipped_bytes, "RGBA")
        self.preview_panel.color = color.white
        self.preview_label.text = "CAMERA / POSE"

    def _update_player(self, dt: float):
        target_x = config.RUNNER_3D_LANE_X[self.target_lane]
        self.player.x = lerp(self.player.x, target_x, min(1.0, dt * config.RUNNER_3D_LANE_LERP_SPEED))
        if abs(self.player.x - target_x) < 0.05:
            self.player.x = target_x
            self.lane = self.target_lane
        if self.jump_timer > 0:
            self.jump_timer = max(0.0, self.jump_timer - dt)
            progress = 1.0 - self.jump_timer / config.RUNNER_3D_JUMP_DURATION
            self.player.y = math.sin(progress * math.pi) * config.RUNNER_3D_JUMP_HEIGHT
        else:
            self.player.y = 0
        if self.slide_timer > 0:
            self.slide_timer = max(0.0, self.slide_timer - dt)
            self.player_body.scale = (0.95, 0.55, 0.55)
            self.player_body.position = (0, 0.55, 0)
            self.player_head.position = (0, 1.05, 0)
        else:
            self.player_body.scale = (0.9, 1.35, 0.55)
            self.player_body.position = (0, 1.05, 0)
            self.player_head.position = (0, 2.0, 0)
        self.player_shadow.x = self.player.x

    def _update_world(self, dt: float):
        speed = config.RUNNER_3D_WORLD_SPEED_START + self.alive_time * config.RUNNER_3D_WORLD_SPEED_GAIN
        for seg in self.road_segments:
            seg.z -= speed * dt
            if seg.z < -config.RUNNER_3D_ROAD_SEGMENT_LENGTH:
                seg.z += config.RUNNER_3D_ROAD_SEGMENT_LENGTH * len(self.road_segments)
        for item in list(self.items):
            item.entity.z -= speed * dt
            if item.entity.z < config.RUNNER_3D_DESPAWN_Z:
                destroy(item.entity)
                self.items.remove(item)

    def _spawn(self, dt: float):
        self.spawn_timer -= dt
        self.coin_timer -= dt
        if self.spawn_timer <= 0:
            self.spawn_timer = max(0.62, config.RUNNER_3D_OBSTACLE_SPAWN_SECONDS - self.alive_time * 0.006)
            lane = self.rng.choice(config.RUNNER_LANES)
            kind = self.rng.choice([OBSTACLE_LOW, OBSTACLE_HIGH, OBSTACLE_FULL])
            self._spawn_obstacle(lane, kind)
        if self.coin_timer <= 0:
            self.coin_timer = config.RUNNER_3D_COIN_SPAWN_SECONDS
            lane = self.rng.choice(config.RUNNER_LANES)
            self._spawn_coin(lane)

    def _spawn_obstacle(self, lane: int, kind: str):
        x = config.RUNNER_3D_LANE_X[lane]
        z = config.RUNNER_3D_OBSTACLE_START_Z
        if kind == OBSTACLE_LOW:
            entity = Entity(model="cube", color=color.red, scale=(1.25, 0.75, 1.0), position=(x, 0.38, z))
        elif kind == OBSTACLE_HIGH:
            entity = Entity(model="cube", color=color.orange, scale=(1.9, 0.28, 0.65), position=(x, 1.62, z))
        else:
            entity = Entity(model="cube", color=color.violet, scale=(1.55, 1.8, 1.05), position=(x, 0.9, z))
        self.items.append(MovingItem(entity=entity, lane=lane, kind=kind))

    def _spawn_coin(self, lane: int):
        x = config.RUNNER_3D_LANE_X[lane]
        z = config.RUNNER_3D_OBSTACLE_START_Z + self.rng.uniform(5, 16)
        entity = Entity(model="sphere", color=color.yellow, scale=0.42, position=(x, 1.25, z))
        self.items.append(MovingItem(entity=entity, lane=lane, kind="coin"))

    def _check_collisions(self):
        for item in list(self.items):
            if abs(item.entity.z - config.RUNNER_3D_PLAYER_Z) > config.RUNNER_3D_COLLISION_Z_WINDOW:
                continue
            if abs(item.entity.x - self.player.x) > config.RUNNER_3D_PLAYER_COLLISION_HALF_WIDTH:
                continue
            if item.kind == "coin":
                self.coins += 1
                destroy(item.entity)
                self.items.remove(item)
                continue
            if item.kind == OBSTACLE_LOW and self.player.y >= config.RUNNER_3D_LOW_CLEAR_HEIGHT:
                continue
            if item.kind == OBSTACLE_HIGH and self.sliding:
                continue
            self.game_over = True
            self.message = "Collision - press R to restart"
            return

    def _update_camera(self, dt: float):
        target = Vec3(self.player.x, 6.2, -12.5)
        camera.position = lerp(camera.position, target, min(1.0, dt * 4.5))
        camera.rotation_x = 20
        camera.rotation_y = 0

    def _update_hud(self, pose_status: str, calibrated: bool, warning: str):
        state = "GAME OVER" if self.game_over else "RUNNING"
        if not self.started and not self.game_over:
            state = "READY"
        self.hud.text = (
            f"{self.last_action.name} | {self.lane_name} | Score {self.score} | Coins {self.coins} | {state}"
        )
        pose_text = _display_safe(pose_status, 48)
        warning_text = _display_safe(warning, 40)
        self.status_text.text = f"Pose {calibrated} | {pose_text} | {warning_text}"
        if self.game_over:
            self.prompt_text.text = "GAME OVER - Press R to restart"
        elif not self.started:
            self.prompt_text.text = "Stand centered to calibrate, or press A/D/W/S"
        else:
            self.prompt_text.text = ""
