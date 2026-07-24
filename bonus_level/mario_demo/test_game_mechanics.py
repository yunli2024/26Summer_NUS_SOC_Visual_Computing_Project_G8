import time
import unittest

from mario_camera_demo import CameraPlatformDemo, GROUND_Y, MAX_LIVES, Player


def bare_demo() -> CameraPlatformDemo:
    demo = CameraPlatformDemo.__new__(CameraPlatformDemo)
    demo.player = Player(grounded=True)
    demo.finished = False
    demo.score = 0
    demo.lives = MAX_LIVES
    demo.checkpoint_x = 420.0
    demo.checkpoint_y = 426.0
    demo.input_frozen_until = 0.0
    demo.jump_buffer_until = 0.0
    demo.buffered_jump_height = 0.22
    demo.camera_x = 0.0
    demo.message = ""
    demo.message_until = 0.0
    demo.checkpoint_marker_x = 1980.0
    demo.checkpoint_activated = False
    demo.goal_x = 3740.0
    return demo


class GameMechanicsTests(unittest.TestCase):
    def test_single_level_contains_varied_mechanics_but_few_enemies(self) -> None:
        demo = bare_demo()
        demo._build_level()
        self.assertEqual(len(demo.enemies), 2)
        self.assertGreaterEqual(len(demo.pipes), 3)
        self.assertGreaterEqual(len(demo.moving_platforms), 2)
        self.assertTrue(any(block.kind == "question" for block in demo.blocks))
        self.assertTrue(any(block.kind == "brick" for block in demo.blocks))
        self.assertTrue(any(block.contents == "mushroom" for block in demo.blocks))

    def test_blocks_have_enough_clearance_to_hit_from_below(self) -> None:
        demo = bare_demo()
        demo._build_level()
        for block in demo.blocks:
            supporting_tops = [
                y for x, y, width, _ in demo.platforms
                if y > block.y + 40.0 and x < block.x + 40.0 and x + width > block.x
            ]
            self.assertTrue(supporting_tops)
            clearance = min(supporting_tops) - (block.y + 40.0)
            self.assertGreaterEqual(clearance, demo.player.height)

    def test_main_route_obstacles_fit_base_jump_envelope(self) -> None:
        demo = bare_demo()
        demo._build_level()
        ground = sorted(
            (x, x + width) for x, y, width, height in demo.platforms
            if y == GROUND_Y and height >= 60.0
        )
        gaps = [next_start - previous_end for (_, previous_end), (next_start, _) in zip(ground, ground[1:])]
        self.assertLessEqual(max(gaps), 120.0)
        base_jump_height = 620.0 ** 2 / (2.0 * 1280.0)
        self.assertLessEqual(max(pipe.height for pipe in demo.pipes), base_jump_height)

    def test_mushroom_block_spawns_powerup(self) -> None:
        demo = bare_demo()
        demo._build_level()
        block = next(block for block in demo.blocks if block.contents == "mushroom")
        demo._activate_block(block)
        self.assertTrue(block.used)
        self.assertEqual(len(demo.powerups), 1)

    def test_hitting_question_block_from_below_activates_it(self) -> None:
        demo = bare_demo()
        demo._build_level()
        block = next(block for block in demo.blocks if block.contents == "mushroom")
        demo.player.x = block.x + 1.0
        demo.player.y = block.y + 45.0
        demo.player.vy = -300.0
        demo._update_player_vertical(0.04)
        self.assertTrue(block.used)
        self.assertEqual(len(demo.powerups), 1)
        self.assertGreaterEqual(demo.player.y, block.y + 40.0)

    def test_question_block_plays_bump_sound_even_after_use(self) -> None:
        demo = bare_demo()
        demo._build_level()
        played = []
        demo._play_sound = played.append
        block = next(block for block in demo.blocks if block.contents == "mushroom")
        demo._activate_block(block)
        demo._activate_block(block)
        self.assertEqual(played.count("bump"), 2)

    def test_thin_platform_is_one_way_but_catches_falling_player(self) -> None:
        demo = bare_demo()
        demo._build_level()
        platform = next(platform for platform in demo.platforms if platform[3] == 22.0)
        x, y, width, _ = platform

        # Moving sideways through its edge is allowed.
        demo.player.x = x - demo.player.width - 2.0
        demo.player.y = y + 2.0
        demo._move_player_horizontal(12.0)
        self.assertGreater(demo.player.x + demo.player.width, x)

        # Rising through its underside is allowed.
        demo.player.x = x + width * 0.5
        demo.player.y = y + 27.0
        demo.player.vy = -300.0
        demo._update_player_vertical(0.04)
        self.assertLess(demo.player.y, y + 22.0)

        # Falling onto its top still produces a landing.
        demo.player.y = y - demo.player.height - 2.0
        demo.player.vy = 100.0
        demo._update_player_vertical(0.02)
        self.assertTrue(demo.player.grounded)
        self.assertAlmostEqual(demo.player.y + demo.player.height, y)

    def test_powerup_absorbs_one_enemy_hit(self) -> None:
        demo = bare_demo()
        demo.player.powered = True
        lives_before = demo.lives
        demo._hurt_player("Hit an enemy", time.perf_counter())
        self.assertEqual(demo.lives, lives_before)
        self.assertFalse(demo.player.powered)
        self.assertGreater(demo.player.invincible_until, time.perf_counter())

    def test_powered_player_can_break_brick(self) -> None:
        demo = bare_demo()
        demo._build_level()
        brick = next(block for block in demo.blocks if block.kind == "brick")
        demo._activate_block(brick)
        self.assertTrue(brick.active)
        demo.player.powered = True
        demo._activate_block(brick)
        self.assertFalse(brick.active)

    def test_pipe_blocks_horizontal_movement(self) -> None:
        demo = bare_demo()
        demo._build_level()
        pipe = demo.pipes[0]
        demo.player.x = pipe.x - demo.player.width - 2.0
        demo.player.y = GROUND_Y - demo.player.height
        demo._move_player_horizontal(12.0)
        self.assertLessEqual(demo.player.x + demo.player.width, pipe.x)

    def test_player_facing_follows_last_horizontal_input(self) -> None:
        demo = bare_demo()
        demo._update_facing(-1.0)
        self.assertEqual(demo.player.facing, -1)
        demo._update_facing(0.0)
        self.assertEqual(demo.player.facing, -1)
        demo._update_facing(1.0)
        self.assertEqual(demo.player.facing, 1)

    def test_base_jump_reaches_normal_platforms(self) -> None:
        demo = bare_demo()
        demo._request_jump(0.22)
        self.assertEqual(demo.player.vy, -620.0)
        theoretical_height = demo.player.vy ** 2 / (2.0 * 1280.0)
        self.assertGreaterEqual(theoretical_height, 150.0)

    def test_normal_jump_is_silent(self) -> None:
        demo = bare_demo()
        played = []
        demo._play_sound = played.append
        demo._request_jump(0.22)
        self.assertEqual(played, [])

    def test_checkpoint_plays_save_sound_once(self) -> None:
        demo = bare_demo()
        played = []
        demo._play_sound = played.append
        demo.player.x = demo.checkpoint_marker_x
        self.assertTrue(demo._activate_checkpoint(time.perf_counter()))
        self.assertFalse(demo._activate_checkpoint(time.perf_counter()))
        self.assertEqual(played, ["checkpoint"])

    def test_stronger_real_jump_gets_more_speed(self) -> None:
        demo = bare_demo()
        demo._request_jump(0.50)
        self.assertLess(demo.player.vy, -620.0)
        self.assertGreaterEqual(demo.player.vy, -720.0)

    def test_jump_requested_just_before_landing_is_buffered(self) -> None:
        demo = bare_demo()
        demo.player.grounded = False
        demo._request_jump(0.50)
        self.assertGreater(demo.jump_buffer_until, time.perf_counter())
        demo.player.grounded = True
        consumed = demo._consume_jump_buffer(demo.jump_buffer_until - 0.01)
        self.assertTrue(consumed)
        self.assertFalse(demo.player.grounded)
        self.assertLess(demo.player.vy, -620.0)

    def test_expired_jump_buffer_does_not_auto_jump(self) -> None:
        demo = bare_demo()
        demo.player.grounded = False
        demo._request_jump()
        expiry = demo.jump_buffer_until
        demo.player.grounded = True
        consumed = demo._consume_jump_buffer(expiry + 0.01)
        self.assertFalse(consumed)
        self.assertTrue(demo.player.grounded)
        self.assertEqual(demo.player.vy, 0.0)

    def test_fall_respawns_at_safe_checkpoint(self) -> None:
        demo = bare_demo()
        demo.player.x = 730.0
        demo.player.y = 700.0
        demo.player.grounded = False
        before = time.perf_counter()
        demo._lose_life("Missed the platform")
        self.assertEqual(demo.lives, MAX_LIVES - 1)
        self.assertEqual(demo.player.x, 420.0)
        self.assertEqual(demo.player.y, 426.0)
        self.assertTrue(demo.player.grounded)
        self.assertGreaterEqual(demo.input_frozen_until, before + 0.55)


if __name__ == "__main__":
    unittest.main()
