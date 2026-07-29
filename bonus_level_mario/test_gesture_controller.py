import unittest

import numpy as np

from gesture_controller import GestureController


def upper_pose() -> tuple[np.ndarray, np.ndarray]:
    points = np.full((17, 2), np.nan, dtype=np.float32)
    points[0] = (320, 95)
    points[5], points[6] = (280, 175), (360, 175)
    points[7], points[8] = (265, 235), (375, 235)
    points[9], points[10] = (255, 305), (385, 305)
    valid = np.zeros(17, dtype=bool)
    valid[[0, 5, 6, 7, 8, 9, 10]] = True
    return points, valid


class HandGestureControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = GestureController(confirmation_frames=2)
        self.points, self.valid = upper_pose()

    def confirm(self, pose: np.ndarray, start: float = 0.0):
        first = self.controller.update(pose, self.valid, start)
        second = self.controller.update(pose, self.valid, start + 0.2)
        return first, second

    def test_neutral_works_without_lower_body(self) -> None:
        state = self.controller.update(self.points, self.valid, 0.0)
        self.assertTrue(state.ready)
        self.assertEqual(state.label, "NEUTRAL")

    def test_left_hand_controls_left(self) -> None:
        pose = self.points.copy()
        pose[7], pose[9] = (245, 180), (200, 185)
        _, state = self.confirm(pose)
        self.assertLess(state.horizontal, -0.5)
        self.assertEqual(state.label, "LEFT")

    def test_right_hand_controls_right(self) -> None:
        pose = self.points.copy()
        pose[8], pose[10] = (395, 180), (440, 185)
        _, state = self.confirm(pose)
        self.assertGreater(state.horizontal, 0.5)
        self.assertEqual(state.label, "RIGHT")

    def test_raised_right_hand_controls_right_without_wide_extension(self) -> None:
        pose = self.points.copy()
        pose[8], pose[10] = (375, 160), (390, 145)
        _, state = self.confirm(pose)
        self.assertGreater(state.horizontal, 0.5)
        self.assertEqual(state.label, "RIGHT")

    def test_both_hands_up_trigger_one_jump(self) -> None:
        pose = self.points.copy()
        pose[9], pose[10] = (285, 120), (355, 120)
        first, second = self.confirm(pose)
        third = self.controller.update(pose, self.valid, 0.4)
        self.assertFalse(first.jump)
        self.assertTrue(second.jump)
        self.assertFalse(third.jump)
        self.assertEqual(third.label, "HANDS UP")

    def test_asymmetric_near_shoulder_hands_still_trigger_jump(self) -> None:
        pose = self.points.copy()
        pose[9], pose[10] = (285, 160), (355, 178)
        first, second = self.confirm(pose)
        self.assertFalse(first.jump)
        self.assertTrue(second.jump)

    def test_hands_resting_at_shoulder_height_do_not_trigger_jump(self) -> None:
        pose = self.points.copy()
        pose[9], pose[10] = (285, 175), (355, 175)
        first, second = self.confirm(pose)
        self.assertFalse(first.jump)
        self.assertFalse(second.jump)

    def test_hands_together_trigger_crouch(self) -> None:
        pose = self.points.copy()
        pose[9], pose[10] = (308, 230), (332, 230)
        first, second = self.confirm(pose)
        self.assertFalse(first.crouching)
        self.assertTrue(second.crouching)
        self.assertEqual(second.label, "CROUCH")

    def test_running_direction_is_preserved_during_jump(self) -> None:
        left = self.points.copy()
        left[7], left[9] = (245, 180), (200, 185)
        self.confirm(left)
        jump_pose = self.points.copy()
        jump_pose[9], jump_pose[10] = (285, 120), (355, 120)
        self.confirm(jump_pose, 0.6)
        jump_state = self.controller.update(jump_pose, self.valid, 1.0)
        self.assertTrue(jump_state.jump)
        self.assertLess(jump_state.horizontal, 0.0)

    def test_single_unstable_frame_does_not_stop_movement(self) -> None:
        left = self.points.copy()
        left[7], left[9] = (245, 180), (200, 185)
        self.confirm(left)
        state = self.controller.update(self.points, self.valid, 0.6)
        self.assertLess(state.horizontal, 0.0)

    def test_bent_arm_does_not_trigger_horizontal_movement(self) -> None:
        pose = self.points.copy()
        pose[9] = (200, 185)
        _, state = self.confirm(pose)
        self.assertEqual(state.horizontal, 0.0)
        self.assertEqual(state.label, "NEUTRAL")

    def test_relaxed_extended_arm_triggers_movement(self) -> None:
        pose = self.points.copy()
        pose[7], pose[9] = (250, 200), (200, 185)
        _, state = self.confirm(pose)
        self.assertLess(state.horizontal, -0.5)
        self.assertEqual(state.label, "LEFT")

    def test_single_wrist_spike_does_not_trigger_movement(self) -> None:
        controller = GestureController(confirmation_frames=3)
        controller.update(self.points, self.valid, 0.0)
        spike = self.points.copy()
        spike[7], spike[9] = (245, 180), (200, 185)
        spike_state = controller.update(spike, self.valid, 0.1)
        settled_state = controller.update(self.points, self.valid, 0.2)
        self.assertEqual(spike_state.horizontal, 0.0)
        self.assertEqual(settled_state.horizontal, 0.0)


if __name__ == "__main__":
    unittest.main()
