import unittest

from barx.action_space import canonicalize_action, robocasa_action, robocasa_noop_action


class ActionSpaceTest(unittest.TestCase):
    def test_normalized_simulator_action_uses_first_seven_values(self):
        raw = list(range(12))
        self.assertEqual(canonicalize_action(raw), raw[:7])

    def test_canonical_actions_pass_through(self):
        canonical = list(range(7))
        self.assertEqual(canonicalize_action(canonical), canonical)

    def test_evaluation_expansion_uses_normalized_layout(self):
        canonical = list(range(7))
        self.assertEqual(robocasa_action(canonical), canonical + [0.0] * 4 + [-1.0])

    def test_noop_uses_normalized_layout(self):
        self.assertEqual(robocasa_noop_action(), [0.0] * 6 + [-1.0] + [0.0] * 4 + [-1.0])

    def test_legacy_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            canonicalize_action(list(range(11)))


if __name__ == "__main__":
    unittest.main()
