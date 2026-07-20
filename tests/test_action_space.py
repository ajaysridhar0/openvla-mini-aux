import unittest

from barx.action_space import canonicalize_action, robocasa_action, robocasa_noop_action


class ActionSpaceTest(unittest.TestCase):
    def test_panda_conversion_matches_original_converter(self):
        raw = list(range(12))
        self.assertEqual(canonicalize_action(raw, "Panda"), raw[:7])
        self.assertEqual(canonicalize_action(raw, "Panda-OG"), raw[:7])
        self.assertEqual(canonicalize_action(raw, "PandaOGGripperOmron"), raw[:7])
        self.assertEqual(canonicalize_action(raw, "PandaOGOmron"), raw[:7])
        self.assertEqual(canonicalize_action(raw, "PandaOmron"), raw[:7])

    def test_non_panda_conversion_matches_original_converter(self):
        raw = list(range(12))
        self.assertEqual(canonicalize_action(raw, "Jaco"), raw[:6] + raw[-2:-1])
        self.assertEqual(canonicalize_action(raw, "IIWAOmron"), raw[:6] + raw[-2:-1])

    def test_legacy_eleven_dimensional_action(self):
        raw = list(range(11))
        self.assertEqual(canonicalize_action(raw, "UR5e"), raw[:6] + raw[-1:])

    def test_canonical_actions_pass_through(self):
        canonical = list(range(7))
        self.assertEqual(canonicalize_action(canonical, "Kinova 3"), canonical)

    def test_evaluation_expansion_matches_original_branches(self):
        canonical = list(range(7))
        self.assertEqual(robocasa_action(canonical, "Panda"), canonical + [0.0] * 4 + [-1.0])
        self.assertEqual(
            robocasa_action(canonical, "Jaco"),
            canonical[:6] + [0.0] * 4 + canonical[6:] + [-1.0],
        )

    def test_noop_matches_original_branches(self):
        self.assertEqual(robocasa_noop_action("Panda"), [0.0] * 6 + [-1.0] + [0.0] * 4 + [-1.0])
        self.assertEqual(robocasa_noop_action("Jaco"), [0.0] * 10 + [-1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
