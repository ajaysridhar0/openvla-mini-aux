import unittest

import numpy as np

from dataset.rlds.language_motion_converter import LanguageMotionEpisodeConverter


class LanguageMotionConverterTest(unittest.TestCase):
    def setUp(self):
        self.converter = LanguageMotionEpisodeConverter()

    @staticmethod
    def episode(action_dimension: int) -> list[dict]:
        return [
            {
                "action": np.zeros(action_dimension, dtype=np.float32),
                "observation": {"state": np.zeros(8, dtype=np.float32)},
            }
            for _ in range(2)
        ]

    def test_canonical_actions_are_converted(self):
        labels = self.converter.convert(self.episode(7))
        self.assertEqual(len(labels), 2)

    def test_legacy_actions_are_rejected(self):
        for action_dimension in (11, 12):
            with self.subTest(action_dimension=action_dimension):
                with self.assertRaisesRegex(ValueError, "canonical 7-D"):
                    self.converter.convert(self.episode(action_dimension))


if __name__ == "__main__":
    unittest.main()
