import unittest

from barx.names import canonicalize_prediction_keys, representations_for_method, resolve_transform_spec


class PaperNamesTest(unittest.TestCase):
    def test_method_aliases_preserve_original_transform_specs(self):
        self.assertEqual(resolve_transform_spec("No Reps"), "action")
        self.assertEqual(
            resolve_transform_spec("Joint Reps"),
            "bbox->,low_level_motion->,ee_pose_2D->,action",
        )
        self.assertEqual(
            resolve_transform_spec("ECoT"),
            "bbox->ee_pose_2D->low_level_motion->,action",
        )

    def test_individual_representation_aliases(self):
        self.assertEqual(
            resolve_transform_spec("bounding_box->,end_effector_trace->,action"),
            "bbox->,ee_pose_2D->,action",
        )
        self.assertEqual(resolve_transform_spec("bounding_box"), "bbox->,action")
        self.assertEqual(resolve_transform_spec("language_motion"), "low_level_motion->,action")
        self.assertEqual(resolve_transform_spec("end_effector_trace"), "ee_pose_2D->,action")

    def test_inference_order(self):
        self.assertEqual(
            representations_for_method("ECoT"),
            ["bbox", "ee_pose_2D", "low_level_motion"],
        )

    def test_prediction_keys_include_paper_names_and_legacy_names(self):
        output = canonicalize_prediction_keys({"bbox": {"mug": [0, 0, 1, 1]}, "action": [0] * 7})
        self.assertIn("bbox", output)
        self.assertIn("bounding_box", output)
        self.assertEqual(output["bbox"], output["bounding_box"])


if __name__ == "__main__":
    unittest.main()
