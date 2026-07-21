"""Convert BARX 7-DoF actions into the released low-level motion labels."""

from typing import ClassVar, Dict, List

import numpy as np


class LanguageMotionConverter:
    """Reproduce the language-motion heuristic used for RoboCasa-X RLDS data."""

    position_thresholds = np.asarray([0.125, 0.125, 0.125])
    rotation_thresholds = np.asarray([0.125, 0.125, 0.125])
    gripper_threshold = 5e-4
    position_labels: ClassVar = [["back", "forward"], ["right", "left"], ["down", "up"]]
    rotation_labels: ClassVar = [["right", "left"], ["up", "down"], ["clockwise", "counterclockwise"]]

    def convert(self, episode: List[Dict]) -> List[str]:
        actions = np.stack([step["action"] for step in episode])
        if actions.shape[-1] == 11:
            actions = np.concatenate([actions[:, :6], actions[:, -1:]], axis=-1)
        elif actions.shape[-1] == 12:
            actions = actions[:, :7]
        elif actions.shape[-1] != 7:
            raise ValueError(f"Unsupported action dimension: {actions.shape[-1]}")

        position, rotation, gripper = np.split(actions, [3, 6], axis=-1)
        position_ratio = position / self.position_thresholds[None]
        rotation_ratio = rotation / self.rotation_thresholds[None]
        position_order = np.argsort(-np.abs(position_ratio), axis=-1)
        rotation_order = np.argsort(-np.abs(rotation_ratio), axis=-1)
        position_active = np.abs(position_ratio) >= 1
        rotation_active = np.abs(rotation_ratio) >= 1

        gripper_state = np.abs([step["observation"]["state"][-2] for step in episode])
        next_gripper_diff = np.concatenate([np.diff(gripper_state), [0]])
        gripper_action_delta = np.concatenate([[0], np.diff(gripper[:, 0])])
        event_starts = np.flatnonzero(np.abs(gripper_action_delta) > 1e-11)
        gripper_events = []
        for event_index, start in enumerate(event_starts):
            candidates = np.flatnonzero(np.abs(next_gripper_diff[start + 4 :]) <= self.gripper_threshold)
            end = len(gripper) - 1 if not len(candidates) else start + 3 + candidates[0]
            if event_index + 1 < len(event_starts):
                end = min(end, event_starts[event_index + 1] - 1)
            label = "open gripper" if gripper_action_delta[start] < 0 else "close gripper"
            gripper_events.append((start, end, label))

        labels = []
        for timestep in range(len(actions)):
            gripper_labels = [label for start, end, label in gripper_events if start <= timestep <= end][:1]
            position_labels = [
                self.position_labels[dim][int(position[timestep, dim] > 0)]
                for dim in position_order[timestep]
                if position_active[timestep, dim]
            ]
            rotation_labels = [
                self.rotation_labels[dim][int(rotation[timestep, dim] > 0)]
                for dim in rotation_order[timestep]
                if rotation_active[timestep, dim]
            ]
            if not gripper_labels and not position_labels and not rotation_labels:
                best_position = position_order[timestep, 0]
                best_rotation = rotation_order[timestep, 0]
                if abs(rotation_ratio[timestep, best_rotation]) > abs(position_ratio[timestep, best_position]):
                    rotation_labels.append(
                        self.rotation_labels[best_rotation][int(rotation[timestep, best_rotation] > 0)] + " slowly"
                    )
                else:
                    position_labels.append(
                        self.position_labels[best_position][int(position[timestep, best_position] > 0)] + " slowly"
                    )
            if position_labels:
                position_labels[0] = "move " + position_labels[0]
            if rotation_labels:
                rotation_labels[0] = "rotate " + rotation_labels[0]
            rotation_dominates = abs(rotation_ratio[timestep, rotation_order[timestep, 0]]) > abs(
                position_ratio[timestep, position_order[timestep, 0]]
            )
            ordered = rotation_labels + position_labels if rotation_dominates else position_labels + rotation_labels
            labels.append(" and ".join(gripper_labels + ordered))
        return labels


def deduplicate_adjacent(labels: List[str]) -> List[str]:
    """Keep the first label from each consecutive run."""
    output = []
    for label in labels:
        if not output or output[-1] != label:
            output.append(label)
    return output
