"""Convert a canonical 7-D BARX episode into language-motion annotations."""

import numpy as np


class LanguageMotionEpisodeConverter:
    smooth_labels = False

    pos_thresh = np.asarray([0.125, 0.125, 0.125])
    rot_thresh = np.asarray([0.125, 0.125, 0.125])

    gripper_thresh = 5e-4

    # The first label in each pair corresponds to a negative action.
    pos_dim_sign_to_language = [
        ["back", "forward"],
        ["right", "left"],
        ["down", "up"],
    ]

    rot_dim_sign_to_language = [
        ["right", "left"],
        ["up", "down"],
        ["clockwise", "counterclockwise"],
    ]

    flat_thresholds = np.concatenate([pos_thresh, rot_thresh, [gripper_thresh]])
    pos_thresh, rot_thresh = np.asarray(pos_thresh), np.asarray(rot_thresh)
    splits = np.cumsum([len(pos_thresh), len(rot_thresh)])

    def convert(self, episode: list[dict]) -> list[str]:
        action = np.stack([step["action"] for step in episode])
        if action.shape[-1] != len(self.flat_thresholds):
            raise ValueError(
                f"Expected canonical 7-D BARX actions, received {action.shape[-1]}-D"
            )

        pos, rot, gripper = np.split(action, self.splits, axis=-1)

        # ratios
        pos_div_thresh = pos / self.pos_thresh[None]
        rot_div_thresh = rot / self.rot_thresh[None]
        pos_sign = (pos > 0).astype(int)  # 0 = negative, 1 = positive
        rot_sign = (rot > 0).astype(int)  # 0 = negative, 1 = positive

        # Determine active dimensions and order them by relative magnitude.
        is_pos_active = np.abs(pos_div_thresh) >= 1.0
        is_rot_active = np.abs(rot_div_thresh) >= 1.0
        pos_sorted_idxs = np.argsort(-np.abs(pos_div_thresh), axis=-1)
        rot_sorted_idxs = np.argsort(-np.abs(rot_div_thresh), axis=-1)

        # Find gripper events from the future absolute gripper-state delta.
        gripper_state = np.abs([step["observation"]["state"][-2] for step in episode])
        next_gripper_diff = np.concatenate(
            [gripper_state[1:] - gripper_state[:-1], [0]]
        )
        gripper_action_delta = np.concatenate([[0], gripper[1:, 0] - gripper[:-1, 0]])
        gripper_event_start_idxs = np.nonzero(np.abs(gripper_action_delta) > 1e-11)[0]

        # Tuples of (start index, end index, label).
        gripper_event_chunks_and_label = []
        for j, start_idx in enumerate(gripper_event_start_idxs):
            # Start four steps after the action begins to avoid transient motion.
            end_idx_candidates = np.nonzero(
                np.abs(next_gripper_diff[start_idx + 4 :]) <= self.gripper_thresh
            )[0]

            if len(end_idx_candidates) == 0:
                end_idx = len(gripper) - 1
            else:
                end_idx = start_idx + 4 + end_idx_candidates[0] - 1

            label = (
                "open gripper"
                if gripper_action_delta[start_idx] < 0
                else "close gripper"
            )

            # Prevent overlap between successive gripper events.
            if j < len(gripper_event_start_idxs) - 1:
                end_idx = min(end_idx, gripper_event_start_idxs[j + 1] - 1)

            gripper_event_chunks_and_label.append((start_idx, end_idx, label))

        movement_labels = []
        dominant_sorted_labels = []
        for t in range(action.shape[0]):
            gripper_labels = []

            for start, end, label in gripper_event_chunks_and_label:
                if start <= t <= end:
                    gripper_labels.append(label)
                    break

            pos_labels, rot_labels = [], []
            for dim in pos_sorted_idxs[t]:
                if is_pos_active[t, dim]:
                    pos_labels.append(
                        self.pos_dim_sign_to_language[dim][pos_sign[t, dim]]
                    )
            for dim in rot_sorted_idxs[t]:
                if is_rot_active[t, dim]:
                    rot_labels.append(
                        self.rot_dim_sign_to_language[dim][rot_sign[t, dim]]
                    )

            # Describe the largest sub-threshold movement as slow.
            if not gripper_labels and not pos_labels and not rot_labels:
                best_pos_dim = pos_sorted_idxs[t][0]
                best_rot_dim = rot_sorted_idxs[t][0]
                if np.abs(rot_div_thresh[t, best_rot_dim]) > np.abs(
                    pos_div_thresh[t, best_pos_dim]
                ):
                    rot_labels.append(
                        self.rot_dim_sign_to_language[best_rot_dim][
                            rot_sign[t, best_rot_dim]
                        ]
                        + " slowly"
                    )
                else:
                    pos_labels.append(
                        self.pos_dim_sign_to_language[best_pos_dim][
                            pos_sign[t, best_pos_dim]
                        ]
                        + " slowly"
                    )

            if pos_labels:
                pos_labels[0] = "move " + pos_labels[0]
            if rot_labels:
                rot_labels[0] = "rotate " + rot_labels[0]

            if np.abs(rot_div_thresh[t, rot_sorted_idxs[t, 0]]) > np.abs(
                pos_div_thresh[t, pos_sorted_idxs[t, 0]]
            ):
                movement_labels.append(
                    " and ".join(gripper_labels + rot_labels + pos_labels)
                )
                dominant_sorted_labels.append(
                    " and ".join(gripper_labels + rot_labels[:1])
                )
            else:
                movement_labels.append(
                    " and ".join(gripper_labels + pos_labels + rot_labels)
                )
                dominant_sorted_labels.append(
                    " and ".join(gripper_labels + pos_labels[:1])
                )

        if self.smooth_labels:
            for t in range(1, action.shape[0] - 1):
                if (
                    dominant_sorted_labels[t] != dominant_sorted_labels[t - 1]
                    and dominant_sorted_labels[t] != dominant_sorted_labels[t + 1]
                ):
                    movement_labels[t] = movement_labels[t - 1]

        return movement_labels


def dedup_lm(language_motions: list[str]) -> list[str]:
    to_keep = []
    for label in language_motions:
        if not to_keep or label != to_keep[-1]:
            to_keep.append(label)

    return to_keep
