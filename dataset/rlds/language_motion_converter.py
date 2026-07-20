"""Conversions to map an episode (list of dictionaries) --> language motions.

Actions are assumed to be 7-dof (single robot)

Episode format should be at minimum:
{
    "observation: ...
    "action": ...
}

"""

from typing import Dict, List
import numpy as np


class LiberoLanguageMotionEpisodeConverter:

    smooth_labels = False

    pos_thresh = np.asarray([0.125, 0.125, 0.125])  # pos
    rot_thresh = np.asarray([0.125, 0.125, 0.125])  # rot

    # operates on diff of episodes["observation/state"][-2]
    gripper_thresh = 5e-4

    # each dim mapped to the signed language (first entry is for negative)
    libero_90_pos_dim_sign_to_language = [
        # ["forward", "back"],
        ["back", "forward"], # flipped
        # ["left", "right"],
        ["right", "left"], # flipped
        ["down", "up"],
    ]

    libero_90_rot_dim_sign_to_language = [
        ["right", "left"],
        ["up", "down"],
        ["clockwise", "counterclockwise"],
    ]

    flat_thresholds = np.concatenate([pos_thresh, rot_thresh, [gripper_thresh]])
    pos_thresh, rot_thresh = np.asarray(pos_thresh), np.asarray(rot_thresh)
    
    # splitting dims along action dimension
    splits = np.cumsum([len(pos_thresh), len(rot_thresh)])
    
    def convert(self, episode: List[Dict]) -> List:
        # H x ac_dim
        action = np.stack([step["action"] for step in episode])
        if action.shape[-1] == 11:
            pose_quat_delta = action[:, :6]
            gripper = action[:, -1:] 
            action = np.concatenate([pose_quat_delta, gripper], axis=-1)
        elif action.shape[-1] == 7:
            pass
        elif action.shape[-1] == 12:
            action = action[:, :7]
        else:
            raise ValueError(f"Action dimension {action.shape[-1]} not supported")
        
        assert (
            action.shape[-1] == len(self.flat_thresholds)
        ), f"Thresholds do not match action dimension {action.shape[-1]}"

        pos, rot, gripper = np.split(action, self.splits, axis=-1)

        # ratios
        pos_div_thresh = pos / self.pos_thresh[None]
        rot_div_thresh = rot / self.rot_thresh[None]
        pos_sign = (pos > 0).astype(int)  # 0 = negative, 1 = positive
        rot_sign = (rot > 0).astype(int)  # 0 = negative, 1 = positive

        # determine the active idxs.
        is_pos_active = np.abs(pos_div_thresh) >= 1.0
        is_rot_active = np.abs(rot_div_thresh) >= 1.0

        # determine the idx ordering (large to small contribution)
        pos_sorted_idxs = np.argsort(-np.abs(pos_div_thresh), axis=-1)
        rot_sorted_idxs = np.argsort(-np.abs(rot_div_thresh), axis=-1)

        # now compute if the gripper moved
        # first get the future absolute gripper delta for each state.
        gripper_state = np.abs([step["observation"]["state"][-2] for step in episode])
        next_gripper_diff = np.concatenate([gripper_state[1:] - gripper_state[:-1], [0]])
        # next get the STARTS for each gripper event
        gripper_action_delta = np.concatenate([[0], gripper[1:, 0] - gripper[:-1, 0]])
        gripper_event_start_idxs = np.nonzero(np.abs(gripper_action_delta) > 1e-11)[0]

        # consists of tuples of (start_idx, end_idx, label)
        gripper_event_chunks_and_label = []
        for j, start_idx in enumerate(gripper_event_start_idxs):
            # determine when change in gripper state decreases below thresh, starting 4 steps after the action begins for safety
            end_idx_candidates = np.nonzero(np.abs(next_gripper_diff[start_idx+4:]) <= self.gripper_thresh)[0]

            if len(end_idx_candidates) == 0:
                end_idx = len(gripper) - 1
            else:
                # since we started counting the candidates from (start_idx + 2)
                # the -1 is because we are asking for the last timestep where the gripper event is active
                end_idx = start_idx + 4 + end_idx_candidates[0] - 1

            label = "open gripper" if gripper_action_delta[start_idx] < 0 else "close gripper"

            # get rid of overlap between subsequent gripper events.
            # this really shouldn't happen tho
            if j < len(gripper_event_start_idxs) - 1:
                end_idx = min(end_idx, gripper_event_start_idxs[j+1] - 1)
            
            # print(label, start_idx, end_idx)

            gripper_event_chunks_and_label.append(
                (start_idx, end_idx, label)
            )

        # decision logic for combining things.
        movement_labels = []
        dominant_sorted_labels = []
        for t in range(action.shape[0]):
            gripper_labels = []

            # assign gripper event, if overlapping.
            for (start, end, label) in gripper_event_chunks_and_label:
                if start <= t <= end:
                    gripper_labels.append(label)
                    break
            
            pos_labels, rot_labels = [], []
            # add labels for active pos idxs in the order of decreasing magnitude.
            for dim in pos_sorted_idxs[t]:
                if is_pos_active[t, dim]:
                    pos_labels.append(
                        self.libero_90_pos_dim_sign_to_language[dim][pos_sign[t, dim]]
                    )
            # add labels for active rot idxs in the order of decreasing magnitude
            for dim in rot_sorted_idxs[t]:
                if is_rot_active[t, dim]:
                    rot_labels.append(
                        self.libero_90_rot_dim_sign_to_language[dim][rot_sign[t, dim]]
                    )

            # if there is NO event assigned, just use the highest abs value dimension across all pos/rot
            # also add "slowly" since technically the movement is below the threshold.
            if (not gripper_labels and not pos_labels and not rot_labels):
                best_pos_dim = pos_sorted_idxs[t][0]
                best_rot_dim = rot_sorted_idxs[t][0]
                if np.abs(rot_div_thresh[t, best_rot_dim]) > np.abs(pos_div_thresh[t, best_pos_dim]):
                    rot_labels.append(
                        self.libero_90_rot_dim_sign_to_language[best_rot_dim][rot_sign[t, best_rot_dim]]
                        + " slowly"
                    )
                else:
                    pos_labels.append(
                        self.libero_90_pos_dim_sign_to_language[best_pos_dim][pos_sign[t, best_pos_dim]]
                        + " slowly"
                    )

            # one big joint movement string.
            if pos_labels:
                pos_labels[0] = "move " + pos_labels[0]
            
            if rot_labels:
                rot_labels[0] = "rotate " + rot_labels[0]

            # combine rot and pos based on which is bigger.
            if np.abs(rot_div_thresh[t, rot_sorted_idxs[t, 0]]) > np.abs(pos_div_thresh[t, pos_sorted_idxs[t, 0]]):
                movement_labels.append(
                    " and ".join(gripper_labels + rot_labels + pos_labels)
                )
                dominant_sorted_labels.append(" and ".join(gripper_labels + rot_labels[:1]))
            else:
                movement_labels.append(
                    " and ".join(gripper_labels + pos_labels + rot_labels)
                )
                dominant_sorted_labels.append(" and ".join(gripper_labels + pos_labels[:1]))

            # print(movement_labels[-1])
        
        if self.smooth_labels:
            # finally, we remove really noisy movement labels (things that only show up once)
            # fall back to the previous label in these cases.
            for t in range(1, action.shape[0] - 1):
                if (dominant_sorted_labels[t] != dominant_sorted_labels[t-1] and 
                    dominant_sorted_labels[t] != dominant_sorted_labels[t+1]):
                    movement_labels[t] = movement_labels[t-1]
        
        return movement_labels


def dedup_lm(language_motions: List[str]):
    to_keep = []
    for t in range(len(language_motions)):
        if not to_keep or language_motions[t] != to_keep[-1]:
            to_keep.append(language_motions[t])

    return to_keep