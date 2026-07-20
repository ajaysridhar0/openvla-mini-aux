
import numpy as np

def get_geomids_for_obj(env, obj_name):
    geom_ids_dict = {}
    relevant_body_names = []
    for body_name in env.sim.model.body_names:
        if body_name is not None and obj_name in body_name:
            relevant_body_names.append(body_name)
    for i in range(env.sim.model.ngeom):
        for relevant_body_name in relevant_body_names:
            if env.sim.model.geom_bodyid[i] == env.sim.model.body_name2id(relevant_body_name):
                if relevant_body_name not in geom_ids_dict:
                    geom_ids_dict[relevant_body_name] = []
                geom_ids_dict[relevant_body_name].append(i)
    return geom_ids_dict


def create_box_vertices(aabb):
    center = aabb[:3]
    half_sizes = aabb[3:]
    return np.array([
        center + [-half_sizes[0], -half_sizes[1], -half_sizes[2]],
        center + [half_sizes[0], -half_sizes[1], -half_sizes[2]],
        center + [half_sizes[0], half_sizes[1], -half_sizes[2]],
        center + [-half_sizes[0], half_sizes[1], -half_sizes[2]],
        center + [-half_sizes[0], -half_sizes[1], half_sizes[2]],
        center + [half_sizes[0], -half_sizes[1], half_sizes[2]],
        center + [half_sizes[0], half_sizes[1], half_sizes[2]],
        center + [-half_sizes[0], half_sizes[1], half_sizes[2]]
    ])


def quaternion_to_rotation_matrix(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def transform_vertices(vertices, pos, quat):
    R = quaternion_to_rotation_matrix(quat)
    return np.dot(vertices, R.T) + pos


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ]


def rotate_image_pts(image_pts, image_height, image_width, rotation_degrees=90):

    # Convert degrees to radians
    rotation_radians = np.radians(rotation_degrees)

    # Create rotation matrix for the specified angle
    rotation_matrix = np.array([
        [np.cos(rotation_radians), np.sin(rotation_radians)],
        [-np.sin(rotation_radians), np.cos(rotation_radians)]
    ])

    # Center of the image
    image_center = np.array([image_height // 2, image_width // 2])

    # Translate points to origin, rotate, then translate back
    image_pts_rotated = np.einsum('ij,...j->...i', rotation_matrix, image_pts - image_center) + image_center

    # Replace the original pts with the rotated version
    image_pts = image_pts_rotated

    # Flip pts across the vertical axis going through the middle of the image
    image_pts[..., 0] = image_width - image_pts[..., 0]
    return image_pts


def get_aabbs_for_body(env, obj_name):
    aabb_dict = {}
    geom_ids_dict = get_geomids_for_obj(env, obj_name)
    for body_name, geom_ids in geom_ids_dict.items():
        body_id = env.sim.model.body_name2id(body_name)
        body_pos = env.sim.data.body_xpos[body_id]
        body_quat = env.sim.data.body_xquat[body_id]

        min_vertix = np.array([np.inf, np.inf, np.inf])
        max_vertix = np.array([-np.inf, -np.inf, -np.inf])
        
        if len(geom_ids) == 0:
            return None
        for geom_id in geom_ids:
            aabb = env.sim.model.geom_aabb[geom_id]
            geom_pos = env.sim.model.geom_pos[geom_id]
            geom_quat = env.sim.model.geom_quat[geom_id]
            combined_pos = body_pos + transform_vertices(geom_pos.reshape(1, 3), np.zeros(3), body_quat).flatten()
            combined_quat = quaternion_multiply(body_quat, geom_quat)
            box_vertices = create_box_vertices(aabb)
            transformed_vertices = transform_vertices(box_vertices, combined_pos, combined_quat)
            min_vertix = np.minimum(min_vertix, np.min(transformed_vertices, axis=0))
            max_vertix = np.maximum(max_vertix, np.max(transformed_vertices, axis=0))
        aabb_dict[body_name] = np.array([min_vertix, max_vertix])
    return aabb_dict


def get_corners(bboxes):
    # Ensure bboxes is a numpy array
    bboxes = np.array(bboxes)
    
    # Check if the input shape is correct
    assert bboxes.shape[1:] == (2, 3), f"Expected shape (N, 2, 3), got {bboxes.shape}"
    
    # Check for NaN or infinity values
    if np.any(np.isnan(bboxes)) or np.any(np.isinf(bboxes)):
        raise ValueError("Input bboxes contain NaN or infinity values")
    
    # Create the corner combinations
    corner_indices = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    
    # Use broadcasting to compute all corners for all bboxes at once
    corners = np.zeros((bboxes.shape[0], 8, 3))
    for i in range(3):
        corners += bboxes[:, corner_indices[:, i], i][:, :, np.newaxis] * np.eye(3)[i]
    
    # Check for NaN or infinity values in the result
    if np.any(np.isnan(corners)) or np.any(np.isinf(corners)):
        raise ValueError("Computed corners contain NaN or infinity values")
    
    return corners


def extra_image_transform_robocasa(pts, image_size):
    center = np.array([image_size//2, image_size//2])

    centered_pose = pts - center

    theta = np.pi/2
    rotat_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])

    rotated_pose = (rotat_matrix @ centered_pose.T).T + center

    rotated_pose[..., 0] = image_size - rotated_pose[..., 0]

    return rotated_pose.astype(np.int32)