"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import argparse
from copy import deepcopy
import datetime
import json
import os
import time
from glob import glob

import h5py
import imageio
import mujoco
import numpy as np
import robosuite

# from robosuite import load_controller_config
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from termcolor import colored

import robocasa
import robocasa.macros as macros
from robocasa.models.fixtures import FixtureType
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format


def is_empty_input_spacemouse(action_dict):
    if (
        np.all(action_dict["right_delta"] == 0)
        and action_dict["base_mode"] == -1
        and np.all(action_dict["base"] == 0)
    ):
        return True
    return False



def disable_robot_collisions(sim):
    """
    Disable collisions for the entire robot arm (including hand/wrist) while preserving 
    only gripper finger collisions. This allows the entire arm to pass through objects 
    while still enabling grasping with the fingers.
    """
    for geom_id, geom_name in enumerate(sim.model.geom_names):
        if "robot0" in geom_name:
            # Only keep collision for gripper fingers
            if "finger" in geom_name:
                continue
            # Disable collision for everything else (arm links, wrist, palm etc)
            sim.model.geom_contype[geom_id] = 0
            sim.model.geom_conaffinity[geom_id] = 0
    
    # Force MuJoCo to update
    sim.forward()


def collect_human_trajectory(
    env,
    device,
    arm,
    env_configuration,
    mirror_actions,
    render=True,
    max_fr=None,
    print_info=True,
    prev_traj_discarded=False,
    disable_tilt=False,
    disable_robot_collisions=False,
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """
    env.reset()

    # import robosuite.utils.transform_utils as T

    # robot = env.unwrapped.robots[0]

    # # from panda
    # # target_pos = [0.36151, -1.83031112, 1.11088364] # from co-training
    # # target_pos = [0.57, -1.83031112, 1.15] # from manual adjustment
    # # target_pos = [0.57094202, -1.84775016, 1.27230109] # from ajay joints
    # target_pos = [0.57094202, -1.81, 1.27230109] # adjusted ajay

    # target_rot = np.array([[0.0, -1.0, 0.0],
    #                        [-1.0, 0.0, 0.0],
    #                        [0.0, 0.0, -1.0]])
    
    # def pad_action(pos, rot):
    #     # action = np.concatenate((pos, rot, [0] * 4, [-1])) 
    #     action = np.concatenate((pos, rot, [-1], [0] * 4, [-1])) # for panda
    #     return action

    # for i in range(100):
    #     curr_pos = robot.sim.data.site_xpos[robot.eef_site_id['right']]
    #     curr_rot = robot.sim.data.site_xmat[robot.eef_site_id['right']].reshape((3, 3))

    #     max_dpos = robot.composite_controller.get_controller("right").output_max[0]
    #     max_drot = robot.composite_controller.get_controller("right").output_max[3]

    #     curr_base_pos, curr_base_rot = robot.composite_controller.get_controller_base_pose("right")

    #     delta_position = target_pos - curr_pos
    #     delta_position = np.clip(delta_position / max_dpos, -1., 1.)

    #     # normalized delta rotation action
    #     delta_rot_mat = target_rot.dot(curr_rot.T)
    #     delta_quat = T.mat2quat(delta_rot_mat)
    #     delta_rotation = T.quat2axisangle(delta_quat)
    #     delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)

    #     # convert to action in base frame
    #     base_angle = T.quat2axisangle(T.mat2quat(curr_base_rot))[2]
    #     x_w = delta_position[0]
    #     y_w = delta_position[1]
    #     x_r = np.cos(base_angle) * x_w + np.sin(base_angle) * y_w
    #     y_r = -np.sin(base_angle) * x_w + np.cos(base_angle) * y_w
    #     delta_position[0] = x_r
    #     delta_position[1] = y_r
    #     roll_w = delta_rotation[0]
    #     pitch_w = delta_rotation[1]
    #     roll_r = np.cos(base_angle) * roll_w + np.sin(base_angle) * pitch_w
    #     pitch_r = -np.sin(base_angle) * roll_w + np.cos(base_angle) * pitch_w
    #     delta_rotation[0] = roll_r
    #     delta_rotation[1] = pitch_r

    #     action = pad_action(delta_position, delta_rotation)
    #     env.step(action)

    #     new_pos = robot.sim.data.site_xpos[robot.eef_site_id['right']]
    #     pos_diff = np.linalg.norm(new_pos - target_pos)
    #     print(f"pos diff iter {i + 1}: {pos_diff}")

    # joint = np.array([robot.sim.data.qpos[x] for x in robot._ref_joint_pos_indexes])
    # breakpoint()

    # camera_id = env.sim.model.camera_name2id(env.render_camera[0])
    # local_pos = env.sim.model.cam_pos[camera_id]
    # local_quat = env.sim.model.cam_quat[camera_id]
    # parent_body_id = env.sim.model.cam_bodyid[camera_id]
    # parent_pos = env.sim.data.xpos[parent_body_id]
    # parent_quat = env.sim.data.xquat[parent_body_id]

    # from scipy.spatial.transform import Rotation as R
    # import numpy as np
    # parent_rot = R.from_quat(parent_quat[[1, 2, 3, 0]])  # Convert (w,x,y,z) to (x,y,z,w)
    # local_rot = R.from_quat(local_quat[[1, 2, 3, 0]])
    # absolute_pos = parent_pos + parent_rot.apply(local_pos)
    # absolute_quat = (parent_rot * local_rot).as_quat()
    # absolute_quat = np.roll(absolute_quat, shift=1)
    # breakpoint()
    
    # invert
    # abs_pos = np.array([ 1.03, -2.2,  1.54])
    # abs_quat = np.array([-0.7277022350759109, -0.4037431553139996, -0.2618616934241865, -0.48874264712590476])

    # # Convert to scipy format (x, y, z, w)
    # parent_rot = R.from_quat(parent_quat[[1, 2, 3, 0]])
    # abs_rot = R.from_quat(abs_quat[[1, 2, 3, 0]])

    # # Inverse of parent rotation
    # parent_rot_inv = parent_rot.inv()

    # # Compute local position
    # local_pos = parent_rot_inv.apply(abs_pos - parent_pos)

    # # Compute local rotation
    # local_rot = (parent_rot_inv * abs_rot).as_quat()  # (x, y, z, w)
    # local_quat = np.roll(local_rot, shift=1)  # back to (w, x, y, z)
    # breakpoint()

    if disable_robot_collisions:
        disable_robot_collisions(env.sim)

    ep_meta = env.get_ep_meta()
    # print(json.dumps(ep_meta, indent=4))
    lang = ep_meta.get("lang", None)
    if print_info and lang is not None:
        print(colored(f"Instruction: {lang}", "green"))

    # print the style and layout ids
    print(colored(f"Style ID: {ep_meta['style_id']}", "blue"))
    print(colored(f"Layout ID: {ep_meta['layout_id']}", "blue"))

    # degugging: code block here to quickly test and close env
    # env.close()
    # return None, True

    if render:
        # ID = 2 always corresponds to agentview
        env.render()

    task_completion_hold_count = (
        -1
    )  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    nonzero_ac_seen = False

    # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    zero_action = np.zeros(env.action_dim)
    for _ in range(1):
        # do a dummy step thru base env to initalize things, but don't record the step
        if isinstance(env, DataCollectionWrapper):
            env.env.step(zero_action)
        else:
            env.step(zero_action)

    discard_traj = False

    # Store the initial layout information
    if hasattr(env, 'env'):  # Handle wrapped environments
        base_env = env.env
    else:
        base_env = env
    
    current_layout_id = base_env._layout_id if hasattr(base_env, '_layout_id') else None
    current_style_id = base_env._style_id if hasattr(base_env, '_style_id') else None

    env.hard_reset = False

    # Store the initial RNG state and layout/style IDs before any resets
    if hasattr(base_env, 'env'):
        # Only store RNG state if this is our first run (not after a discard)
        if not prev_traj_discarded:
            initial_rng_state = base_env.env.rng.bit_generator.state
    
    # Loop until we get a reset from the input or the task completes
    while True:
        start = time.time()

        # Set active robot
        active_robot = env.robots[device.active_robot]
        active_arm = device.active_arm

        # Get the newest action
        input_ac_dict = device.input2action(mirror_actions=mirror_actions)
        
        if disable_tilt and input_ac_dict is not None:
            input_ac_dict["right_delta"][3:5] = 0

        # If action is none, then this a reset so we should break
        if input_ac_dict is None:
            discard_traj = True
            break

        action_dict = deepcopy(input_ac_dict)

        # set arm actions
        for arm in active_robot.arms:
            controller_input_type = active_robot.part_controllers[arm].input_type
            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]
            else:
                raise ValueError

        if is_empty_input_spacemouse(action_dict):
            if not nonzero_ac_seen:
                if render:
                    env.render()
                continue
        else:
            nonzero_ac_seen = True

        # Maintain gripper state for each robot but only update the active robot with action
        env_action = [
            robot.create_action_vector(all_prev_gripper_actions[i])
            for i, robot in enumerate(env.robots)
        ]
        env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
        env_action = np.concatenate(env_action)

        # Run environment step
        obs, _, _, _ = env.step(env_action)
        if render:
            env.render()

        # print(np.array([robot.sim.data.qpos[x] for x in robot._ref_joint_pos_indexes]))

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

        # limit frame rate if necessary
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

        # # Print robot qpos for each robot
        # for i, robot in enumerate(env.robots):
        #     print(f"Robot {i} qpos:", robot._joint_positions)

        # print(len(env_action))
        # print("env_action", env_action)

    if nonzero_ac_seen and hasattr(env, "ep_directory"):
        ep_directory = env.ep_directory
    else:
        ep_directory = None

    # cleanup for end of data collection episodes
    env.close()

    return ep_directory, discard_traj


def gather_demonstrations_as_hdf5(directory, out_dir, env_info, start_seed, excluded_episodes=None):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.
    The strucure of the hdf5 file is as follows.
    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected
        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration
        demo2 (group)
        ...
    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, f"demo_seed{start_seed}.hdf5")
    print("Saving hdf5 to", hdf5_path)
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        # print("Processing {} ...".format(ep_directory))
        if (excluded_episodes is not None) and (ep_directory in excluded_episodes):
            # print("\tExcluding this episode!")
            continue

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        actions_abs = []
        # success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                if "actions_abs" in ai:
                    actions_abs.append(ai["actions_abs"])
            # success = success or dic["successful"]

        if len(states) == 0:
            continue

        # # Add only the successful demonstration to dataset
        # if success:

        # print("Demonstration is successful and has been saved")
        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # store ep meta as an attribute
        ep_meta_path = os.path.join(directory, ep_directory, "ep_meta.json")
        if os.path.exists(ep_meta_path):
            with open(ep_meta_path, "r") as f:
                ep_meta = f.read()
            ep_data_grp.attrs["ep_meta"] = ep_meta

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        if len(actions_abs) > 0:
            print(np.array(actions_abs).shape)
            ep_data_grp.create_dataset("actions_abs", data=np.array(actions_abs))

        # else:
        #     pass
        #     # print("Demonstration is unsuccessful and has NOT been saved")

    print("{} successful demos so far".format(num_eps))

    if num_eps == 0:
        f.close()
        return

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["robocasa_version"] = robocasa.__version__
    grp.attrs["robosuite_version"] = robosuite.__version__
    grp.attrs["mujoco_version"] = mujoco.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()

    return hdf5_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robocasa.models.assets_root, "demonstrations_private"),
    )
    parser.add_argument("--environment", type=str, default="Kitchen")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="PandaOmron",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--obj_groups",
        type=str,
        nargs="+",
        default=None,
        help="In kitchen environments, either the name of a group to sample object from or path to an .xml file",
    )

    parser.add_argument(
        "--obj_init_range",
        type=float,
        nargs="+",
        default=None
    )

    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be, eg. 'NONE' or 'WHOLE_BODY_IK', etc. Or path to controller json file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="spacemouse",
        choices=["keyboard", "keyboardmobile", "spacemouse", "dummy"],
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=4.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=4.0,
        help="How much to scale rotation user inputs",
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--renderer", type=str, default="mjviewer", choices=["mjviewer", "mujoco"]
    )
    parser.add_argument(
        "--max_fr", default=30, type=int, help="If specified, limit the frame rate"
    )

    parser.add_argument("--layout", type=int, nargs="+", default=-1)
    parser.add_argument(
        "--style", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
    )
    parser.add_argument("--generative_textures", action="store_true")
    parser.add_argument("--use_distractors", action="store_true")
    parser.add_argument(
        "--repeat_env",
        action="store_true",
        help="Flag to indicate if the environment should be repeated"
    )
    parser.add_argument(
        "--disable_tilt",
        action="store_true",
        help="Flag to disable tilt controls when collecting demonstrations"
    )

    parser.add_argument(
        "--start_seed",
        type=int,
        default=0
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Arguments
    args = parse_args()

    # Get controller config
    # controller_config = load_controller_config(default_controller=args.controller)
    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots if isinstance(args.robots, str) else args.robots[0],
    )

    if controller_config["type"] == "WHOLE_BODY_MINK_IK":
        # mink-speicific import. requires installing mink
        from robosuite.examples.third_party_controller.mink_controller import (
            WholeBodyMinkIK,
        )

    env_name = args.environment

    # Create argument configuration
    config = {
        "env_name": env_name,
        "robots": args.robots,
        "controller_configs": controller_config,
        "use_distractors": args.use_distractors,
        "initialization_noise": None,
        "obj_init_range": args.obj_init_range
    }

    if args.generative_textures is True:
        config["generative_textures"] = "100p"

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in env_name:
        config["env_configuration"] = args.config

    # Mirror actions if using a kitchen environment
    if env_name in ["Lift"]:  # add other non-kitchen tasks here
        if args.obj_groups is not None:
            print(
                "Specifying 'obj_groups' in non-kitchen environment does not have an effect."
            )
        mirror_actions = False
        if args.camera is None:
            args.camera = "agentview"
        # special logic: "free" camera corresponds to Null camera
        elif args.camera == "free":
            args.camera = None
    else:
        mirror_actions = True

        config["layout_and_style_ids"] = [(l, s) for l in args.layout for s in args.style if (l, s) != (8, 3)]

        ### update config for kitchen envs ###
        if args.obj_groups is not None:
            config.update({"obj_groups": args.obj_groups})
        if args.camera is None:
            args.camera = "robot0_agentview_center"
            # args.camera = "robot0_agentview_right"
        # special logic: "free" camera corresponds to Null camera
        elif args.camera == "free":
            args.camera = None

        # config["translucent_robot"] = True
        # config["randomize_cameras"] = True

        # by default use obj instance split A
        config["obj_instance_split"] = "A"
        # config["obj_instance_split"] = None
        # config["obj_registries"] = ("aigen",)

    seed = args.start_seed
    # Create environment
    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer=args.renderer,
        rng=np.random.default_rng(seed),
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime("%Y-%m-%d-%H-%M-%S")

    if not args.debug:
        # wrap the environment with data collection wrapper
        tmp_directory = "/tmp/{}".format(time_str)
        env = DataCollectionWrapper(env, tmp_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
            vendor_id=macros.SPACEMOUSE_VENDOR_ID,
            product_id=macros.SPACEMOUSE_PRODUCT_ID,
        )
    else:
        raise ValueError

    # make a new timestamped directory
    new_dir = os.path.join(args.directory, time_str)
    os.makedirs(new_dir)

    excluded_eps = []

    prev_traj_discarded = False

    # collect demonstrations
    started = False
    while True:
        if args.repeat_env:
            env.unwrapped.rng = np.random.default_rng(seed)
            
        ep_directory, discard_traj = collect_human_trajectory(
            env,
            device,
            args.arm,
            args.config,
            mirror_actions,
            render=(args.renderer != "mjviewer"),
            max_fr=args.max_fr,
            prev_traj_discarded=prev_traj_discarded,
            disable_tilt=args.disable_tilt,
        )

        if not discard_traj:
            seed += 1
            started=True

        print("Keep traj?", not discard_traj)
        prev_traj_discarded = discard_traj

        if not args.debug:
            if discard_traj and ep_directory is not None:
                excluded_eps.append(ep_directory.split("/")[-1])
            hdf5_path = gather_demonstrations_as_hdf5(
                tmp_directory, new_dir, env_info, args.start_seed, excluded_episodes=excluded_eps
            )
            if started:
                convert_to_robomimic_format(hdf5_path)

        # if args.repeat_env:
        #     start = time.time()
        #     # Properly clean up the old environment
        #     if hasattr(env, 'close'):
        #         env.close()
        #     del env
        #     # Create new environment
        #     env = robosuite.make(
        #         **config,
        #         has_renderer=True,
        #         has_offscreen_renderer=False,
        #         render_camera=args.camera,
        #         ignore_done=True,
        #         use_camera_obs=False,
        #         control_freq=20,
        #         renderer=args.renderer,
        #         rng=np.random.default_rng(seed),
        #     )

        #     # Wrap this with visualization wrapper
        #     env = VisualizationWrapper(env)

        #     # Add the data collection wrapper if not in debug mode
        #     if not args.debug:
        #         env = DataCollectionWrapper(env, tmp_directory)
            
        #     end = time.time()

        #     print(f"Env creation time: {end - start} seconds")
