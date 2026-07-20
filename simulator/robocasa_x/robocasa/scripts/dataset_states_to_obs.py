"""
Script to extract observations from low-dimensional simulation states in a robocasa dataset.
Adapted from robomimic's dataset_states_to_obs.py script.
"""
import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
import multiprocessing
import queue
import time
import traceback
import torch
import wandb

import robocasa.utils.robomimic.robomimic_tensor_utils as TensorUtils
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
import robocasa.utils.robomimic.robomimic_dataset_utils as DatasetUtils
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
from notebooks.utils import get_aabbs_for_body, get_corners, extra_image_transform_robocasa
from functools import partial

# from robomimic.utils.log_utils import log_warning


def extract_trajectory(
    env,
    initial_state,
    states,
    actions,
    done_mode,
    add_datagen_info=False,
    add_aux_info=True,
    process_num=0,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a
            success state. If 1, done is 1 at the end of each trajectory.
            If 2, do both.
    """
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state)

    ep_meta = json.loads(initial_state["ep_meta"])
    # hack: add the cam configs in, since it's been modified
    ep_meta["cam_configs"] = deepcopy(env.env._cam_configs)
    initial_state["ep_meta"] = json.dumps(ep_meta, indent=4)

    traj = dict(
        obs=[],
        next_obs=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        # actions_abs=[],
        states=np.array(states),
        initial_state_dict=initial_state,
        datagen_info=[],
    )

    if add_aux_info:
        traj['eef_2d_trace'] = []
        traj['obj_main_bbox'] = []
        traj['low_level_motion'] = []


    # TODO: add aux info here
    world_to_camera_transform = get_camera_transform_matrix(
        env.env.sim,
        "robot0_agentview_center",
        env.env.camera_heights[0],
        env.env.camera_widths[0],
    )

    project_to_camera_space = partial(
        project_points_from_world_to_camera, 
        world_to_camera_transform=world_to_camera_transform,
        camera_height=env.env.camera_heights[0],
        camera_width=env.env.camera_widths[0]
    )

    extra_transform = partial(
        extra_image_transform_robocasa,
        image_size=env.env.camera_widths[0],
    )

    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(traj_len):
        obs = deepcopy(env.reset_to({"states": states[t]}))

        # extract datagen info
        if add_datagen_info:
            datagen_info = env.base_env.get_datagen_info(action=actions[t])
        else:
            datagen_info = {}

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # get the absolute action
        # action_abs = env.base_env.convert_rel_to_abs_action(actions[t])

        # collect transition
        traj["obs"].append(obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["datagen_info"].append(datagen_info)

        if add_aux_info:
            traj['eef_2d_trace'].append(extra_transform(project_to_camera_space(obs['robot0_eef_pos'])))
            bbox_coords = get_aabbs_for_body(env.env, 'obj_main')['obj_main']
            corners = get_corners(bbox_coords[None])[0] # 8, 3
            bbox_2d_corners = project_to_camera_space(corners) # 8, 2
            bbox_2d = np.zeros(4,)
            bbox_2d[:2] = extra_transform(np.min(bbox_2d_corners, axis=0))
            bbox_2d[2:] = extra_transform(np.max(bbox_2d_corners, axis=0))
            traj['obj_main_bbox'].append(bbox_2d)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["datagen_info"] = TensorUtils.list_of_flat_dict_to_dict_of_list(
        traj["datagen_info"]
    )

    if add_aux_info:
        # TODO: add aux info here
        # if process_num == 0:  # Only debug first process
        #     from remote_pdb import set_trace
        #     set_trace(port=4444)  # You can choose any available port
        traj['eef_2d_trace'] = np.stack(traj['eef_2d_trace'])
        traj['obj_main_bbox'] = np.stack(traj['obj_main_bbox'])
        

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj


""" The process that writes over the generated files to memory """


def write_traj_to_file(
    args, output_path, total_samples, total_run, processes, mul_queue
):
    f = h5py.File(args.dataset, "r")
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    start_time = time.time()
    num_processed = 0

    # Initialize wandb
    wandb.init(
        project="dataset-processing",
        name=f"process-{os.path.basename(args.dataset)}",
        config={
            "input_dataset": args.dataset,
            "output_path": output_path,
            "num_processes": processes,
            "camera_names": args.camera_names,
            "camera_height": args.camera_height,
            "camera_width": args.camera_width,
        }
    )

    try:
        while (total_run.value < (processes)) or not mul_queue.empty():
            if not mul_queue.empty():
                num_processed = num_processed + 1
                item = mul_queue.get()
                ep = item[0]
                traj = item[1]
                process_num = item[2]
                try:
                    ep_data_grp = data_grp.create_group(ep)
                    ep_data_grp.create_dataset(
                        "actions", data=np.array(traj["actions"])
                    )
                    ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
                    ep_data_grp.create_dataset(
                        "rewards", data=np.array(traj["rewards"])
                    )
                    ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
                    # ep_data_grp.create_dataset(
                    #     "actions_abs", data=np.array(traj["actions_abs"])
                    # )
                    
                    for k in traj["obs"]:
                        if args.no_compress:
                            ep_data_grp.create_dataset(
                                "obs/{}".format(k), data=np.array(traj["obs"][k])
                            )
                        else:
                            ep_data_grp.create_dataset(
                                "obs/{}".format(k),
                                data=np.array(traj["obs"][k]),
                                compression="gzip",
                            )
                        if args.include_next_obs:
                            if args.no_compress:
                                ep_data_grp.create_dataset(
                                    "next_obs/{}".format(k),
                                    data=np.array(traj["next_obs"][k]),
                                )
                            else:
                                ep_data_grp.create_dataset(
                                    "next_obs/{}".format(k),
                                    data=np.array(traj["next_obs"][k]),
                                    compression="gzip",
                                )

                    # if "datagen_info" in traj:
                    #     print('✅' + str(traj["datagen_info"].keys()))
                    #     for k in traj["datagen_info"]:
                    #         print(k)
                    #         ep_data_grp.create_dataset(
                    #             "datagen_info/{}".format(k),
                    #             data=np.array(traj["datagen_info"][k]),
                    #         )
                    
                    # copy datagen dict (if applicable)
                    # if process_num == 0:  # Only debug first process
                    #     from remote_pdb import set_trace
                    #     set_trace(port=4444)  # You can choose any available port
                    if "data/{}/datagen_info".format(ep) in f:
                        datagen_dict = f["data/{}/datagen_info".format(ep)]
                        print(f"Process {process_num}: Datagen dict keys: {list(datagen_dict.keys())}")
                        for k in datagen_dict:
                            k_str = str(k) if not isinstance(k, (str, bytes)) else k
                            print(f"Process {process_num}: Key {k} -> {k_str}, Type: {type(k)}")
                            try:
                                # Check if it's a group or dataset
                                if isinstance(datagen_dict[k], h5py.Group):
                                    print(f"Process {process_num}: {k} is a group")
                                    # Create a group in the output file
                                    group = ep_data_grp.create_group(f"datagen_info/{k_str}")
                                    # Copy all datasets from this group
                                    for subk in datagen_dict[k]:
                                        data = datagen_dict[k][subk][()]
                                        group.create_dataset(subk, data=np.array(data))
                                else:
                                    data = datagen_dict[k][()]
                                    print(f"Process {process_num}: Data type: {type(data)}")
                                    ep_data_grp.create_dataset(
                                        "datagen_info/{}".format(k_str),
                                        data=np.array(data),
                                    )
                            except Exception as e:
                                print(f"Process {process_num}: Error with key {k}: {str(e)}")
                                raise e

                    # copy aux info
                    if 'eef_2d_trace' in traj:
                        ep_data_grp.create_dataset(
                            'eef_2d_trace',
                            data=np.array(traj['eef_2d_trace'])
                        )
                    if 'obj_main_bbox' in traj:
                        ep_data_grp.create_dataset(
                            'obj_main_bbox',
                            data=np.array(traj['obj_main_bbox'])
                        )

                    # copy action dict (if applicable)
                    if "data/{}/action_dict".format(ep) in f:
                        action_dict = f["data/{}/action_dict".format(ep)]
                        for k in action_dict:
                            ep_data_grp.create_dataset(
                                "action_dict/{}".format(k),
                                data=np.array(action_dict[k][()]),
                            )

                    # episode metadata
                    ep_data_grp.attrs["model_file"] = traj["initial_state_dict"][
                        "model"
                    ]  # model xml for this episode
                    ep_data_grp.attrs["ep_meta"] = traj["initial_state_dict"][
                        "ep_meta"
                    ]  # ep meta data for this episode
                    # if "ep_meta" in f["data/{}".format(ep)].attrs:
                    #     ep_data_grp.attrs["ep_meta"] = f["data/{}".format(ep)].attrs["ep_meta"]
                    ep_data_grp.attrs["num_samples"] = traj["actions"].shape[
                        0
                    ]  # number of transitions in this episode

                    total_samples.value += traj["actions"].shape[0]

                    # Log metrics to wandb
                    elapsed_time = time.time() - start_time
                    wandb.log({
                        "demos_processed": num_processed,
                        "transitions_processed": total_samples.value,
                        "processing_rate": num_processed / elapsed_time,
                        "transitions_per_second": total_samples.value / elapsed_time,
                        "percent_complete": (num_processed / len(f["data"])) * 100,
                    })

                    print(
                        "ep {}: wrote {} transitions to group {} at process {} with {} finished. Datagen rate: {:.2f} sec/demo".format(
                            num_processed,
                            ep_data_grp.attrs["num_samples"],
                            ep,
                            process_num,
                            total_run.value,
                            (time.time() - start_time) / num_processed,
                        )
                    )

                except Exception as e:
                    print("++" * 50)
                    print(
                        f"Error at Process {process_num} on episode {ep} with \n\n {e}"
                    )
                    print("++" * 50)
                    raise Exception("Write out to file has failed")

    except KeyboardInterrupt:
        print("Control C pressed. Closing File and ending \n\n\n\n\n\n\n")

    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples.value
    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
    )
    print("total processes end {}".format(total_run.value))
    data_grp.attrs["env_args"] = json.dumps(
        env.serialize(), indent=4
    )  # environment info
    print("Wrote {} total samples to {}".format(total_samples.value, output_path))

    f_out.close()
    f.close()

    DatasetUtils.extract_action_dict(dataset=output_path)
    DatasetUtils.make_demo_ids_contiguous(dataset=output_path)
    for num_demos in [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        75,
        80,
        90,
        100,
        125,
        150,
        200,
        250,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1500,
        2000,
        2500,
        3000,
        4000,
        5000,
        10000,
    ]:
        DatasetUtils.filter_dataset_size(
            output_path,
            num_demos=num_demos,
        )

    print("Writing has finished")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Final wandb logging
    wandb.log({
        "total_demos_processed": num_processed,
        "total_transitions": total_samples.value,
        "total_time": elapsed_time,
        "final_processing_rate": num_processed / elapsed_time,
    })
    wandb.finish()

    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    return


# runs multiple trajectory. If there has been an unrecoverable error, the system puts the current work back into the queue and exits
def extract_multiple_trajectories(
    process_num, current_work_array, work_queue, lock, args2, num_finished, mul_queue
):
    try:
        extract_multiple_trajectories_with_error(
            process_num, current_work_array, work_queue, lock, args2, mul_queue
        )
    except Exception as e:
        work_queue.put(current_work_array[process_num])
        print("*>*" * 50)
        print("Error process num {}:".format(process_num))
        print(e)
        print(traceback.format_exc())
        print("*>*" * 50)
        print()

    num_finished.value = num_finished.value + 1


def retrieve_new_index(process_num, current_work_array, work_queue, lock):
    with lock:
        if work_queue.empty():
            return -1
        try:
            tmp = work_queue.get(False)
            current_work_array[process_num] = tmp
            return tmp
        except queue.Empty:
            return -1


def extract_multiple_trajectories_with_error(
    process_num, current_work_array, work_queue, lock, args, mul_queue
):
    # create environment to use for data processing

    if False and args.add_datagen_info:
        import mimicgen.utils.file_utils as MG_FileUtils

        env_meta = MG_FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    else:
        env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    if args.generative_textures:
        env_meta["env_kwargs"]["generative_textures"] = "100p"
    if args.randomize_cameras:
        env_meta["env_kwargs"]["randomize_cameras"] = True
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        reward_shaping=args.shaped,
    )

    start_time = time.time()

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[: args.n]

    ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
    while (not work_queue.empty()) and (ind != -1):
        try:
            # print("Running {} index".format(ind))
            ep = demos[ind]

            # prepare initial state to reload from
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
            initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get(
                "ep_meta", None
            )

            # extract obs, rewards, dones
            actions = f["data/{}/actions".format(ep)][()]

            traj = extract_trajectory(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                done_mode=args.done_mode,
                add_datagen_info=args.add_datagen_info,
                process_num=process_num,
            )

            # maybe copy reward or done signal from source file
            if args.copy_rewards:
                traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            if args.copy_dones:
                traj["dones"] = f["data/{}/dones".format(ep)][()]

            ep_grp = f["data/{}".format(ep)]

            states = ep_grp["states"][()]
            initial_state = dict(states=states[0])
            initial_state["model"] = ep_grp.attrs["model_file"]
            initial_state["ep_meta"] = ep_grp.attrs.get("ep_meta", None)

            # store transitions

            # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
            #            consistent as well
            # print("(process {}): ADD TO QUEUE index {}".format(process_num, ind))
            mul_queue.put([ep, traj, process_num])

            ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
        except Exception as e:
            print("_" * 50)
            print("Process {}:".format(process_num))
            print("Error processing demo index {}: {}".format(ind, e))
            print(traceback.format_exc())
            print("_" * 50)
            del env
            env = EnvUtils.create_env_for_data_processing(  # when it errors, it like blows up the environment for some reason
                env_meta=env_meta,
                camera_names=args.camera_names,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                reward_shaping=args.shaped,
            )

    f.close()
    print("Process {} finished".format(process_num))


def dataset_states_to_obs_multiprocessing(args):
    # create environment to use for data processing

    # output file in same directory as input file
    output_name = args.output_name
    if output_name is None:
        if len(args.camera_names) == 0:
            output_name = os.path.basename(args.dataset)[:-5] + "_ld.hdf5"
        else:
            image_suffix = str(args.camera_width)
            image_suffix = (
                image_suffix + "_randcams" if args.randomize_cameras else image_suffix
            )
            if args.generative_textures:
                output_name = os.path.basename(args.dataset)[
                    :-5
                ] + "_gentex_im{}.hdf5".format(image_suffix)
            else:
                output_name = os.path.basename(args.dataset)[:-5] + "_im{}.hdf5".format(
                    image_suffix
                )

    output_path = os.path.join(os.path.dirname(args.dataset), output_name)

    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    f = h5py.File(args.dataset, "r")
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f["mask/{}".format(args.filter_key)])
        ]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    if args.n is not None:
        demos = demos[: args.n]

    num_demos = len(demos)
    f.close()

    env_meta = DatasetUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    num_processes = 1 if args.debug else args.num_procs

    index = multiprocessing.Value("i", 0)
    lock = multiprocessing.Lock()
    total_samples_shared = multiprocessing.Value("i", 0)
    num_finished = multiprocessing.Value("i", 0)
    mul_queue = multiprocessing.Queue()
    work_queue = multiprocessing.Queue()
    for index in range(num_demos):
        work_queue.put(index)
    current_work_array = multiprocessing.Array("i", num_processes)
    processes = []
    for i in range(num_processes):
        process = multiprocessing.Process(
            target=extract_multiple_trajectories,
            args=(
                i,
                current_work_array,
                work_queue,
                lock,
                args,
                num_finished,
                mul_queue,
            ),
        )
        processes.append(process)

    process1 = multiprocessing.Process(
        target=write_traj_to_file,
        args=(
            args,
            output_path,
            total_samples_shared,
            num_finished,
            num_processes,
            mul_queue,
        ),
    )
    processes.append(process1)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    print("Finished Multiprocessing")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        help="name of output hdf5 dataset",
    )

    parser.add_argument(
        "--filter_key",
        type=str,
        help="filter key for input dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped",
        action="store_true",
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=128,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=128,
        help="(optional) width of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards",
        action="store_true",
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones",
        action="store_true",
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to include next obs in dataset
    parser.add_argument(
        "--include-next-obs",
        action="store_true",
        help="(optional) include next obs in dataset",
    )

    # flag to disable compressing observations with gzip option in hdf5
    parser.add_argument(
        "--no_compress",
        action="store_true",
        help="(optional) disable compressing observations with gzip option in hdf5",
    )

    parser.add_argument(
        "--num_procs",
        type=int,
        default=5,
        help="number of parallel processes for extracting image obs",
    )

    parser.add_argument(
        "--add_datagen_info",
        action="store_true",
        help="(optional) add datagen info (used for mimicgen)",
    )

    parser.add_argument("--generative_textures", action="store_true")

    parser.add_argument("--randomize_cameras", action="store_true")

    parser.add_argument("--debug", action="store_true", help="run with single process for debugging")

    # Add wandb-related arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="dataset-processing",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="wandb entity (username or team name)",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="disable wandb logging",
    )

    args = parser.parse_args()

    
    # Initialize wandb if enabled
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )
    
    dataset_states_to_obs_multiprocessing(args)
