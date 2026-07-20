import h5py
import numpy as np
import os
import glob
import tqdm
import sys
def filter_noop_actions(input_path, output_path, threshold=1e-2):
    """
    Filter out individual timesteps where the L1 norm of actions[:, :6] is below a threshold.
    Creates a new HDF5 file with the filtered data, preserving compression settings.
    """
    total_timesteps = 0
    total_filtered = 0
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        # Copy top-level attributes
        for key, value in f_in.attrs.items():
            f_out.attrs[key] = value
        data_group = f_in['data']
        out_data_group = f_out.create_group('data')
        for key, value in data_group.attrs.items():
            out_data_group.attrs[key] = value
        for demo_name in data_group.keys():
            demo_group = data_group[demo_name]
            actions = demo_group['actions'][:]
            # Calculate L1 norms for the position/rotation components
            l1_norms = np.sum(np.abs(actions[:, :6]), axis=1)
            valid_indices = np.where(l1_norms >= threshold)[0]
            if len(valid_indices) == 0:
                print(f"Warning: Demo {demo_name} has no valid actions above threshold")
                continue
            total_timesteps += len(actions)
            total_filtered += len(valid_indices)
            percent_kept = (len(valid_indices) / len(actions)) * 100
            print(f"Demo {demo_name}: Original timesteps: {len(actions)}, "
                  f"Filtered timesteps: {len(valid_indices)} ({percent_kept:.1f}% kept)")
            out_demo_group = out_data_group.create_group(demo_name)
            for key, value in demo_group.attrs.items():
                out_demo_group.attrs[key] = value
            def copy_filtered_group(src_group, dst_group, valid_indices):
                """Helper function to recursively copy and filter groups"""
                for name, item in src_group.items():
                    if isinstance(item, h5py.Dataset):
                        filtered_data = item[:][valid_indices]
                        dst_group.create_dataset(
                            name,
                            data=filtered_data,
                            compression=item.compression,
                            compression_opts=item.compression_opts,
                            dtype=item.dtype
                        )
                    elif isinstance(item, h5py.Group):
                        subgroup = dst_group.create_group(name)
                        # Copy attributes
                        for key, value in item.attrs.items():
                            subgroup.attrs[key] = value
                        # Recursively copy subgroups
                        copy_filtered_group(item, subgroup, valid_indices)
            # Copy and filter all content in the demo group
            copy_filtered_group(demo_group, out_demo_group, valid_indices)
        # Copy remaining groups
        for name, item in f_in.items():
            if name != 'data':
                if isinstance(item, h5py.Dataset):
                    f_out.create_dataset(
                        name,
                        data=item[:],
                        compression=item.compression,
                        compression_opts=item.compression_opts,
                        dtype=item.dtype
                    )
                elif isinstance(item, h5py.Group):
                    out_group = f_out.create_group(name)
                    copy_filtered_group(item, out_group, slice(None))  # Copy everything for non-data groups
    # Print overall statistics
    total_percent_kept = (total_filtered / total_timesteps) * 100 if total_timesteps > 0 else 0
    print(f"\nOverall statistics:")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Timesteps after filtering: {total_filtered}")
    print(f"Percentage kept: {total_percent_kept:.1f}%")
    # Print file sizes
    original_size = os.path.getsize(input_path) / (1024 * 1024)  # Convert to MB
    filtered_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
    print(f"\nFile sizes:")
    print(f"Original file: {original_size:.1f} MB")
    print(f"Filtered file: {filtered_size:.1f} MB")
if __name__ == "__main__":
    import glob
    import os
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="/Users/jensengao/local_docs/iliad/xembod/robocasa_xembod/data/PandaOmron/PnPCounterToSink/carrot/2025-03-27-17-22-59/demo_seed0_im224_libero.hdf5")
    args = parser.parse_args()
    # Get array job ID and total number of jobs from command line arguments
    # job_id = int(sys.argv[1])
    # n_jobs = int(sys.argv[2])

    job_id = 0
    n_jobs = 1

    data_dir =  "/iliad/u/jenseng/xembod/robocasa_xembod/datasets/v0.1/single_stage/kitchen_pnp"
    file_paths = glob.glob(os.path.join(data_dir, "**", "demo_im224_libero.hdf5"), recursive=True)
    # file_paths = [args.file_path]
    # Split files among jobs
    files_per_job = len(file_paths) // n_jobs
    start_idx = job_id * files_per_job
    end_idx = start_idx + files_per_job if job_id < n_jobs - 1 else len(file_paths)
    job_file_paths = file_paths[start_idx:end_idx]
    print(f"Job {job_id}/{n_jobs} processing {len(job_file_paths)} files")
    for file_path in tqdm.tqdm(job_file_paths):
        filter_noop_actions(file_path, file_path.replace(".hdf5", "_filter_noop_simple.hdf5"))