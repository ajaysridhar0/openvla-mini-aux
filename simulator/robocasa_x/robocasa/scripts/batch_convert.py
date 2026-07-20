import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("libero_dataset_states_to_obs_single.py")

# Regex pattern to match chunked HDF5 files
CHUNK_FILE_PATTERN = re.compile(r"demo.hdf5")

def find_chunk_files(root_dir, rand_cams=False):
    chunk_files = []
    if rand_cams:
        skip_file = "demo_gentex_im320_randcams.hdf5"
    else:
        skip_file = "demo_gentex_im320.hdf5"
    for dirpath, _, filenames in os.walk(root_dir):
        if skip_file in filenames:
            continue  # Skip this directory if the exclusion file exists
        for filename in filenames:
            if CHUNK_FILE_PATTERN.match(filename):
                full_path = os.path.join(dirpath, filename)
                chunk_files.append(full_path)
    return chunk_files

def run_script_on_file(hdf5_file, camera):
    cmd = [
        sys.executable, "-u",
        str(SCRIPT_PATH),
        "--dataset", hdf5_file,
        "--generative_textures",
        # "--randomize_cameras",
        "--camera_names", camera,
    ]

    prefix = f"[{os.path.split(os.path.dirname(hdf5_file))[-1]}]"

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    try:
        for line in process.stdout:
            print(f"{prefix} {line}", end="", flush=True)  # stream stdout line-by-line
    except Exception as e:
        print(f"{prefix} ERROR: {e}", flush=True)

    process.wait()
    if process.returncode == 0:
        return f"{prefix} Finished successfully"
    else:
        return f"{prefix} Error (return code {process.returncode})"

def main(root_dir, camera, max_workers=10):
    chunk_files = find_chunk_files(root_dir)
    print(f"Found {len(chunk_files)} chunk files.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_script_on_file, f, camera): f for f in chunk_files}
        for future in as_completed(futures):
            print(future.result(), flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python batch_convert.py <root_directory> <camera>")
    else:
        main(sys.argv[1], sys.argv[2])
