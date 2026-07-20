import h5py
import os
import re

CHUNK_FILE_PATTERN = re.compile(r"demo_gentex_im320_randcams.hdf5")

def find_chunk_files(root_dir):
    chunk_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if CHUNK_FILE_PATTERN.match(filename):
                full_path = os.path.join(dirpath, filename)
                chunk_files.append(full_path)
    return chunk_files

successes = []

root = "/iliad/u/jenseng/xembod/robocasa_xembod/data/mg"
chunk_files = find_chunk_files(root)
for chunk_file in chunk_files:
    f = h5py.File(chunk_file, "r")
    try:
        demo = f["data"]["demo_99"]
        successes.append(1)
    except:
        print(f"bad file at {chunk_file}")
        successes.append(0)
    f.close()

print(f"{sum(successes)}/{len(successes)} good files")