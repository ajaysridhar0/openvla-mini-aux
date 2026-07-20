import h5py
import imageio
import os
from PIL import Image

# path = "/iliad/u/jenseng/xembod/robocasa_xembod/data/mg/IIWAOmron/PnPCounterToSink/100demos_seed20/demo_gentex_im320_randcams.hdf5"
# path = "/iliad/u/jenseng/xembod/robocasa_xembod/data/mg/Kinova3Omron/PnPCounterToSink/100demos_seed10/demo_gentex_im320_randcams.hdf5"
path = "/iliad/u/jenseng/xembod/robocasa_xembod/data/mg/UR5eOmron/PnPCounterToSink/100demos_seed0/demo_gentex_im320_randcams.hdf5"

file =  h5py.File(path, "r")
save_path = "pics/cts_ur5e/"
os.makedirs(save_path, exist_ok=True)

for demo_name in list(file['data'].keys())[:20]:
    demo = file['data'][demo_name]
    obses = demo['obs']['agentview_rgb'][:]
    
    im = Image.fromarray(obses[0])
    im.save(save_path + f"{demo_name}.jpg")
    