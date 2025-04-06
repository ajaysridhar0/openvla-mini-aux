"""
xla_train.py

TPU (PyTorch-XLA) wrapper script for training Vision-Language-Action (VLA) Policies, built on top of pretrained VLMs,
trained using mixtures of the Open-X Embodiment dataset. Performs training in PyTorch XLA, using the XLA implementation
of Fully-Sharded Data Parallel (FSDP) to run distributed across TPU Pods.

TODO (kpertsch) :: Might be good to add install / TPU setup instructions to README?

Note =>> This script runs the explicit `xla_multiprocessing.spawn()` wrapping the logic in `vla-scripts/xla_train.py`
"""

import os

import torch_xla.distributed.xla_multiprocessing as xmp

# Set "TPU/XLA Backend" as environment variable =>> lets us control native PyTorch vs. PyTorch XLA if both installed!
os.environ["PRISMATIC_USE_TPU"] = "true"


# Note that we defer the `train` import until *after* we've initialized the XLA context!
def xla_train(_device_id: int) -> None:
    from train import train

    train()


if __name__ == "__main__":
    xmp.spawn(xla_train, args=())
