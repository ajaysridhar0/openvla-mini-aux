# BARX MimicGen subset

This directory contains the MimicGen components required to reproduce BARX
data generation for:

- PnP Counter to Sink
- PnP Sink to Counter
- Turn On Sink Faucet
- Flip Mug Upright

The code is derived from NVIDIA's MimicGen 1.0 release and Jensen Gao's BARX
cross-embodiment extensions at source revision
`62aca464ee14edeac48769c6be82464a7337a5d3`.

BARX changes:

- register only the RoboCasa interfaces used by BARX;
- use RoboCasa's MuJoCo-native environment wrapper instead of legacy
  `mujoco_py`;
- support RoboSuite 1.5 composite controllers and the canonical 12-D mobile
  robot action order;
- add the Flip Mug Upright configuration and environment interface;
- reconstruct model-free public episodes from portable `ep_meta`;
- add a bounded `max_attempts` generation option; and
- lazy-load the unrelated Google Drive download dependency.

The complete NVIDIA license is retained in [`LICENSE`](LICENSE). This package
and derivatives are restricted to non-commercial research or evaluation use
under that license.
