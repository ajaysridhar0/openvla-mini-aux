# BARX policy code

This directory contains the MiniVLA-derived training stack and the RoboCasa-X
evaluator used by BARX. Install it through the repository-root uv project; do
not create a second environment from this directory.

Public method names (`no_reps`, `joint_reps`, `ecot`) and representation names
(`bounding_box`, `language_motion`, `end_effector_trace`) are resolved by the
root `barx` compatibility package. Stored RLDS field names remain unchanged.

- Training implementation: `vla-scripts/train.py`
- RoboCasa-X evaluator: `experiments/robot/robocasa_x/evaluate.py`
- Paper workflow: `../docs/experiments.md`

VQ configuration files are included, but their `checkpoints/model.pt` files
are intentionally absent from this release. Re-train them from the released
data with `vla-scripts/pretrain_vq.py` or wait for a later checkpoint release.
