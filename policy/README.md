# BARX policy code

This directory contains the MiniVLA-derived training stack and the RoboCasa-X
evaluator used by BARX. Use the repository-root uv project as the shared
environment.

Public method names (`no_reps`, `joint_reps`, `ecot`) and representation names
(`bounding_box`, `language_motion`, `end_effector_trace`) are resolved by the
root `barx` compatibility package. Stored RLDS field names remain unchanged.

- Training implementation: `vla-scripts/train.py`
- RoboCasa-X evaluator: `experiments/robot/robocasa_x/evaluate.py`
- Paper workflow: `../docs/experiments.md`

VQ configuration files and `vla-scripts/pretrain_vq.py` support training the
task-specific tokenizers from the simulation dataset. Checkpoint paths are
ignored by Git so locally trained artifacts remain local.
