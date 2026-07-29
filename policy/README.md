# BARX policy code

This directory contains the MiniVLA training code and RoboCasa-X evaluator used
by BARX. Install and run them through the commands in the root README and
[`docs/experiments.md`](../docs/experiments.md).

The training commands accept the paper method names `no_reps`, `joint_reps`,
and `ecot`, plus the representation names `bounding_box`, `language_motion`,
and `end_effector_trace`.

- Training implementation: `vla-scripts/train.py`
- RoboCasa-X evaluator: `experiments/robot/robocasa_x/evaluate.py`
VQ configuration files and `vla-scripts/pretrain_vq.py` support training the
task-specific tokenizers from the simulation dataset.
