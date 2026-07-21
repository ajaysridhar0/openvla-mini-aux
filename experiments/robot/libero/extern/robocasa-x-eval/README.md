# Pinned RoboCasa-X simulator stack

This directory is an implementation detail of `barx setup-sim`.

- `repos.lock.json` pins RoboSuite, RoboCasa, RoboMimic, and MimicGen.
- `bootstrap.sh` checks out those sources and obtains kitchen assets.
- `validate.sh` verifies commits, imports, generated macro files, assets, and the BARX mug-task extension.
- `mimicgen_flip_mug.patch` adds the mug task and the missing Jaco robot registration.
- `robomimic_lazy_imports.patch` avoids loading policy-training dependencies during data generation.
- `robocasa_lazy_imports.patch` avoids loading action-conversion dependencies during state rendering.

The forks are imported directly from their pinned source trees. They are not editable-installed because their old,
mutually inconsistent package pins would replace the BARX Torch and TensorFlow environment.

Users should run `barx setup-sim`; see [`BARX_README.md`](../../../../../BARX_README.md) for the complete workflow.
