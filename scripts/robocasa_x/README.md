# RoboCasa-X conversion internals

These scripts implement `barx render` and `barx build-rlds`. The public workflow and smoke-test commands live in
[`BARX_README.md`](../../BARX_README.md).

- `render_dataset.py` replays one state HDF5 into aligned third-person and wrist observations.
- `batch_render.py` applies that converter to every `demo.hdf5` below a root.
- `build_rlds.py` writes the shared BARX RLDS schema and handles Panda versus non-Panda action layouts.
- `language_motion.py` derives the motion-language auxiliary labels.
- `visualize_evidence.py` makes optional HDF5 or RLDS evidence videos.
