# RoboCasa-X runtime

This directory contains the RoboCasa runtime modified for the BARX
simulation experiments. Install it through the repository-root uv project;
do not create a separate environment here.

Only the four paper tasks are public release entry points:

- `BARXPnPCounterToSink` (PnP Counter to Sink)
- `BARXPnPSinkToCounter` (PnP Sink to Counter)
- `BARXTurnOnSinkFaucet` (Turn On Sink Faucet)
- `BARXFlipMugUpright` (Flip Mug Upright)

The corresponding original RoboCasa tasks remain registered and usable. The
`BARX` prefix makes the paper-specific object variability, robot pose, and
initialization rules explicit instead of silently replacing upstream behavior.
Public launchers select the matching robot, gripper, and calibrated camera as
one embodiment configuration.

The repository intentionally omits demonstration collection, state rendering,
MimicGen generation, general RoboCasa demos, and upstream documentation. The
released HDF5 files are already rendered and annotated. Dataset playback,
asset download, and macro setup utilities remain under `robocasa/scripts/`.

See the root [installation](../docs/installation.md),
[data](../docs/data.md), and [experiment](../docs/experiments.md)
documentation for the supported workflow.

RoboCasa retains its upstream MIT license in `LICENSE`; BARX-specific changes
are summarized in the root `THIRD_PARTY_NOTICES.md`.
