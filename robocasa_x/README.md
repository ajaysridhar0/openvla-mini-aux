# RoboCasa-X runtime

This directory contains the RoboCasa runtime modified for the BARX simulation
experiments. Use the repository-root uv project as the shared environment.

The public release entry points are the four paper tasks:

- `XPnPCounterToSink` (PnP Counter to Sink)
- `XPnPSinkToCounter` (PnP Sink to Counter)
- `XTurnOnSinkFaucet` (Turn On Sink Faucet)
- `XFlipMugUpright` (Flip Mug Upright)

The corresponding original RoboCasa tasks remain registered and usable. The
`X` prefix makes the paper-specific object variability, robot pose, and
initialization rules explicit instead of silently replacing upstream behavior.
Public launchers select the matching robot, gripper, and calibrated camera as
one embodiment configuration.

The retained RoboCasa core has only the compatibility extension points needed
by those variants: injectable robot offsets and initial joint poses, deterministic
RNG/episode-metadata replay, per-instance object exclusions, and capture of the
processed MuJoCo XML. Their defaults preserve upstream task sampling. The
shared RoboCasa-X controller loader provides one Omron action order without a
robosuite fork; paper-specific cameras and fixed-base joints are installed only
by the `X*` task mixin.

The supported workflow starts from the rendered, annotated HDF5 files and uses
the four `X*` tasks for evaluation. Asset download and macro setup utilities
are available under `robocasa/scripts/`.

See the root [installation](../docs/installation.md),
[data](../docs/data.md), and [experiment](../docs/experiments.md)
documentation for the supported workflow.

RoboCasa retains its upstream MIT license in `LICENSE`; BARX-specific changes
are summarized in the root `THIRD_PARTY_NOTICES.md`.
