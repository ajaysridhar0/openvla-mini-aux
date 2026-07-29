# RoboCasa-X

This directory contains the RoboCasa-X simulation environments used by BARX.
The benchmark has four tasks:

- `XPnPCounterToSink` (PnP Counter to Sink)
- `XPnPSinkToCounter` (PnP Sink to Counter)
- `XTurnOnSinkFaucet` (Turn On Sink Faucet)
- `XFlipMugUpright` (Flip Mug Upright)

The `X` prefix distinguishes the BARX task settings from the standard RoboCasa
tasks, which remain available under their original names. Each BARX embodiment
selects its robot, gripper, and calibrated camera together.

Use the root README to download the kitchen assets and run an evaluation.
Additional asset utilities are available under `robocasa/scripts/`.

See the root [installation](../docs/installation.md),
[data](../docs/data.md), and [experiment](../docs/experiments.md)
documentation for the supported workflow.
