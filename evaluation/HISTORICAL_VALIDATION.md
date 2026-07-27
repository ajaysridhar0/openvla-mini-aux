# Historical condition validation

The paper rollouts did not save MuJoCo XML or simulator states. We therefore
validated the reconstructed condition protocol against the first policy-input
frame in retained rollout videos. This checks the visible layout, style,
fixtures, objects, robot initialization, camera, and placements; it cannot
prove equality of unobserved velocities or contact state.

## Method

The audit uses the paper evaluator's global seed 7, creates one environment,
and replays every reset sequentially from episode 1. Episode `n` uses seed
`999 + n`, followed by the original 10 no-op settling steps. The reconstructed
320×180 observation is passed through the historical imageio MP4 path, which
resizes it to 320×192, before comparison with the decoded historical frame.

This sequential replay is important. Creating a later episode directly from
its seed does not reproduce the paper condition because RoboCasa retains some
Python-side sampled scene state across hard resets.

## Spot-check results

The following checks span all four tasks, four embodiments, and seeds
1000–1004. Visual inspection confirmed matching target and distractor object
instances, fixture layout and style, camera viewpoint, robot pose, and visible
object placements.

| Task | Embodiment | Episode / seed | Layout / style | MAE | PSNR (dB) | SSIM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Flip Mug Upright | Panda | 1 / 1000 | 8 / 0 | 3.521 | 33.757 | 0.9424 |
| Flip Mug Upright | Panda | 2 / 1001 | 8 / 7 | 3.487 | 34.085 | 0.9419 |
| PnP Counter to Sink | Panda-OG | 3 / 1002 | 7 / 7 | 2.789 | 35.143 | 0.9500 |
| PnP Sink to Counter | UR5e | 4 / 1003 | 4 / 9 | 2.945 | 34.474 | 0.9562 |
| Turn On Sink Faucet | Jaco | 5 / 1004 | 7 / 2 | 2.579 | 35.594 | 0.9434 |

Repeated historical runs of the same Flip Mug condition produced first-frame
SSIM values of 0.981–0.990 with one another. The reconstructed frames are
therefore close, but not bit-identical, to the historical renderer output.

## Issues found by the audit

The comparison identified and corrected three release-cleanup regressions:

- supplying explicit arm joint positions had bypassed the paper evaluator's
  zero-height Omron torso initialization, shifting the camera vertically;
- the global NumPy seed used by robosuite arm initialization was not restored;
- the public sampling pool initially included four mug models outside the
  paper task's feasible pool, changing the selected mug instance.

The torso and camera geometry now agrees with the historical source to floating
point precision in the inspected Panda scene. The mug exclusions and calibrated
cameras are scoped to the `X*` benchmark tasks, so standard RoboCasa tasks keep
their upstream behavior.

A later failure-video review found a protocol flaw rather than a replay
regression. In Panda counter-to-sink seed 1001, the requested lemon is placed on
the counter but projects to approximately `(-91, 213)` in the 320×180 policy
image. The visible sink object is a distractor. A static audit found the same
problem in 34 of the 100 consecutive counter-to-sink seeds for every
embodiment; the other three task sets pass the center-in-frame check.

The release therefore distinguishes two protocols:

- `historical-consecutive-seeds`: reconstructs the original 1000–1099 sequence
  for provenance and is not suitable for public evaluation.
- `visible-target-v1`: scans the same candidate stream, preserves the original
  scene and object distribution, and accepts only targets visible in the policy
  camera.

This correction changes the selected counter-to-sink conditions, so results
under `visible-target-v1` must not be described as using the exact historical
100-seed set.

## Repeating the check

Given access to a retained local rollout directory, repeat the check with:

```bash
uv run --locked --no-dev python scripts/audit_historical_rollouts.py \
  --task flip_mug_upright \
  --embodiment panda \
  --rollout-dir /path/to/historical/act \
  --episodes 1 2 \
  --output /tmp/flip-mug-audit.json \
  --artifacts-dir /tmp/flip-mug-frames
```

The script deliberately replays all prerequisite episodes even when only later
episode numbers are selected.
