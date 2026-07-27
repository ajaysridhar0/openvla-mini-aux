# Frozen evaluation conditions

BARX evaluates every method on the same 100 post-settling simulator states for
each task/embodiment pair. Each bundle under
`conditions/<task>/<embodiment>.*` stores the episode metadata, processed
MuJoCo XML, and settled state with integrity hashes.

The public release uses the `visible-target-v1` protocol. Pick-and-place
targets are sampled from RoboCasa `obj_set1`, instance split `A`:
`apple`, `banana`, `can`, `carrot`, `cucumber`, `lemon`, `orange`, and
`sponge`. Generation rejects a candidate unless the instruction, target
category, source fixture, and target object agree and at least 25 target
segmentation pixels are visible in the policy camera.

The original evaluator used consecutive seeds 1000–1099. That reconstructed
historical set contains valid object identities, but 34 counter-to-sink targets
are outside the policy camera for every embodiment. It is retained as a
documented historical protocol, not used by the public release. The complete
machine-readable result is in
[`HISTORICAL_VISIBILITY_AUDIT.json`](HISTORICAL_VISIBILITY_AUDIT.json). The
corrected protocol scans candidates sequentially from seed 1000 until it
collects 100 visible conditions, so accepted seed IDs are not necessarily
consecutive.

The checked-in bundles scan through seed 1152, reject 53 candidates, and use
the same 100 accepted seeds for all six embodiments. The smallest accepted
target occupies 81 rendered pixels.

Historical paper runs did not retain MuJoCo states. These bundles are
reconstructed from the paper snapshot, not extracted from original rollouts.
RoboCasa also retains Python-side scene state across resets, so generation must
scan candidates sequentially rather than construct a later condition directly.

Generate one bundle after installing the RoboCasa assets:

```bash
uv run --locked --no-dev python scripts/generate_eval_conditions.py \
  --task pnp_counter_to_sink \
  --embodiment panda
```

The public `scripts/evaluate.py` launcher loads these bundles by default and
fails before model inference if a bundle is absent, corrupt, semantically
invalid, or places the target outside the camera. After restoring each state,
it also checks the rendered target-pixel count before requesting an action.

Restore and hash-check every condition in one generated bundle with:

```bash
uv run --locked --no-dev python scripts/verify_eval_conditions.py \
  --task pnp_counter_to_sink \
  --embodiment panda
```

See [historical validation](HISTORICAL_VALIDATION.md) for the video-based spot
check against the retained paper rollouts and its limitations. If those private
rollouts are available, the check can be repeated with
`scripts/audit_historical_rollouts.py`.

Audit every checked-in bundle without loading simulator assets:

```bash
uv run --locked --no-dev python scripts/audit_eval_conditions.py
```

The checked-in result is [`VISIBILITY_AUDIT.json`](VISIBILITY_AUDIT.json).
