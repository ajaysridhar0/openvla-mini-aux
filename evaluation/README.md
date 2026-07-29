# Evaluation conditions

BARX evaluates every method on the same 100 fixed conditions for each task and
embodiment, so models are compared on the same scenes. Each bundle under
`conditions/<task>/<embodiment>.*` stores the scene and simulator state needed
to replay those trials.

Pick-and-place targets come from RoboCasa `obj_set1`, instance split `A`:
`apple`, `banana`, `can`, `carrot`, `cucumber`, `lemon`, `orange`, and
`sponge`. The conditions use the task-specific scene filters, and each target
must be visible in the policy camera.

The `scripts/evaluate.py` launcher uses the included condition bundles by
default and checks target visibility before model inference. See the root
README for a one-trial example or omit `--episodes 1` to run all 100
conditions.

To generate a new bundle after installing the RoboCasa assets:

```bash
uv run --locked --no-dev python scripts/generate_eval_conditions.py \
  --task pnp_counter_to_sink \
  --embodiment panda
```

Check a generated bundle with:

```bash
uv run --locked --no-dev python scripts/verify_eval_conditions.py \
  --task pnp_counter_to_sink \
  --embodiment panda
```
