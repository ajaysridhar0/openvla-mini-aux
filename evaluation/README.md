# Frozen evaluation conditions

BARX evaluates every method on the same 100 post-settling simulator states for
each task/embodiment pair. A condition bundle consists of a JSON metadata file
and a compressed archive containing the exact processed MuJoCo XML and settled
state under `conditions/<task>/<embodiment>.*`. Asset paths are package-relative,
so bundles are portable across installations.

The bundles are regenerated from the frozen paper code and seeds 1000–1099.
Historical paper runs did not retain their MuJoCo states, so the bundles should
be described as reconstructed from the paper snapshot, not as extracted from
the original rollouts. Each bundle records its source revision and verifies its
complete metadata, model XML, and simulator-state hashes during evaluation.
RoboCasa mutates resolved object-placement dictionaries while loading a scene;
those redundant fields are left out of the post-load metadata comparison,
while object identity and placement remain covered by the XML and state hashes.

Generation is sequential: episode 1 uses seed 1000, episode 2 uses seed 1001,
and so on. Do not construct a later paper episode directly from its seed.
RoboCasa retains some sampled Python-side scene state across hard resets, so a
later condition is reproduced only by replaying all preceding resets in the
same environment. The generator does this automatically.

Generate one bundle after installing the RoboCasa assets:

```bash
uv run --locked python scripts/generate_eval_conditions.py \
  --task pnp_counter_to_sink \
  --embodiment panda
```

The public `scripts/evaluate.py` launcher loads these bundles by default and
fails before model inference if a bundle is absent, corrupt, or incompatible
with the current simulator. This prevents different methods from silently
receiving different sampled objects, layouts, styles, or initial poses.

Restore and hash-check every condition in one generated bundle with:

```bash
uv run --locked python scripts/verify_eval_conditions.py \
  --task pnp_counter_to_sink \
  --embodiment panda
```

See [historical validation](HISTORICAL_VALIDATION.md) for the video-based spot
check against the retained paper rollouts and its limitations. If those private
rollouts are available, the check can be repeated with
`scripts/audit_historical_rollouts.py`.
