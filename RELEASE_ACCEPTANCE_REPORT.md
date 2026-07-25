# BARX public release acceptance report

- Overall verdict: **PASS**
- Date: 2026-07-24 (America/Los_Angeles)
- Tester: Independent no-context public beta acceptance tester (Codex)
- Repository commit: `ce18937e1b11dd1b0d932fbe1d092b3433a8c28f`
- Branch requested: `barx-release-integration`
- Machine / GPU: Linux x86-64; NVIDIA A40, 46,068 MiB; driver 610.43.02
- Fresh clone: yes, anonymous public HTTPS clone
- Empty BARX and Hugging Face caches: yes
- Anonymous Hugging Face access: yes; common token variables were explicitly unset
- Repository edits: none
- Protocol scope: `RELEASE_TEST_README.md` Sections 0–8 and 10, in order
- Logged command time: 2 h 32 min 14.7 s, including explained failed attempts

The branch's subsequent child commit adds this report, the
[sanitized exact-command ledger](RELEASE_TEST_COMMAND_LOG.md), and the
`uv >= 0.11.11` prerequisite hardening. It does not change the MG, data,
training, or evaluation implementation tested at the exact commit above.

| Section | Status | Duration | Evidence | Notes |
| --- | --- | ---: | --- | --- |
| 0. System and clone | PASS | 23.0 s | `$BARX_EVIDENCE/00-system/`; command IDs `000`–`001` | Anonymous clone resolved exactly to the requested SHA; system CMake 3.28.3 and A40 were available. |
| 1. Install, assets, tests | PASS | 9 min 9.7 s | `$BARX_EVIDENCE/01-install/`; command IDs `010`–`014` | Locked sync passed with documented `uv >= 0.11.11`; 72 unit tests passed; all four RoboCasa archives installed; EGL imports passed. Duration includes runner-environment retries below. |
| 2. Artifacts and checkpoints | PASS | 1 h 6 min 40.6 s | `$BARX_EVIDENCE/02-artifacts/`; command IDs `020`–`022` | Anonymous immutable-revision download passed. The released 5.55 GB checkpoint matched its manifest; all 12 checkpoint and 24 dataset collection members were publicly resolvable. |
| 3. Raw HDF5 | PASS | 26 min 30.3 s | `$BARX_EVIDENCE/03-hdf5/`; command IDs `030`–`034` | XP PnP: 18 files / 1,800 demos / 24.19 GiB. Target human set: 1 file / 50 demos / 0.35 GiB. Manifest hashes and portable HDF5 inspection passed. |
| 4. Published RLDS | PASS | 0.4 s | `$BARX_EVIDENCE/04-rlds/`; command IDs `040`–`041` | Exactly one `dataset_info.json`, one `features.json`, and 256 TFRecord shards; the 24 public ungated dataset IDs matched the checked-in manifests. |
| 5. HDF5-to-RLDS conversion | PASS | 49.1 s | `$BARX_EVIDENCE/05-conversion/`; command IDs `050`–`051` | Converted all 50 Panda Flip Mug episodes and decoded the resulting `panda_flip_mug/1.0.0` dataset. |
| 6. One-step training | PASS | 5 min 35.4 s | `$BARX_EVIDENCE/06-training/`; command IDs `060`–`064` | Paper and adaptation commands dry-resolved as documented; a real single-A40 optimizer step completed with JSONL-only tracking and no checkpoint write. |
| 7. One-trial evaluation | PASS | 40 min 52.7 s | `$BARX_EVIDENCE/07-eval/`; rollout under `$BARX_ARTIFACT_ROOT/rollouts/`; command IDs `070`–`074` | All 100 frozen conditions restored and hash-checked. One public-checkpoint trial completed normally; its visible failed outcome agreed with `success=false`. |
| 8. MimicGen generation | PASS | 2 min 13.5 s | `$BARX_EVIDENCE/08-mimicgen/`; outputs under `$BARX_MG_ROOT/`; command IDs `080`–`086` | Five demos were prepared separately with unchanged source SHA-256. One success was generated in one attempt with canonical 12-D actions; video visibly ends with the mug upright. |
| 10. Git cleanliness | PASS | 0.03 s | `$BARX_EVIDENCE/09-final/git-clean.txt`; command ID `100` | Porcelain status was empty, both diffs exited zero, and HEAD remained the exact starting SHA. |

## First blocking issue

No required repository blocker occurred. Every repository acceptance gate passed at
the exact requested commit. The exact payload, exit code, duration, and output
hashes for every attempt are retained in
`$BARX_EVIDENCE/BARX_RELEASE_EXACT_COMMANDS.md` and
`$BARX_COMMAND_LOG/commands.jsonl`.

The following failed attempts were explained runner or auxiliary-review issues,
not repository failures:

1. `000-system-clone` attempt 1 exited 1 before cloning because an ambient
   user-local `cmake` launcher raised `ModuleNotFoundError: cmake`. The external
   acceptance root was still empty of repository, artifact, and Hugging Face
   state. Attempt 2 used the system-first PATH, found `/usr/bin/cmake` 3.28.3,
   and passed.
2. `011-uv-sync` attempt 1 was interrupted with exit 130 after 272.5 seconds
   when the shared network `uv` cache stalled. The partial virtual environment
   was moved to an external quarantine and a fresh acceptance-local cache was
   selected. Attempt 2 exited 1 because the preinstalled `uv` 0.6.2 was below
   the README's explicit `uv >= 0.11.11` prerequisite and could not parse the
   current lock metadata. After installing public `uv` 0.11.32, attempt 3
   exposed the same shadowing user-local `cmake` launcher during a build.
   Another clean-state assertion quarantined partial state; attempt 4 with the
   system-first PATH passed the exact locked sync.
3. `085-mimicgen-video-inspection` attempt 1 exited 1 only in an added
   contact-sheet helper: a 2.35-second video yields no frame at one sample per
   five seconds. The repository was clean and the original generated MP4 was
   retained. Attempt 2 sampled at 5 fps and passed. This helper was additional
   human-review evidence, not a release command.

The large checkpoint and raw-HDF5 downloads each encountered one ordinary HTTP
read timeout internally and resumed within the original command invocation.
They were not command retries; all final manifest checks passed.

## Human review

- HDF5 inspection: the XP sample had 350 canonical 12-D actions and
  350 × 180 × 320 RGB frames for `XPnPCounterToSink`, with the instruction
  “pick the can from the counter and place it in the sink.” The human sample
  had 244 canonical 12-D actions and matching RGB frames for
  `XFlipMugUpright`, with the instruction “flip the mug on the counter
  upright.” Both contained actions, simulator states, observations, rewards,
  dones, action dictionaries, and auxiliary information.
- Training metrics: the one-step smoke run reached step 1 with loss
  4.855754, L1 loss 0.272353, learning rate `2e-5`, and 10.306 s step time.
  Required config/statistics/metrics files existed and, as requested, no final
  checkpoint was written.
- Evaluation video: valid H.264, 320 × 192, 30 fps, 20.0 s / 600 frames. The
  Panda Counter-to-Sink view was usable and advanced normally. The arm moved
  toward the cucumber but did not grasp or place it; the cucumber remained on
  the counter, agreeing with the episode instruction and terminal
  `success=false`.
- MimicGen video: valid H.264, 512 × 512, 20 fps, 2.35 s / 47 frames. One
  Panda Flip Mug Upright attempt advanced normally. The red mug was visibly on
  its side at the start and upright at the end, agreeing with
  `num_attempts=1`, `num_success=1`, and the 230-step generated trajectory.

## Paper-facing assessment

1. **Representative end-to-end execution: yes.** The acceptance run covered
   anonymous acquisition, locked install, public artifacts, raw and published
   datasets, HDF5-to-RLDS conversion, one real optimizer step, checkpoint
   loading, frozen-condition evaluation, and bounded MimicGen regeneration.
2. **Inputs and outputs needed for paper-scale runs: yes.** The immutable
   collections, 1,800-demo XP PnP subset, released checkpoint, full command
   interfaces, and dry-resolved paper/adaptation launch configurations were
   available and structurally verified.
3. **Paper metric reproduced: no claim.** Section 9 paper-scale reproduction
   was outside this requested acceptance scope. The one-step training,
   one-trial evaluation, and one-success MimicGen jobs are smoke tests only.
   The 0/1 evaluation outcome and 1/1 MimicGen outcome must not be interpreted
   as success-rate estimates or compared with paper metrics.

## Suggested fixes

1. Add an executable `uv >= 0.11.11` assertion immediately before the first
   locked sync, so an older preinstalled `uv` fails with a direct prerequisite
   message rather than a misleading dependency-resolution error. This is
   applied in the report-only child commit: `pyproject.toml` now enforces the
   minimum and the walkthrough upgrades an older resolver before syncing.
2. Add a preflight check that reports shadowed build tools (especially
   `cmake`) or show a system-first PATH example. This would make the public
   walkthrough more robust on machines with broken user-local launchers.

No source-code or release-artifact fix was required by this run.

## Independent MimicGen corroboration

A second no-context runner independently tested the MimicGen gate at the same
exact commit from another fresh anonymous clone and cache. It also passed:

- locked `mg` installation and all 72 unit tests;
- anonymous download and checksum verification of the 50-demo human HDF5;
- five-demo source preparation with identical before/after source SHA-256;
- one Panda Flip Mug success on attempt 1 of at most 25;
- canonical generated actions with shape `[246, 12]` and simulator states with
  shape `[246, 172]`;
- a valid H.264 512 × 512 rollout whose dense visual review showed the mug
  rotate from side-lying to upright; and
- final Git cleanliness at the starting commit.

That runner retained 21 append-only command records: 18 passes and three
explained runner/reporting failures. Its generated HDF5 SHA-256 was
`4322567ce8363045e05d677974715b09cc472993bd47e761efc1383d9681a9c5`.
The different valid trajectory length from the whole-repository run is
expected simulator variation; both runs satisfied the same bounded acceptance
contract.

## Evidence integrity

- The repository-provided logger preserved a literal script, stdout, stderr,
  exit code, duration, and SHA-256 hashes for every substantive acceptance
  command.
- Failed attempts remain in the append-only ledger and were not overwritten.
- The exact-command Markdown replaces only external absolute paths with
  declared placeholders; public repository and Hugging Face identifiers remain
  literal.
- The final repository cleanliness gate passed after every artifact and report
  was written outside the checkout.
