# BARX public release checklist

The public launch is ready only after every section below passes from a fresh,
anonymous environment. Artifact repository IDs and revisions must be immutable
and recorded in the release checkout.

## Data

- [x] Raw HDF5 repository is public, CC BY 4.0 licensed, and pinned.
- [x] All 240 raw files match `dataset/manifest.csv` by path, size, and SHA-256.
- [x] Processed RLDS repositories are public, complete, and grouped in the
      labeled BARX RLDS collection.
- [x] The raw HDF5 repository is grouped in the labeled BARX HDF5 collection.
- [x] A fresh anonymous download passes `scripts/verify_raw_data.py`.
- [x] All 24 RLDS cards state CC BY 4.0, conversion provenance, intended use,
      limitations, and attribution.

## Conversion and generation

- [x] Raw HDF5-to-RLDS conversion completes for one paper dataset and matches
      the published RLDS schema.
- [x] The BARX MimicGen compatibility snapshot, license, and task
      configurations are vendored and dependency-locked.
- [x] Human source preparation and one MimicGen generation smoke test complete
      without cluster-specific paths.
- [x] Generated HDF5 output passes the same portable-data verifier.

## Training and checkpoints

- [x] Pretraining model checkpoints are public, ungated, and revision-pinned.
- [x] Training configuration aliases cover the released paper datasets.
- [x] A one-step training smoke test completes from downloaded public artifacts.
- [x] The documented full training commands and expected resource requirements
      are accurate.
- [x] The base VLM, VQ tokenizer, and 12 policy-checkpoint cards state Apache
      2.0, provenance, intended use, limitations, and upstream attribution.

## Evaluation

- [x] A one-trial headless evaluation completes from public artifacts.
- [x] Frozen evaluation conditions and paper-protocol defaults are documented.
- [x] Evaluation output and aggregation are machine-readable and do not
      overwrite previous runs.

## Anonymous release acceptance

- [x] Start from a fresh clone with empty Hugging Face and simulator caches.
- [x] Follow the public README without unpublished context or credentials.
- [x] Exercise installation, data download, raw verification, RLDS conversion,
      MimicGen generation, training, and evaluation.
- [x] Run the complete unit test suite.
- [x] Confirm `git status --porcelain` is empty at the end.
- [x] Resolve every release-blocking finding and repeat the affected step before
      announcing the release.

## Recorded evidence

The no-context acceptance run at the pinned release commit passed on
2026-07-25. It exercised every gate above, including 72 unit tests, one real
optimizer step on an A40, one headless simulator trial, one bounded MimicGen
success, and final Git cleanliness. See
[`RELEASE_ACCEPTANCE_REPORT.md`](../RELEASE_ACCEPTANCE_REPORT.md) and
[`RELEASE_TEST_COMMAND_LOG.md`](../RELEASE_TEST_COMMAND_LOG.md).

On 2026-07-26, a separate token-disabled metadata audit fetched all 14 model
cards, all 24 RLDS cards, and all three BARX collections from empty local card
cache entries. Every repository was public and ungated and exposed the expected
license and documentation sections. This follow-up changed documentation and
repository cards only; it did not alter code, pinned artifact payloads, training
configuration, or simulator behavior, so the expensive execution gates did not
need to be repeated.
