# BARX public release checklist

The public launch is ready only after every section below passes from a fresh,
anonymous environment. Artifact repository IDs and revisions must be immutable
and recorded in the release checkout.

## Data

- [x] Raw HDF5 repository is public, CC BY 4.0 licensed, and pinned.
- [x] All 240 raw files match `dataset/manifest.csv` by path, size, and SHA-256.
- [ ] Processed RLDS repositories are public, complete, and grouped in the
      labeled BARX RLDS collection.
- [x] The raw HDF5 repository is grouped in the labeled BARX HDF5 collection.
- [ ] A fresh anonymous download passes `scripts/verify_raw_data.py`.

## Conversion and generation

- [ ] Raw HDF5-to-RLDS conversion completes for one paper dataset and matches
      the published RLDS schema.
- [ ] The BARX MimicGen fork and task configurations are public and pinned.
- [ ] Human source preparation and one MimicGen generation smoke test complete
      without cluster-specific paths.
- [ ] Generated HDF5 output passes the same portable-data verifier.

## Training and checkpoints

- [ ] Pretraining model checkpoints are public, ungated, and revision-pinned.
- [ ] Training configuration aliases cover the released paper datasets.
- [ ] A one-step training smoke test completes from downloaded public artifacts.
- [ ] The documented full training commands and expected resource requirements
      are accurate.

## Evaluation

- [ ] A one-trial headless evaluation completes from public artifacts.
- [ ] Frozen evaluation conditions and paper-protocol defaults are documented.
- [ ] Evaluation output and aggregation are machine-readable and do not
      overwrite previous runs.

## Anonymous release acceptance

- [ ] Start from a fresh clone with empty Hugging Face and simulator caches.
- [ ] Follow the public README without unpublished context or credentials.
- [ ] Exercise installation, data download, raw verification, RLDS conversion,
      MimicGen generation, training, and evaluation.
- [ ] Run the complete unit test suite.
- [ ] Confirm `git status --porcelain` is empty at the end.
- [ ] Resolve every release-blocking finding and repeat the affected step before
      announcing the release.
