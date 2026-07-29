# Public artifact licenses and provenance

BARX publishes code, datasets, and model artifacts under different terms. This
page records their scope so a public repository being downloadable is not
mistaken for an unspecified right to reuse it.

## Datasets: CC BY 4.0

The following BARX data artifacts are released under the
[Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/):

- the `ajaysri/barx-raw-hdf5` archive; and
- all 24 processed RLDS/TFDS repositories in the BARX RLDS collection.

The processed RLDS repositories are conversions of the released raw HDF5
demonstrations. Their cards identify the paper dataset variant, task,
embodiment, episode count, conversion code, and raw source. Cite the BARX paper
and preserve the dataset-card attribution when redistributing the data or a
derived format.

Some demonstrations were generated with MimicGen, while target-50
demonstrations were collected by human teleoperation in simulation. The
included MimicGen code uses NVIDIA's source-code license; BARX distributes the
released demonstration data under CC BY 4.0.

## Model artifacts: Apache 2.0

The following BARX model artifacts are released under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0):

- the BARX MiniVLA base VLM;
- the XP-900 PnP VQ action tokenizer; and
- all 12 Joint Reps and No Reps source-pretraining checkpoint repositories in
  the BARX model collection.

Their cards record training-data provenance, task and method, intended use,
limitations, and upstream components. Preserve the notices and attribution for
Qwen2.5, DINOv2, SigLIP, MiniVLA/OpenVLA, RoboCasa/RoboCasa-X, and BARX when
redistributing modified artifacts. Apache 2.0 for a BARX artifact does not
replace a separately applicable upstream license.

## Code

- New BARX code is MIT licensed under the repository root `LICENSE`.
- MiniVLA/OpenVLA-derived policy code retains the MIT license in
  `policy/LICENSE`.
- RoboCasa-X retains the MIT license in `robocasa_x/LICENSE`.
- The included MimicGen code retains NVIDIA's non-commercial
  research/evaluation license in
  `third_party/mimicgen/LICENSE`.

See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for pinned versions
and source locations.

## Citation

```bibtex
@inproceedings{sridhar2026barx,
  title     = {Cross-Embodiment Transfer via Behavior-Aligned Representations},
  author    = {Sridhar, Ajay and Gao, Jensen and Yang, Jonathan and Mercat, Jean and Belkhale, Suneel and Sadigh, Dorsa},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```
