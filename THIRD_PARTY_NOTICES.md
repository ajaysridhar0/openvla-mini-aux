# Third-party notices

BARX builds on the following projects. Bundled components retain their
original copyright notices and licenses.

| Component | Location/source | Version | License |
| --- | --- | --- | --- |
| OpenVLA / MiniVLA-derived policy code | `policy/` | imported snapshot `b15f06e` | MIT (`policy/LICENSE`) |
| RoboCasa | `robocasa_x/` | modified 0.2.0 snapshot `9338b1a` | MIT (`robocasa_x/LICENSE`) |
| robosuite | [official upstream](https://github.com/ARISE-Initiative/robosuite) | commit `2ebb2a0` (1.5.1) | MIT |
| dlimp | uv Git dependency | commit `040105d` | upstream terms |
| VQ-BeT | uv Git dependency | commit `09d4851` | MIT |
| robosuite-models | locked package dependency | 1.0.0 | upstream terms |
| robomimic | locked package dependency | 0.3.0 | upstream terms |
| MimicGen support | `third_party/mimicgen/` | upstream 1.0.1 plus BARX task/runtime patches, source revision `62aca46` | NVIDIA Source Code License — non-commercial research or evaluation (`third_party/mimicgen/LICENSE`) |

MimicGen was used to synthesize the rendered training data and is available as
the optional `mg` dependency. It is not covered by BARX's MIT license. Training
consumes its final HDF5 files through the unified converter. See `uv.lock` for
the complete resolved dependency graph.

Public BARX datasets and model artifacts have their own distribution terms and
provenance, which do not replace the licenses above. See
`docs/artifact_licenses.md`.
