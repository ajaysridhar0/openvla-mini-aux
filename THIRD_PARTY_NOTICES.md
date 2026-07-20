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

MimicGen was used to synthesize data, but its source is not redistributed in
this repository and it is not required to train from the final release data.
See `uv.lock` for the complete resolved dependency graph.
