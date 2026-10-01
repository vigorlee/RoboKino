# Code and Media Provenance

## FluxBisim simulation implementation

- Source: [FluxVLA/FluxBisim](https://github.com/FluxVLA/FluxBisim).
- Revision: [`621df43e08df3f27b5355e65e3f2c2918750bccb`](https://github.com/FluxVLA/FluxBisim/commit/621df43e08df3f27b5355e65e3f2c2918750bccb).
- Imported on: 2026-10-02.
- License: Apache License 2.0, preserved verbatim in [LICENSE](LICENSE).
- Attribution: FluxVLA Engine Team / LimX Dynamics; original per-file notices retained.
- URDF acknowledgement: AgileX Robotics, as stated in the upstream documentation.

The import contains `controllers/`, `data_collect/`, `envs/`, `tasks/`, `rviz/`, `evaluate.py`, `fluxbisim_benchmark.sh`, `.flake8`, `.pre-commit-config.yaml`, `.gitignore`, and `LICENSE`. Each file's original Git blob SHA and mode are recorded in [docs/upstream-import.json](docs/upstream-import.json).

The original Python source, YAML/URDF configuration, ROS visualization, launcher, lint configuration, and Apache license are preserved without content changes. The imported `.gitignore` has a prominent modification notice and additional local environment/output patterns. The new `robokino_benchmark.sh` invokes the unchanged upstream launcher from the repository root.

The upstream presentation image `docs/framework.jpg`, video attachment, README, and reported leaderboard were excluded from the import. No upstream image is included in this repository.

## RoboKino additions

The README files, `docs/runtime.md`, contribution guide, root marker, dependency list, Git attributes, new launcher, and repository-checking utility were added for RoboKino. RoboKino-authored documentation and utilities use the [MIT license](licenses/MIT-RoboKino.txt), preserving the original project's copyright and license terms.

`docs/images/robokino-framework.pdf` is the author-supplied `v7.pdf`, copied unchanged. `docs/images/robokino-framework.png` is a 220-dpi rendering of its single page. No FluxBisim presentation media were used to produce these files.

## External assets, datasets, and policy software

The following resources are linked as optional external dependencies; their contents are not redistributed here:

| Resource | Provider | Role |
| --- | --- | --- |
| [FluxBisimAssets](https://huggingface.co/datasets/limxdynamics/FluxBisimAssets) | LimX Dynamics | USD scenes, robots, and objects expected by the imported configuration; dataset card specifies CC BY-NC 4.0 |
| [FluxBisimData](https://huggingface.co/datasets/limxdynamics/FluxBisimData) | LimX Dynamics | Upstream demonstrations in LeRobot format |
| [FluxVLA](https://github.com/FluxVLA/FluxVLA) | FluxVLA contributors | Data conversion, policy training, and ROS inference clients |
| [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html) | NVIDIA | Simulator runtime |

The asset collection acknowledges YCB Benchmark, Lightwheel SimReady, X-Humanoid ArtVIP, NVIDIA Omniverse USD, and AgileX Robotics. Consult each external resource's own license and documentation; the source-code license in this repository does not relicense those resources.
