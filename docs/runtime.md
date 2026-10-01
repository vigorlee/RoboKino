# Runtime and Asset Notes

## Launch from the repository root

The robot configuration uses relative paths such as `assets/piper_left.usd`. Run the collection commands from the RoboKino root. The new evaluation launcher changes to this directory automatically.

Use Isaac Sim 4.5.0's `python.sh`, rather than a generic virtual environment, for simulator entry points. Several scripts instantiate `SimulationApp` before parsing command-line arguments, so even `--help` on those Python scripts requires the simulator. The standalone `robokino_benchmark.sh --help` and `tools/check_repository.py` do not start Isaac Sim.

## Asset configuration

[`envs/cfg/robots.yaml`](../envs/cfg/robots.yaml) defines the mobile base and two Piper arms. [`envs/cfg/objects.yaml`](../envs/cfg/objects.yaml) maps scene and object names to USD files and initial poses. The configuration expects these environment files:

```text
assets/environments/KitchenRoom/kitchen_room.usd
assets/environments/Apartment/apartment.usd
assets/environments/Apartment/apartment_short.usd
assets/environments/industry/industry.usd
assets/mobile_aloha_base.usd
assets/piper_left.usd
assets/piper_right.usd
```

Each task also references object-specific assets from `objects.yaml`. Preserve the downloaded directory hierarchy. USD files can refer to textures and other USD layers, so replacing a single file may not be sufficient.

For custom RoboKino scenes, keep the scene key, base pose, task object names, robot articulation, end-effector prims, joint order, and camera prims consistent across the YAML and Python wrappers. Changing the robot requires updating the URDF and RMPflow configuration under `controllers/motion/` as well as `envs/robots.py`.

The collection CLI maps `--env kitchen` to `TABLE_NAME=kitchen_table` and `BASE_NAME=base_kitchen`, with equivalent mappings for `apartment`, `apartmentshort`, and `industry`. Collection configuration names are relative to the task's script directory; they are not arbitrary repository-relative paths.

## Data output

HDF5 episodes contain extensible image datasets at `observations/images/<camera>`, joint positions at `observations/qpos`, and actions at `action`. Images are stored as uint8 RGB/RGBA arrays; joint and action arrays are numeric vectors in the imported robot order. The file-level `sim` attribute is set to true.

Collection saves only successful episodes. The original writers start episode numbering at zero for each process and open HDF5 files in write mode. Use a new output directory for each collection run. Failed attempts can be retried indefinitely until the requested `max_demo` count is reached or the simulator is stopped.

Some YAML files contain `headless`, but the current collection scripts create a graphical `SimulationApp` before reading the configuration. Editing that YAML field alone does not enable headless collection.

## ROS evaluation

The evaluator publishes RGB images on `/camera_l/color/image_raw`, `/camera_r/color/image_raw`, and `/camera_f/color/image_raw`, and joint states on `/puppet/joint_left` and `/puppet/joint_right`. It receives action commands on `/master/joint_left` and `/master/joint_right`. Consult [`evaluate.py`](../evaluate.py) for reset and action-chunk signaling.

Ensure the simulator and policy client share the ROS master and compatible message definitions. The preserved launcher starts its own `roscore` and RViz, so use it with an available ROS master port. A compatible `cv_bridge` build is required for the simulator's Python version; sourcing a ROS installation alone does not resolve a Python ABI mismatch.

The evaluator logs success rate and completion steps. It does not yet implement the complete process-evaluation panel in the RoboKino overview figure. GPU simulation, scene physics, policy performance, and ROS interoperability must be validated in the target runtime.
