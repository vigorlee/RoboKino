# Contributing to RoboKino

For a reproducibility issue, include the repository commit, Isaac Sim version, task/configuration, asset source, launch command, and relevant error output. Use [GitHub Issues](https://github.com/vigorlee/RoboKino/issues) to report problems.

## Adding a task

1. Add a task definition and success checker under `tasks/`, and expose the class in `tasks/__init__.py`.
2. Add a scripted controller under `controllers/tasks/`, using the existing RMPflow motion interface where appropriate.
3. Add the collection script and camera/task configuration under `data_collect/<task>/`.
4. Register required objects and scene/base poses in `envs/cfg/`.
5. If policy evaluation is supported, add a matching `EVAL_TASK_PRESETS` entry in `evaluate.py` and a task choice in `robokino_benchmark.sh`.
6. Update both README task tables and record the source/license of any new assets.

## Validation

Run `python tools/check_repository.py` with PyYAML installed. For runtime changes, additionally run the affected task in Isaac Sim with the required assets. Check a saved HDF5 episode for matching observation/action lengths, image dimensions, joint ordering, and successful task completion. For evaluation changes, verify the ROS client and simulator exchange the expected observations and actions.

The upstream pre-commit configuration is preserved for optional formatting checks. Full simulator tests require a GPU and external dependencies and are not performed by the repository-only checks.

## Licenses and attribution

Retain the Apache 2.0 copyright headers on imported source files. When modifying one, add a prominent notice describing your changes and update `THIRD_PARTY.md`. Record new third-party dependencies and asset licenses. RoboKino-authored standalone additions use the MIT license in `licenses/MIT-RoboKino.txt`.
