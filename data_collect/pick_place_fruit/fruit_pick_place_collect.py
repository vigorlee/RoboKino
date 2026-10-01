# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data-collection script for the PickPlaceFruit task.

The right arm picks the fruit and places it on the plate. ``(observation,
action)`` pairs are recorded into an HDF5 file for each successful episode.
"""

from omni.isaac.kit import SimulationApp

sim_app = SimulationApp({'headless': False})

import argparse  # noqa: E402
import os  # noqa: E402
from typing import Any, Dict, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import rootutils  # noqa: E402
import yaml  # noqa: E402
from isaacsim.core.utils.rotations import euler_angles_to_quat  # noqa: E402

rootutils.setup_root(__file__, pythonpath=True)

from controllers.tasks.pick_place_fruit import \
    PickPlaceFruitController  # noqa: E402
from data_collect.utils import append_hdf5  # noqa: E402
from data_collect.utils import extract_action_from_obs  # noqa: E402
from data_collect.utils import extract_state_from_obs  # noqa: E402
from data_collect.utils import init_hdf5  # noqa: E402
from envs import IsaacsimEnv  # noqa: E402


def setup_controllers(
    env: IsaacsimEnv,
) -> Tuple[PickPlaceFruitController, PickPlaceFruitController]:
    """Build and return the scripted controller for the right arm.

    Args:
        env: Active Isaac Sim environment.

    Returns:
        The right-arm pick-and-place controller used to collect demonstrations.
    """
    right_events_dt = [0.008, 0.008, 0.5, 0.2, 0.008, 0.005, 0.008, 0.4, 0.008]
    right_controller = PickPlaceFruitController(
        name='right_controller',
        robot_articulation=env.robot.right_manipulator,
        gripper=env.robot.right_manipulator.gripper,
        events_dt=right_events_dt,
    )
    return right_controller


def collect_episode_data(
    env: IsaacsimEnv,
    right_controller: PickPlaceFruitController,
    fruit_name: str,
    plate_name: str,
) -> Dict[str, Any]:
    """Run a single episode with the right arm and record the data.

    Args:
        env: Active Isaac Sim environment.
        right_controller: Scripted right-arm pick-and-place controller.
        fruit_name: Scene object name of the fruit to pick.
        plate_name: Scene object name of the target plate.

    Returns:
        A dict with two parallel lists under the keys ``'states'`` and
        ``'actions'``.
    """
    obs = env.step()

    episode_data = {'states': [], 'actions': []}

    cnt = 0
    sample_step = 1

    while not right_controller.is_done():
        event = right_controller.get_current_event()

        # Abort the episode early when the gripper has lost the fruit.
        if event in [4, 5, 6]:
            if not env.task.checker.check_gripper(env):
                print('The gripper is not holding the object.')
                break

        if cnt % sample_step == 0:
            state = extract_state_from_obs(obs)
            episode_data['states'].append(state)

        target_orientation = euler_angles_to_quat(
            np.array([0.0, np.pi, np.pi / 2]))

        right_actions = right_controller.forward(
            picking_position=np.array(obs[fruit_name]['pos']),
            placing_position=np.array(obs[plate_name]['pos']) +
            np.array([0.0, 0.0, 0.05]),
            current_joint_positions=np.array(
                list(obs['piper']['dof_pos'].values())[8:], dtype=float),
            end_effector_orientation=target_orientation,
            end_effector_offset=np.array([0.0, 0.0, 0.13]),
        )

        obs = env.step([right_actions])

        # Clamp gripper joints while the fruit should be grasped.
        if event in [3, 4, 5, 6]:
            obs['piper']['dof_pos']['rlink7'] = 0.0
            obs['piper']['dof_pos']['rlink8'] = 0.0

        if cnt % sample_step == 0:
            action = extract_action_from_obs(obs)
            episode_data['actions'].append(action)

        cnt += 1

    return episode_data


def main():
    """Parse CLI arguments, run the collection loop and save successful episodes."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='kitchen')
    parser.add_argument(
        '--config', type=str, default='banana_pick_place_config.yaml')
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to save the collected data')
    args = parser.parse_args()

    cfg_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(cfg_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # Populate task-specific environment variables consumed by the task class.
    table_name = f"{args.env}_table"
    os.environ['TABLE_NAME'] = table_name
    base_name = f"base_{args.env}"
    os.environ['BASE_NAME'] = base_name

    fruit_name = config.get('fruit', 'banana')
    os.environ['FRUIT_NAME'] = fruit_name
    plate_name = config.get('plate', 'plate')
    os.environ['PLATE_NAME'] = plate_name

    env = IsaacsimEnv(config)
    env.reset()
    max_demo = config.get('max_demo', 100)
    episodes = []

    # Let the simulator settle before starting the first episode.
    for i in range(60):
        env.world.step(render=True)

    while sim_app.is_running() and len(episodes) < max_demo:
        print(f"Current episode: {len(episodes) + 1}/{max_demo}")

        env.reset()
        for i in range(60):
            env.world.step(render=True)

        right_controller = setup_controllers(env)

        for i in range(60):
            env.world.step(render=True)

        episode_data = collect_episode_data(
            env,
            right_controller,
            fruit_name=fruit_name,
            plate_name=plate_name,
        )

        if env.task.checker.check(env):
            episodes.append(True)

            if args.data_path:
                data_path = args.data_path
            else:
                data_path = os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(__file__))),
                    'data',
                    args.env,
                    env.task_name,
                )

            os.makedirs(data_path, exist_ok=True)
            h5_path = os.path.join(
                data_path,
                f"episode_{len(episodes) - 1:05d}.hdf5")  # noqa: E231

            init_hdf5(h5_path, episode_data['states'][0])

            steps = len(episode_data['states'])

            for i in range(steps):
                meta_data = episode_data['states'][i].copy()
                if i < len(episode_data['actions']):
                    meta_data['action'] = episode_data['actions'][i]
                append_hdf5(h5_path, meta_data)
            print(
                'Success, states:',
                len(episode_data['states']),
                'actions:',
                len(episode_data['actions']),
            )
        else:
            print('Failed')

    sim_app.close()


if __name__ == '__main__':
    main()
