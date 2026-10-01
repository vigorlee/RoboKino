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
"""Data-collection script for the ScrewPitcherLid task.

The left arm holds the pitcher steady while the right arm picks up the lid,
places it on top of the pitcher, and rotates the wrist to screw it on.
``(observation, action)`` pairs are recorded into an HDF5 file for each
successful episode.
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

from controllers.tasks.screw_pitcher_lid import (  # noqa: E402
    LeftArmPickController, RightArmPickRotateController)
from data_collect.utils import append_hdf5  # noqa: E402
from data_collect.utils import extract_action_from_obs  # noqa: E402
from data_collect.utils import extract_state_from_obs  # noqa: E402
from data_collect.utils import init_hdf5  # noqa: E402
from envs import IsaacsimEnv  # noqa: E402


def setup_controllers(
    env: IsaacsimEnv,
) -> Tuple[LeftArmPickController, RightArmPickRotateController]:
    """Build and return the scripted controllers for both arms.

    Args:
        env: Active Isaac Sim environment.

    Returns:
        A tuple ``(left_controller, right_controller)`` ready for data collection.
    """
    left_events_dt = [
        0.008, 0.008, 0.5, 0.2, 0.04, 0.004, 0.004, 0.2, 0.008, 0.008
    ]
    right_events_dt = [
        0.008, 0.008, 0.5, 0.2, 0.01, 0.004, 0.004, 0.005, 0.2, 0.05
    ]

    left_controller = LeftArmPickController(
        name='left_controller',
        robot_articulation=env.robot.left_manipulator,
        gripper=env.robot.left_manipulator.gripper,
        events_dt=left_events_dt,
    )

    right_controller = RightArmPickRotateController(
        name='right_controller',
        robot_articulation=env.robot.right_manipulator,
        gripper=env.robot.right_manipulator.gripper,
        events_dt=right_events_dt,
    )
    return left_controller, right_controller


def collect_episode_data(
    env: IsaacsimEnv,
    left_controller: LeftArmPickController,
    right_controller: RightArmPickRotateController,
    pitcher_name: str,
    lid_name: str,
) -> Dict[str, Any]:
    """Run a single episode with dual-arm collaboration and record the data.

    Args:
        env: Active Isaac Sim environment.
        left_controller: Scripted left-arm pick controller (holds the pitcher).
        right_controller: Scripted right-arm pick-and-rotate controller.
        pitcher_name: Scene object name of the pitcher.
        lid_name: Scene object name of the lid.

    Returns:
        A dict with two parallel lists under the keys ``'states'`` and
        ``'actions'``.
    """
    obs = env.step()

    episode_data = {'states': [], 'actions': []}

    cnt = 0
    sample_step = 1

    # Left-arm waypoints: approach and stabilise the pitcher body.
    left_picking_position = np.array(obs[pitcher_name]['pos']) + np.array(
        [-0.088, 0.0, 0.0])
    left_placing_position = np.array(obs[pitcher_name]['pos']) + np.array(
        [0.0, 0.2, 0.0])
    left_target_orientation = euler_angles_to_quat(np.array([0, np.pi / 2, 0]))

    # Right-arm waypoints: pick the lid and place it on top of the pitcher.
    right_picking_position = np.array(obs[lid_name]['pos']) + np.array(
        [-0.01, 0.01, 0.065])
    right_placing_position = np.array(obs[pitcher_name]['pos']) + np.array(
        [-0.125, 0.115, 0.184])
    right_target_orientation = euler_angles_to_quat(
        np.array([0, np.pi, np.pi / 2]))

    while not right_controller.is_done():
        left_event = left_controller.get_current_event()
        right_event = right_controller.get_current_event()

        # Abort the episode early when the right gripper loses the lid.
        if right_event in [4, 5, 6, 7]:
            if not env.task.checker.check_gripper(env):
                print('The gripper is not holding the object.')
                break

        if cnt % sample_step == 0:
            state = extract_state_from_obs(obs)
            episode_data['states'].append(state)

        right_actions = right_controller.forward(
            picking_position=right_picking_position,
            placing_position=right_placing_position,
            current_joint_positions=np.array(
                list(obs['piper']['dof_pos'].values())[8:], dtype=float),
            end_effector_orientation=right_target_orientation,
            end_effector_offset=np.array([0.0, 0.0, 0.07]),
        )

        # The left arm only needs to act during its first few phases; after
        # that it stays put while the right arm finishes the rotation.
        if left_event < 4:
            left_actions = left_controller.forward(
                picking_position=left_picking_position,
                placing_position=left_placing_position,
                current_joint_positions=np.array(
                    list(obs['piper']['dof_pos'].values())[:8], dtype=float),
                end_effector_orientation=left_target_orientation,
                end_effector_offset=np.array([-0.3, 0.084, 0.110]),
            )
            obs = env.step([left_actions, right_actions])
        else:
            obs = env.step([None, right_actions])

        # Clamp gripper joints while the respective arm should hold something.
        if left_event in [3]:
            obs['piper']['dof_pos']['llink7'] = 0.0
            obs['piper']['dof_pos']['llink8'] = 0.0
        if right_event in [3, 4, 5, 6, 7]:
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
    parser.add_argument('--env', type=str, default='apartmentshort')
    parser.add_argument(
        '--config', type=str, default='pitcher_lid_screw_config.yaml')
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

    pitcher_name = config.get('pitcher', 'pitcher')
    os.environ['PITCHER_NAME'] = pitcher_name
    lid_name = config.get('pitcherlid', 'pitcherlid')
    os.environ['LID_NAME'] = lid_name

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

        left_controller, right_controller = setup_controllers(env)

        for i in range(60):
            env.world.step(render=True)

        episode_data = collect_episode_data(
            env,
            left_controller,
            right_controller,
            pitcher_name=pitcher_name,
            lid_name=lid_name,
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
