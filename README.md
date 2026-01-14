# RoboKino

## About

**RoboKino** is an Isaac Sim-based benchmark and data-generation framework for fine-grained dual-arm manipulation. It extends the Aloha dual-arm platform in simulation and leverages Isaac Sim's photorealistic rendering capabilities.

RoboKino structures dual-arm behaviors into five cooperative atomic controllers that compose long-horizon task sequences. It supports controlled task variants with domain randomization and provides diagnostic evaluation beyond binary success via stage-wise progress checks and coordination analysis.

### Key Features

- **5 Atomic Dual-Arm Controllers**: Composable controllers for complex collaborative tasks
- **10,000+ Expert Trajectories**: LeRobot v2.1-compliant dataset with human-operated demonstrations
- **500 Interactive Assets**: Reusable assets spanning domestic, desktop/office, and industrial scenes
- **Domain Randomization**: Enhanced task diversification and generalization
- **Isaac Sim Integration**: Photorealistic rendering and physics simulation
- **Aloha-Compatible**: Hardware-aligned dual-arm configuration
- **Fine-Grained Evaluation**: Stage-wise progress assessment and coordination analysis
- **End-to-End Pipeline**: Unified data acquisition, training, and inference workflow

### Performance Metrics

- Average Trajectory Smoothness: **81.3%**
- Average Spatial Complexity: **82.4%**

## Atomic Controllers

RoboKino provides five cooperative dual-arm atomic controllers that can be composed into long-horizon task sequences:

### 1. Pick/Place Controller
Coordinated grasping and placement operations for dual-arm object manipulation.

### 2. Open/Close Controller
Bimanual operations for opening/closing containers, drawers, doors, and other articulated objects.

### 3. Handover Controller
Object transfer between arms, enabling in-hand manipulation and workspace extension.

### 4. Rotate Controller
Synchronized rotation of objects requiring two-handed manipulation (e.g., valves, lids, steering wheels).

### 5. Pull/Push Controller
Coordinated pushing and pulling actions for drawers, sliding doors, and heavy objects.

## Task Scenarios

RoboKino supports diverse manipulation scenarios across three domains:

### Domestic Scenes
Kitchen and household tasks requiring dual-arm coordination (cooking, cleaning, organizing).

### Desktop/Office Scenes
Office manipulation tasks (document handling, tool organization, device operation).

### Industrial Scenes
Manufacturing and assembly tasks requiring precise bimanual coordination.

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with RTX support (recommended for Isaac Sim)
- Isaac Sim 2023.1.0 or later
- CUDA 11.8+ and compatible drivers

### Setup Environment

1. **Install Isaac Sim**

Follow the [official Isaac Sim installation guide](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html).

2. **Clone the repository**

```bash
git clone --recurse-submodules https://github.com/vigorlee/RoboKino.git
cd RoboKino
```

3. **Create and activate conda environment**

```bash
conda create -n robokino python=3.10
conda activate robokino
```

4. **Install the package**

```bash
# Basic installation
pip install -e .

# Install with training dependencies
pip install -e ".[train]"

# Install with all dependencies
pip install -e ".[all]"
```

### Verify Installation

Test your installation by running a simple demo:

```bash
python examples/demo_replay.py
```

## Getting Started

### 1. Basic Demo Replay

Start with the simplest example to verify your setup:

```bash
python examples/1_replay_demo.py
```

This script demonstrates:
- Loading LeRobot v2.1-compliant trajectories
- Replaying demonstrations in Isaac Sim
- Basic environment and robot control
- Visualization of dual-arm coordination

### 2. Understanding Atomic Controllers

Learn about the five atomic controllers:

```bash
python examples/2_atomic_controllers.py
```

Features:
- Individual controller demonstrations
- Parameter configurations
- Composing controllers into sequences
- Switching between controllers

### 3. Task Composition

Create complex tasks by composing atomic controllers:

```bash
python examples/3_compose_tasks.py
```

This example shows:
- Long-horizon task planning
- Controller sequencing
- State transitions between controllers
- Task-specific parameter tuning

### 4. Domain Randomization

Explore domain randomization capabilities:

```bash
python examples/4_domain_randomization.py
```

Demonstrates:
- Visual randomization (lighting, textures, camera poses)
- Physical randomization (object properties, friction)
- Task variants generation
- Evaluation across randomized scenarios

### 5. Model Evaluation

Evaluate trained policies on RoboKino tasks.

#### Parameters

**Model Configuration:**
- `--model_path`: Path to trained model checkpoint
- `--model_type`: Model architecture (e.g., 'act', 'diffusion_policy', 'vla')
- `--device`: Device for inference ('cuda' or 'cpu')

**Task Configuration:**
- `--task`: Task name or path to task configuration
- `--controller`: Specific atomic controller to evaluate
- `--num_episodes`: Number of evaluation episodes (default: 50)
- `--max_steps`: Maximum steps per episode (default: 500)

**Evaluation Options:**
- `--enable_recording`: Record evaluation videos
- `--save_metrics`: Save detailed metrics to file
- `--output_dir`: Directory for outputs (default: './eval_results')
- `--seed`: Random seed for reproducibility

**Visualization:**
- `--render`: Enable real-time rendering
- `--camera_view`: Camera perspective ('front', 'side', 'top', 'ego')
- `--show_coordination`: Visualize inter-arm coordination metrics

## Dataset Structure

RoboKino datasets follow the LeRobot v2.1 format for maximum compatibility:

```
data/
├── domestic/
│   ├── task_name/
│   │   ├── episode_000000.hdf5
│   │   ├── episode_000001.hdf5
│   │   └── ...
├── desktop_office/
│   └── ...
├── industrial/
│   └── ...
└── metadata.json
```

### Data Format

Each episode contains:
- **Observations**: RGB images, depth, proprioceptive states
- **Actions**: Joint positions/velocities for dual arms
- **Rewards**: Stage-wise progress and task completion
- **Metadata**: Task parameters, controller sequence, randomization settings

### Loading Datasets

```python
from robokino.dataset import RoboKinoDataset

# Load dataset
dataset = RoboKinoDataset(
    data_path="./data/domestic",
    task="pick_and_place",
    split="train"
)

# Access episodes
episode = dataset[0]
print(f"Actions shape: {episode['actions'].shape}")
print(f"Observations: {episode['observations'].keys()}")
```

## Standard Evaluation Metrics

RoboKino defines four standardized metrics for unified evaluation:

### Primary Metrics

1. **Task Completion Time (TCT)**: Time taken to complete the task successfully
2. **Success Rate (SR)**: Percentage of successful task completions
3. **Dangerous Behavior Rate (DBR)**: Frequency of unsafe actions (collisions, constraint violations)
4. **Task Completion Proportion (TCP)**: Percentage of subtasks completed (for partial success assessment)

### Additional Analysis

- **Stage-wise Progress**: Completion status of individual task stages
- **Inter-arm Coordination**: Synchronization and coordination quality metrics
- **Trajectory Quality**: Smoothness, efficiency, and spatial accuracy

## Assets Library

RoboKino includes 500+ interactive assets:

### Asset Categories

- **Containers**: Boxes, bins, drawers, cabinets
- **Kitchen Items**: Pots, utensils, appliances
- **Office Objects**: Documents, staplers, organizers
- **Tools**: Screwdrivers, wrenches, assembly components
- **Articulated Objects**: Doors, valves, levers

### Using Assets

```python
from robokino.assets import AssetLibrary

# Load asset library
library = AssetLibrary()

# Get asset by category
kitchen_assets = library.get_by_category("kitchen")

# Spawn asset in scene
pot = library.spawn_asset("pot_001", position=[0.5, 0, 0.8])
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution

- New atomic controllers
- Additional task scenarios
- Improved evaluation metrics
- Dataset expansion
- Documentation improvements

## Citation

If you use RoboKino in your research, please cite our paper:

```bibtex
@article{robokino2026,
  title={RoboKino: A Scalable Benchmark for Dual-Arm Manipulation with Fine-Grained Evaluation},
  author={[Authors TBD]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT License

Copyright (c) 2026 vigorlee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- Isaac Sim team for the powerful simulation platform
- LeRobot team for the standardized data format
- Aloha project for the dual-arm platform inspiration
- The open-source robotics community

## Support

- **Documentation**: [Coming Soon]
- **Issues**: [GitHub Issues](https://github.com/vigorlee/RoboKino/issues)
- **Discussions**: [GitHub Discussions](https://github.com/vigorlee/RoboKino/discussions)

---

*RoboKino establishes a solid foundation for scalable dual-arm robotic systems with community-wide reproducibility.*
