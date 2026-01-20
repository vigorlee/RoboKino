# RoboKino

## About

**RoboKino** is an Isaac Sim-based benchmark and data-generation framework for fine-grained dual-arm manipulation. It extends the Aloha dual-arm platform in simulation and leverages Isaac Sim's photorealistic rendering capabilities.

RoboKino structures dual-arm behaviors into five cooperative atomic controllers that compose long-horizon task sequences. It supports controlled task variants with domain randomization and provides diagnostic evaluation beyond binary success via stage-wise progress checks and coordination analysis.

### Key Features

- **5 Atomic Dual-Arm Controllers**: Composable controllers for complex collaborative tasks
- **10,000+ Expert Trajectories**: LeRobot v2.1-compliant dataset
- **500 Interactive Assets**: Reusable assets spanning domestic, desktop/office, and industrial scenes
- **Domain Randomization**: Enhanced task diversification and generalization
- **Isaac Sim Integration**: Photorealistic rendering and physics simulation
- **Aloha-Compatible**: Hardware-aligned dual-arm configuration
- **Fine-Grained Evaluation**: Stage-wise progress assessment and coordination analysis
- **End-to-End Pipeline**: Unified data acquisition, training, and inference workflow



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
Kitchen and household tasks requiring dual-arm coordination (cleaning, organizing).

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

3. **By adding mappings in the .bashrc file, you can invoke the Python environment using the mapped variables**

```bash
  alias benchmark_python='~/.local/share/ov/pkg/isaac-sim-<version>/python.sh' #添加映射
  benchmark_python <python_file_name> #运行代码
```

4. **Install the package**

```bash
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
    ~/.local/share/ov/pkg/isaac-sim-4.2.0/python.sh -m pip install -r requirements.txt
```

## Project Structure

BENCHMARK_ENV
├── ACT              # (For reference) Training and inference code for ACT models based on this dataset
├── assets           # Assets related to scenes, robots, and objects
├── controllers      # Robotic arm motion controllers
├── data_collect     # Data collection pipelines for tasks
├── envs             # Simulator configuration and environment setup
├── tasks            # Task randomization and evaluation metric definitions

## Training Data Collection

### Fruit Pick-and-Place in Kitchen and Living Room Environments
```bash
    benchmark_python data_collect/pick_place_fruit/fruit_pick_place_collect.py --table <TABLE> --config <CONFIG> --base <BASE>
```
#### Table
- `kitchen_table`
- `apartment_table`
#### Config
- `apple_pick_place_config.yaml`
- `banana_pick_place_config.yaml`
- `carrot_pick_place_config.yaml`
- `cucumber_pick_place_config.yaml`
- `mangosteen_pick_place_config.yaml`
- `whiteradish_pick_place_config.yaml`
### Base
- `base_kitchen`
- `base_apartment`

## Standard Evaluation Metrics

RoboKino defines four standardized metrics for unified evaluation:

### Primary Metrics

1. **Task Completion Time (TCT)**: Time taken to complete the task successfully
2. **Success Rate (SR)**: Percentage of successful task completions
3. **Dangerous Behavior Rate (DBR)**: Frequency of unsafe actions (collisions, constraint violations)
4. **Task Completion Proportion (TCP)**: Percentage of subtasks completed (for partial success assessment)

### Additional Analysis

- **Stage-wise Progress**: Completion status of individual task stages
- **Trajectory Quality**: Smoothness, efficiency, and spatial accuracy

## Assets Library

RoboKino includes 500+ interactive assets:

### Asset Categories

- **Containers**: Boxes, bins, drawers, cabinets
- **Kitchen Items**: Pots, utensils, appliances
- **Office Objects**: Documents, staplers, organizers
- **Tools**: Screwdrivers, wrenches, assembly components
- **Articulated Objects**: Doors, valves

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

<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=vigorlee/RoboKino&type=Date)](https://star-history.com/#vigorlee/RoboKino&Date)
-->

---

*RoboKino establishes a solid foundation for scalable dual-arm robotic systems with community-wide reproducibility.*
