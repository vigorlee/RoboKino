# RoboKino

**双臂操作仿真、专家数据采集与策略评估。** [English](README.md)

[![RoboKino 平台总览](docs/images/robokino-framework.png)](docs/images/robokino-framework.pdf)

上图由 RoboKino 作者提供，展示项目整体设计。当前代码的发布范围见下表。

## 当前包含的功能

本仓库已导入 [FluxBisim](https://github.com/FluxVLA/FluxBisim) 的仿真实现，包含双臂任务控制器、场景与机器人配置、专家轨迹采集和 ROS 策略评估接口。导入版本固定为 `621df43e08df3f27b5355e65e3f2c2918750bccb`，原始版权声明和 Apache 2.0 许可证完整保留。

| 原子技能 | 示例任务 | 场景 |
| --- | --- | --- |
| 抓取 / 放置 | 将水果放入盘中 | `kitchen`、`apartment` |
| 打开 / 关闭 | 放入螺母后关闭盒子 | `industry` |
| 双臂交接 | 交接并收纳书籍 | `apartment`、`industry` |
| 拉 / 推 | 将苹果放入抽屉 | `apartment` |
| 旋转 | 拧上水壶盖 | `apartmentshort` |

水果配置支持苹果、香蕉、胡萝卜、黄瓜、山竹和白萝卜。导入的机器人配置采用移动 ALOHA 风格底座和两条 Piper 机械臂。

## 安装与运行

下面使用 Linux/bash 和 **Isaac Sim 4.5.0** 的自带 Python。先安装兼容的 NVIDIA 驱动和仿真器，再执行：

```bash
git clone https://github.com/vigorlee/RoboKino.git
cd RoboKino

export ISAAC_SIM_PATH=/path/to/isaac-sim-4.5.0
"${ISAAC_SIM_PATH}/python.sh" -m pip install -r requirements.txt
alias benchmark_python="${ISAAC_SIM_PATH}/python.sh"
```

仿真所需 USD 资产需单独准备。现有配置可使用原始 [FluxBisimAssets](https://huggingface.co/datasets/limxdynamics/FluxBisimAssets)：

```bash
mkdir -p assets
hf download limxdynamics/FluxBisimAssets \
  --repo-type dataset \
  --local-dir assets
```

`hf` 命令需另行安装。该资产库由原作者维护，数据集卡标注许可证为 **CC BY-NC 4.0**。资产许可证与本仓库代码许可证分别适用。使用自己的 RoboKino 资产时，修改 `envs/cfg/objects.yaml` 和 `envs/cfg/robots.yaml`，并核对关节与 prim 名称。

## 数据采集

在仓库根目录启动香蕉抓取任务：

```bash
benchmark_python data_collect/pick_place_fruit/fruit_pick_place_collect.py \
  --env kitchen \
  --config banana_pick_place_config.yaml \
  --data_path data/robokino/pick_place_banana
```

修改 YAML 中的 `max_demo` 可控制成功示范数量。所有采集脚本支持 `--env`、`--config` 和 `--data_path`。采集程序使用图形界面，只保存成功回合，输出 HDF5 文件，包含三个摄像头观测、关节位置和动作。

**每次采集请使用新的输出目录；重复启动会复用 `episode_00000.hdf5` 等文件名。**

其余四类任务的完整命令、配置名和数据结构见 [英文 README](README.md#data-collection)。LeRobot v2.1 转换与模型训练使用外部 FluxVLA 工作流；本仓库尚未发布 RoboKino 专属训练集和训练代码。

## 策略评估

评估需安装 ROS 1 Noetic、RViz，并确保 Isaac Sim 的 Python 能导入 `rospy`、`cv_bridge`、ROS 消息类型和 OpenCV。`cv_bridge` 的二进制必须与 Python 版本兼容。

```bash
source /opt/ros/noetic/setup.bash
PYTHON_BIN="${ISAAC_SIM_PATH}/python.sh" \
  bash robokino_benchmark.sh pick_place_banana --num-episodes 100
```

另开终端启动匹配任务的策略推理客户端。可以对接 FluxVLA 的评估客户端，相关说明见 [英文 README](README.md#policy-evaluation)。支持的任务预设为 `close_box`、`handover_book`、`pick_place_banana`、`pull_push_drawer` 和 `screw_pitcher_lid`。

当前评估代码输出成功率和成功回合的完成步数统计。图中 TCT、DBR、TCP 及更完整的过程诊断属于后续实现和验证范围；此版本没有发布实测排行榜。

## 代码结构与检查

- `controllers/`：RMPflow 运动控制和五类任务控制器。
- `data_collect/`：数据采集脚本、YAML 参数与 HDF5 工具。
- `envs/`：Isaac Sim 环境、摄像头、机器人与场景配置。
- `tasks/`：任务定义、随机化与成功检查。
- `evaluate.py`：仿真与 ROS 策略接口。
- `docs/images/`：作者提供的 RoboKino 图片。
- `robokino_benchmark.sh`：RoboKino 启动入口。

不启动仿真器的仓库检查：

```bash
python -m pip install 'PyYAML>=6,<7'
python tools/check_repository.py
```

检查包含 Python 语法、YAML、任务配置对应关系、文档链接和图片文件完整性。实际运行仍需仿真器、GPU 和资产。

## 来源与许可证

导入的 FluxBisim 代码使用 [Apache 2.0](LICENSE)；RoboKino 新增文档、启动脚本和检查工具使用 [MIT](licenses/MIT-RoboKino.txt)。详情见 [THIRD_PARTY.md](THIRD_PARTY.md) 和 [NOTICE](NOTICE)。完整上游代码来源清单位于 [docs/upstream-import.json](docs/upstream-import.json)。

感谢 FluxVLA / LimX Dynamics、AgileX Robotics、NVIDIA Isaac Sim、LeRobot 及原始资产提供者。正式论文引用信息待发布；当前请引用 [RoboKino 仓库](https://github.com/vigorlee/RoboKino) 与实际使用的提交版本，并在使用上游实现时注明 FluxBisim。
