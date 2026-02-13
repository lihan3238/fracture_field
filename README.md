# Fracture Field Drone Watcher

这是一个基于 YOLOv8 的自动驾驶/游戏辅助工具，用于识别并唤醒屏幕上处于休眠状态的采矿无人机。

项目采用**自监督学习（Self-Supervised Learning）**的闭环流程：无需人工手动标注大量数据，而是通过传统算法（模板匹配）进行“冷启动”预标注，人工简单清洗后，训练出强大的 AI 模型。

## 📁 目录结构

```
fracture_field/
├── run_ai.py          # [核心] 主运行程序。负责监控、识别、操作以及数据收集。
├── train_model.py     # [训练] 模型训练程序。自动划分数据集并微调 YOLOv8。
├── assets/            # [资源] 只需存放 l_off.png 和 l_on.png (程序自动生成镜像以匹配右侧)。
├── dataset/           # [数据] 存放自动生成的训练图片和标注。
│   ├── images/
│   ├── labels/
│   └── data.yaml
└── README.md          # 说明文档
```

## 🚀 快速开始

### 1. 环境准备

请确保安装了 Python 3.8+，并安装以下依赖：

```bash
pip install ultralytics opencv-python pyautogui mss numpy
```

### 2. 工作流循环

本项目设计为可无限迭代的闭环，请按顺序执行：

#### 🟢 阶段一：冷启动 (无模型)
直接运行主程序：
```bash
python run_ai.py
```
- 程序会显示 **"Using TEMPLATE MATCHING fallback"**。
- 它会使用传统的模板匹配算法尝试寻找无人机。
- **关键功能**：它会自动截取屏幕，生成 YOLO 格式的标注文件，保存在 `dataset/images/train` 中。
- **建议**：运行 1-2 分钟，让无人机出现在屏幕不同位置，收集约 30-50 张截图后按 `Ctrl+C` 停止。

#### 🟡 阶段二：数据清洗 (关键)
打开文件夹 `dataset/images/train`：
1. 浏览自动生成的截图。
2. **删除**那些明显识别错误（框错了地方，或者框里甚至没有无人机）的图片。
3. **重要**：同时手动删除 `dataset/labels/train` 中对应的同名 `.txt` 文件。

#### 🟠 阶段三：模型进化
运行训练脚本：
```bash
python train_model.py
```
- 脚本会自动将 20% 的清洗数据划分为验证集。
- 自动下载 `yolov8n.pt` 并开始迁移学习。
- 训练完成后，模型保存在 `runs/detect/train/weights/best.pt`。

#### 🔴 阶段四：完全体 (AI 模式)
再次运行主程序：
```bash
python run_ai.py
```
- 程序检测到模型存在，自动切换为 **AI 模式**。
- 此时识别抗遮挡能力、大小变化适应力大幅提升。
- 程序会继续在后台收集新的数据（标记为 AI 产生的）。
- **循环**：如果你发现 AI 还有误判，重复阶段二和三，你的 AI 会越来越强。

## ⚙️ 配置说明

在 `run_ai.py` 中可以调整以下头部参数：

```python
CONF_THRESHOLD = 0.5    # AI 模式下的置信度阈值
COLLECT_DATA = True     # 是否继续收集训练数据 (觉得模型够强了可以关掉)
```

## ⌨️ 快捷键

- `Ctrl + C`: 停止程序
