# Fracture Field Drone Watcher

这是一个基于 YOLOv8 的自动驾驶/游戏辅助工具，用于识别并唤醒屏幕上处于休眠状态的采矿无人机。

项目采用**主动学习（Active Learning）**的闭环流程：
1. **冷启动**：无模型时，使用传统模板匹配算法进行初步识别和数据收集。
2. **人工反馈**：用户只需在 `logs/` 文件夹中删除错误的截图。
3. **难例挖掘 (Hard Negative Mining)**：脚本自动将删除的错误样本标记为“负样本”（背景），让 AI 学习“什么不是无人机”。
4. **自动训练**：一键训练，支持左右镜像增强，自动迭代模型。

## 📁 目录结构

```
fracture_field/
├── run_ai.py          # [主程序] 负责监控、识别、操作。会自动选择最新的 best.pt 模型。
├── train_model.py     # [训练] 模型训练脚本。支持 Windows 优化 (workers=0) 和镜像增强。
├── clean_dataset.py   # [清洗] 数据清洗脚本。根据 logs 文件夹的内容自动修正数据集标签。
├── assets/            # [资源] 存放模板图片 (template_off.png, template_on.png)。
├── dataset/           # [数据] YOLO 格式数据集。
│   ├── images/train/  # 存放完整的训练截图
│   ├── labels/train/  # 存放对应的 YOLO 格式标签
│   └── data.yaml      # 类别配置 (0: drone_off, 1: drone_on)
├── logs/              # [日志] 存放识别到的目标裁剪图 (用于人工快速审核)
└── runs/              # [模型] 存放训练生成的模型 (detect/train/weights/best.pt)
```

## 🚀 快速开始

### 1. 环境准备

请确保安装 Python 3.8+，并安装依赖：

```bash
pip install ultralytics opencv-python pyautogui mss numpy
```

### 2. 核心工作流 (Loop)

本项目设计为**边用边学**的循环。你越用，它越强。

#### 🟢 步骤一：运行与收集 (Run)
运行主程序：
```bash
python run_ai.py
```
- **如果没有模型**：显示 `Using TEMPLATE MATCHING fallback`。它使用传统算法寻找无人机。
- **如果有模型**：自动加载 `runs/detect` 下最新的 `best.pt`，使用 YOLOv8 进行识别。
- **数据收集**：
    - 程序会将**全屏截图**保存在 `dataset/images/train/`。
    - 程序会将**识别目标的裁剪图**保存在 `logs/`。

#### 🟡 步骤二：人工审核 (Filter)
运行一段时间后，停止程序。打开项目根目录下的 `logs/` 文件夹：
1. **浏览图片**：快速查看裁剪的小图。
2. **删除错误**：
    - 如果图片里**不是无人机**（是石头、UI、背景等），**直接删除该文件**。
    - 如果图片里**是无人机**，**保留不动**。

#### 🟠 步骤三：同步清洗 (Clean)
运行清洗脚本：
```bash
python clean_dataset.py
```
- 脚本会对比 `logs/` 和 `dataset/`。
- **你删除的**：脚本会找到对应的原始标注文件，将其对应的标签**清空**（变成负样本/背景）。这让 AI 学会“这个东西看起来像，但不是”。
- **你保留的**：脚本会确认该标签有效。

#### 🔴 步骤四：模型训练 (Train)
运行训练脚本：
```bash
python train_model.py
```
- **配置优化**：
    - `workers=0` (防止 Windows 下多进程报错)。
    - `fliplr=0.5` (开启水平翻转增强，不需要区分左右朝向，数据量等效翻倍)。
    - `epochs=100` (默认训练 100 轮)。
- 训练完成后，新模型会保存在 `runs/detect/trainX/weights/best.pt`。

#### 🔵 回到步骤一
- 再次运行 `python run_ai.py`。
- 它会自动检测到新训练的模型，加载并开始工作。
- 此时 AI 的**准确率 (Precision)** 应该显著提升，误报减少。

## ⚙️ 类别说明

以前版本区分左右 (l_off, r_off 等)，现在简化为核心状态：
- `0: drone_off` (休眠/灰色) - 需要唤醒的目标
- `1: drone_on` (工作/彩色) - 正在工作的目标

*利用 `train_model.py` 中的 `fliplr` 增强，无需手动收集左右两侧的数据。*

## ⌨️ 快捷键

- `Ctrl + C`: 在终端中停止程序运行。
