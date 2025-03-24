# Smoke Buster: Infrared-RGB Video Fusion for Human Sensing  
![Python](https://img.shields.io/badge/Python-100%25-blue?logo=python) [![Stars](https://img.shields.io/github/stars/once123456/Smoke-Buster-Infrared-RGB-Video-Fusion-for-Human-Sensing)](https://github.com/once123456/Smoke-Buster-Infrared-RGB-Video-Fusion-for-Human-Sensing/stargazers) [^1]


## 项目概述
本项目主要用于处理可见光和红外视频，将两者进行融合，去除融合后视频中的烟雾，并且识别视频中的人体目标。项目主要由三个核心部分组成：图像融合、图像去烟以及人体识别。

## 项目结构
项目包含以下几个主要文件：
1. **`fusion.py`**：实现 RGB 图像和红外图像的拉普拉斯金字塔融合。
2. **`recognition.py`**：进行人体识别，使用预训练的 YOLOv5 模型，并结合卡尔曼滤波器对人体进行追踪。
3. **`image_haze_removel.py`**：使用预训练的 LightDehaze_Net 模型对图像进行去烟处理。
4. **`process_video.py`**：主程序文件，读取可见光和红外视频，依次调用图像融合、去烟和人体识别功能，最后将处理后的帧写入输出视频。

## 环境要求
- Python 3.9 及以上版本
- 依赖库：
  - `torch`
  - `cv2`（OpenCV）
  - `numpy`
  - `PIL`（Pillow==9.5.0）

## 安装依赖
在项目根目录下，使用以下命令安装所需的依赖库：
```bash
pip install torch torchvision opencv-python numpy pillow
```

## 运行步骤
1. **准备数据**：
   - 将可见光视频命名为 `rgb.mp4`，红外视频命名为 `tr.mp4`，并将它们放在项目根目录下。
   - 确保 `trained_weights` 文件夹存在，并且其中包含预训练的 `trained_LDNet.pth` 模型文件。
2. **运行主程序**：
   在项目根目录下，运行以下命令启动视频处理程序：
   ```bash
   python process_video.py
   ```
3. **查看结果**：
   处理完成后，会在项目根目录下生成一个名为 `output_video.mp4` 的视频文件，其中包含融合、去烟和人体识别后的结果。

## 代码说明

### `fusion.py`
该文件中的 `laplacian_pyramid_fusion` 函数实现了 RGB 图像和红外图像的拉普拉斯金字塔融合。具体步骤包括：
1. 调整红外图像的尺寸，使其与可见光图像一致。
2. 将红外图像转换为三通道图像。
3. 构建拉普拉斯金字塔，并将可见光和红外图像的金字塔进行融合。
4. 重构融合后的图像。

### `recognition.py`
`detect_humans` 函数用于进行人体识别。具体步骤如下：
1. 使用预训练的 YOLOv5 模型对输入图像进行目标检测。
2. 筛选出检测结果中的人体目标。
3. 为每个检测到的人体目标分配或创建一个卡尔曼滤波器，用于追踪目标。
4. 在图像上绘制人体边界框和标签。

### `image_haze_removel.py`
`image_haze_removel` 函数使用预训练的 LightDehaze_Net 模型对输入图像进行去烟处理。具体步骤如下：
1. 将输入图像转换为 PyTorch 张量，并将其移动到 GPU 上。
2. 加载预训练的 LightDehaze_Net 模型。
3. 使用模型对图像进行去烟处理。
4. 将处理后的图像转换回 NumPy 数组，并调整其格式。

### `process_video.py`
主程序文件，主要步骤如下：
1. 打开可见光和红外视频文件。
2. 获取视频的帧率、宽度和高度，并创建输出视频文件。
3. 加载预训练的 YOLOv5 模型。
4. 逐帧读取视频，对每一帧进行图像融合、去烟和人体识别处理。
5. 将处理后的帧写入输出视频文件。
6. 释放视频捕获和写入对象。

## 注意事项
- 请确保你的系统支持 CUDA，以便在 GPU 上运行去烟和人体识别模型，提高处理速度。
- 若视频文件路径或名称发生变化，请相应地修改 `process_video.py` 文件中的相关代码。
- 若预训练模型文件路径或名称发生变化，请相应地修改 `image_haze_removel.py` 文件中的相关代码。
