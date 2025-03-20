import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import image_data_loader
import lightdehazeNet
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm


def image_haze_removel(input_image):
    hazy_image = (np.asarray(input_image) / 255.0)
    hazy_image = torch.from_numpy(hazy_image).float()
    hazy_image = hazy_image.permute(2, 0, 1)
    hazy_image = hazy_image.cuda().unsqueeze(0)

    ld_net = lightdehazeNet.LightDehaze_Net().cuda()
    ld_net.load_state_dict(torch.load('trained_weights/trained_LDNet.pth'))

    dehaze_image = ld_net(hazy_image)
    return dehaze_image


def video_dehaze(input_video_path, output_video_path):
    # 打开输入视频文件
    cap = cv2.VideoCapture(input_video_path)

    # 获取视频的帧率、宽度、高度和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 定义视频编码器并创建输出视频文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 创建 tqdm 进度条
    progress_bar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # 将 OpenCV 的 BGR 格式转换为 PIL 的 RGB 格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)

            # 应用去雾算法
            dehaze_image = image_haze_removel(pil_image)
            dehaze_image = dehaze_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            dehaze_image = np.clip(dehaze_image * 255, 0, 255).astype(np.uint8)

            # 将 PIL 的 RGB 格式转换回 OpenCV 的 BGR 格式
            dehaze_frame = cv2.cvtColor(dehaze_image, cv2.COLOR_RGB2BGR)

            # 将去雾后的帧写入输出视频文件
            out.write(dehaze_frame)

            # 更新进度条
            progress_bar.update(1)
        else:
            break

    # 关闭进度条
    progress_bar.close()

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Dehazing')
    parser.add_argument('--input_video', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output_video', type=str, required=True, help='Path to the output video file')
    args = parser.parse_args()

    video_dehaze(args.input_video, args.output_video)
    