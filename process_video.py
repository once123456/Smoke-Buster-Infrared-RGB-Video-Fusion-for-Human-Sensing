import warnings
# 忽略 FutureWarning 警告
warnings.filterwarnings("ignore", category=FutureWarning)
import cv2
import torch
import numpy as np
from PIL import Image
from fusion import laplacian_pyramid_fusion
from recognition import detect_humans
from image_haze_removel import image_haze_removel
from dehaze import DarkChannel, AtmLight, TransmissionEstimate, TransmissionRefine, Recover


# 打开可见光和红外视频文件
visible_video = cv2.VideoCapture('rgb.mp4')
if not visible_video.isOpened():
    print("Error opening visible video file")
    exit()

infrared_video = cv2.VideoCapture('tr.mp4')
if not infrared_video.isOpened():
    print("Error opening infrared video file")
    visible_video.release()
    exit()

# 获取视频的帧率、宽度和高度
fps = visible_video.get(cv2.CAP_PROP_FPS)
width = int(visible_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(visible_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# 加载预训练的YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

kalman_filters = []

print("请选择去烟算法:")
print("1. 原有的去烟算法")
print("2. dehaze.py 中的去烟算法")
choice = input("请输入你的选择 (1 或 2): ")


while True:
    # 读取可见光和红外视频的帧
    ret_visible, visible_frame = visible_video.read()
    ret_infrared, infrared_frame = infrared_video.read()

    # 如果任一视频读取结束，则退出循环
    if not ret_visible or not ret_infrared:
        break

    # 以灰度模式读取红外帧
    infrared_frame = cv2.cvtColor(infrared_frame, cv2.COLOR_BGR2GRAY)

    # 进行拉普拉斯金字塔融合
    fused_frame = laplacian_pyramid_fusion(visible_frame, infrared_frame)
    if choice == '1':
    # 去烟处理
        pil_image = Image.fromarray(cv2.cvtColor(fused_frame, cv2.COLOR_BGR2RGB))
        dehaze_image = image_haze_removel(pil_image)
        dehaze_image = dehaze_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        dehaze_image = np.clip(dehaze_image * 255, 0, 255).astype(np.uint8)
        dehaze_frame = cv2.cvtColor(dehaze_image, cv2.COLOR_RGB2BGR)
    elif choice == '2':
        I = fused_frame.astype('float64') / 255
        dark = DarkChannel(I, 15)
        A = AtmLight(I, dark)
        te = TransmissionEstimate(I, A, 15)
        t = TransmissionRefine(fused_frame, te)
        dehaze_frame = Recover(I, t, A, 0.1)
        dehaze_frame = np.clip(dehaze_frame * 255, 0, 255).astype(np.uint8)
    else:
        print("无效的选择，请重新运行程序并输入 1 或 2。")
        break

    # 检测人体并标注
    annotated_frame, kalman_filters = detect_humans(dehaze_frame, model, kalman_filters)

    # 将处理后的帧写入输出视频
    out.write(annotated_frame)

# 释放视频捕获和写入对象
visible_video.release()
infrared_video.release()
out.release()
    