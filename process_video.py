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

import os
from glob import glob
import tensorflow as tf

# 导入RetinexNet相关函数和类
from RetinexNet_master.model import LowLightEnhance
from RetinexNet_master.utils import load_images, save_images
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


# 初始化RetinexNet模型
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 设置内存增长模式
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"设置 GPU 内存增长模式时出现错误: {e}")

tf.compat.v1.disable_eager_execution()

with tf.compat.v1.Session() as sess:
    model_retinex = LowLightEnhance(sess)
    frame_count = 0
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

        # 保存去雾后的图片
        dehaze_image_path = f'dehaze_frame_{frame_count}.png'
        cv2.imwrite(dehaze_image_path, dehaze_frame)

        # 使用RetinexNet处理去雾后的图片
        test_low_data = [load_images(dehaze_image_path)]
        test_high_data = []
        test_low_data_names = [dehaze_image_path]
        save_dir = './retinex_results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        decom_flag = 0
        model_retinex.test(test_low_data, test_high_data, test_low_data_names, save_dir=save_dir, decom_flag=decom_flag)

        # 读取RetinexNet处理后的图片
        [_, name] = os.path.split(dehaze_image_path)
        suffix = name[name.find('.') + 1:]
        name = name[:name.find('.')]
        result_path = os.path.join(save_dir, name + "_S." + suffix)
        retinex_image = cv2.imread(result_path)

        # 检测人体并标注
        annotated_frame, kalman_filters = detect_humans(retinex_image, model, kalman_filters)

        # 将处理后的帧写入输出视频
        out.write(annotated_frame)

        frame_count += 1

# 释放视频捕获和写入对象
visible_video.release()
infrared_video.release()
out.release()

# 删除临时保存的去雾图片
for file in glob('dehaze_frame_*.png'):
    os.remove(file)