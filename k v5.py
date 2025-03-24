import warnings
# 忽略 FutureWarning 警告
warnings.filterwarnings("ignore", category=FutureWarning)
import cv2
import numpy as np
import torch

def laplacian_pyramid_fusion(visible_frame, infrared_frame, levels=5):
    # 调整红外图片尺寸与可见光图片一致
    infrared_frame = cv2.resize(infrared_frame, (visible_frame.shape[1], visible_frame.shape[0]))
    # 将红外图片转换为三通道图像
    infrared_frame = cv2.cvtColor(infrared_frame, cv2.COLOR_GRAY2BGR)

    # 构建拉普拉斯金字塔
    def build_laplacian_pyramid(img, levels):
        pyramid = []
        current_img = img
        for i in range(levels):
            lower = cv2.pyrDown(current_img)
            higher = cv2.pyrUp(lower, dstsize=(current_img.shape[1], current_img.shape[0]))
            laplacian = cv2.subtract(current_img, higher)
            pyramid.append(laplacian)
            current_img = lower
        pyramid.append(current_img)
        return pyramid

    # 融合拉普拉斯金字塔
    visible_pyramid = build_laplacian_pyramid(visible_frame, levels)
    infrared_pyramid = build_laplacian_pyramid(infrared_frame, levels)
    fused_pyramid = []
    for i in range(levels + 1):
        # 简单地取两者的平均值进行融合
        fused = (visible_pyramid[i] + infrared_pyramid[i]) // 2
        fused_pyramid.append(fused)

    # 重构图像
    def reconstruct_image(pyramid):
        current_img = pyramid[-1]
        for i in range(len(pyramid) - 2, -1, -1):
            upsampled = cv2.pyrUp(current_img, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
            current_img = cv2.add(upsampled, pyramid[i])
        return current_img

    fused_frame = reconstruct_image(fused_pyramid)
    return fused_frame

def detect_humans(frame, model, kalman_filters):
    # 进行目标检测
    results = model(frame)
    # 获取检测结果中的人体边界框信息
    detections = results.pandas().xyxy[0]
    humans = detections[detections['name'] == 'person']

    new_kalman_filters = []
    for _, row in humans.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 寻找最匹配的卡尔曼滤波器
        min_distance = float('inf')
        best_kalman = None
        for kalman in kalman_filters:
            prediction = kalman.predict()
            predicted_x = int(prediction[0].item())
            predicted_y = int(prediction[1].item())
            distance = np.sqrt((center_x - predicted_x) ** 2 + (center_y - predicted_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                best_kalman = kalman

        if best_kalman is not None:
            # 更新匹配的卡尔曼滤波器
            measurement = np.array([[center_x], [center_y]], dtype=np.float32)
            best_kalman.correct(measurement)
            new_kalman_filters.append(best_kalman)
        else:
            # 创建新的卡尔曼滤波器
            kalman = cv2.KalmanFilter(4, 2)
            kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            kalman.processNoiseCov = 1e-5 * np.eye(4, dtype=np.float32)
            kalman.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)
            kalman.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
            new_kalman_filters.append(kalman)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 对于未匹配的卡尔曼滤波器，使用预测结果
    for kalman in kalman_filters:
        if kalman not in new_kalman_filters:
            prediction = kalman.predict()
            predicted_x = int(prediction[0].item())
            predicted_y = int(prediction[1].item())
            # 简单假设边界框大小
            x1 = predicted_x - 20
            y1 = predicted_y - 40
            x2 = predicted_x + 20
            y2 = predicted_y + 40
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, 'Predicted Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            new_kalman_filters.append(kalman)

    return frame, new_kalman_filters

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

    # 检测人体并标注
    annotated_frame, kalman_filters = detect_humans(fused_frame, model, kalman_filters)

    # 将处理后的帧写入输出视频
    out.write(annotated_frame)

# 释放视频捕获和写入对象
visible_video.release()
infrared_video.release()
out.release()