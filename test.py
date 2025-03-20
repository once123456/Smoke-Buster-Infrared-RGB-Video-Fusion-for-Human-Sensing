import cv2
import numpy as np
import torch

def laplacian_pyramid_fusion(visible_frame, infrared_frame, levels=5):
    # 调整红外帧尺寸与可见光帧一致
    infrared_frame = cv2.resize(infrared_frame, (visible_frame.shape[1], visible_frame.shape[0]))
    # 将红外帧转换为三通道图像
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

def detect_humans(frame):
    # 加载预训练的YOLOv5模型
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # 进行目标检测
    results = model(frame)
    # 获取检测结果中的人体边界框信息
    detections = results.pandas().xyxy[0]
    humans = detections[detections['name'] == 'person']
    # 在图片上绘制边界框
    for _, row in humans.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# 打开可见光和红外视频文件
visible_video = cv2.VideoCapture('visible_video.mp4')
infrared_video = cv2.VideoCapture('infrared_video.mp4')

# 获取视频的帧率、宽度和高度
fps = visible_video.get(cv2.CAP_PROP_FPS)
width = int(visible_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(visible_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('laplacian_annotated_fused_video.mp4', fourcc, fps, (width, height))

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
    annotated_frame = detect_humans(fused_frame)

    # 写入处理后的帧到输出视频
    out.write(annotated_frame)

    # 显示处理后的帧（可选）
    cv2.imshow('Annotated Fused Frame', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获和写入对象，关闭所有窗口
visible_video.release()
infrared_video.release()
out.release()
cv2.destroyAllWindows()