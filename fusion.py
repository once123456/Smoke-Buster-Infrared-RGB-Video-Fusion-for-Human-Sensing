import cv2
import numpy as np
import torch

def laplacian_pyramid_fusion(visible_image_path, infrared_image_path, levels=5):
    # 读取可见光图片
    visible_image = cv2.imread(visible_image_path)
    # 以灰度模式读取红外图片
    infrared_image = cv2.imread(infrared_image_path, 0)
    # 调整红外图片尺寸与可见光图片一致
    infrared_image = cv2.resize(infrared_image, (visible_image.shape[1], visible_image.shape[0]))
    # 将红外图片转换为三通道图像
    infrared_image = cv2.cvtColor(infrared_image, cv2.COLOR_GRAY2BGR)

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
    visible_pyramid = build_laplacian_pyramid(visible_image, levels)
    infrared_pyramid = build_laplacian_pyramid(infrared_image, levels)
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

    fused_image = reconstruct_image(fused_pyramid)
    return fused_image

def detect_humans(image):
    # 加载预训练的YOLOv5模型
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # 进行目标检测
    results = model(image)
    # 获取检测结果中的人体边界框信息
    detections = results.pandas().xyxy[0]
    humans = detections[detections['name'] == 'person']
    # 在图片上绘制边界框
    for _, row in humans.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# 示例调用
visible_image_path = 'imageset/fusion2(3).jpg'
infrared_image_path = 'imageset/fusion2（2）honwai.png'
fused_image = laplacian_pyramid_fusion(visible_image_path, infrared_image_path)
# 检测人体并标注
annotated_image = detect_humans(fused_image)
# 保存标注后的融合图片
cv2.imwrite('laplacian_annotated_fused_image.jpg', annotated_image)
