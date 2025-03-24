import cv2
import numpy as np


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
    