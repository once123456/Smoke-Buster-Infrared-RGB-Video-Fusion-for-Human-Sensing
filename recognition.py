import cv2
import numpy as np
import torch


class CustomKalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(8, 4)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], np.float32)
        self.kalman.processNoiseCov = 1e-5 * np.eye(8, dtype=np.float32)
        self.kalman.measurementNoiseCov = 1e-1 * np.eye(4, dtype=np.float32)
        self.unmatched_count = 0


def detect_humans(frame, model, kalman_filters, unmatched_threshold=3, distance_threshold=50):
    # 进行目标检测
    results = model(frame)
    # 获取检测结果中的人体边界框信息
    detections = results.pandas().xyxy[0]
    humans = detections[detections['name'] == 'person']

    new_kalman_filters = []
    matched_kalman = []
    for _, row in humans.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1

        # 寻找最匹配的卡尔曼滤波器
        min_distance = float('inf')
        best_kalman = None
        for kalman in kalman_filters:
            prediction = kalman.kalman.predict()
            predicted_x = int(prediction[0].item())
            predicted_y = int(prediction[1].item())
            distance = np.sqrt((center_x - predicted_x) ** 2 + (center_y - predicted_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                best_kalman = kalman

        if best_kalman is not None and min_distance < distance_threshold:
            # 更新匹配的卡尔曼滤波器
            measurement = np.array([[center_x], [center_y], [width], [height]], dtype=np.float32)
            best_kalman.kalman.correct(measurement)
            new_kalman_filters.append(best_kalman)
            matched_kalman.append(best_kalman)
            best_kalman.unmatched_count = 0  # 重置未匹配计数
        else:
            # 创建新的卡尔曼滤波器
            kalman = CustomKalmanFilter()
            kalman.kalman.statePre = np.array([[center_x], [center_y], [width], [height], [0], [0], [0], [0]],
                                              np.float32)
            new_kalman_filters.append(kalman)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 对于未匹配的卡尔曼滤波器，使用预测结果
    for kalman in kalman_filters:
        if kalman not in matched_kalman:
            kalman.unmatched_count += 1
            if kalman.unmatched_count < unmatched_threshold:
                prediction = kalman.kalman.predict()
                predicted_x = int(prediction[0].item())
                predicted_y = int(prediction[1].item())
                predicted_width = int(prediction[2].item())
                predicted_height = int(prediction[3].item())
                x1 = predicted_x - predicted_width // 2
                y1 = predicted_y - predicted_height // 2
                x2 = predicted_x + predicted_width // 2
                y2 = predicted_y + predicted_height // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, 'Predicted Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                new_kalman_filters.append(kalman)

    return frame, new_kalman_filters
