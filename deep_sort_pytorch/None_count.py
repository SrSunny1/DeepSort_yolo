import cv2
from deep_sort.deep_sort import DeepSort
from ultralytics import YOLO
import numpy as np

# 加载 YOLOv8 模型
model = YOLO("yolov8n.pt")  # 使用预训练的 YOLOv8 模型

track_path = "ckpt.t7"
# 初始化 DeepSORT 跟踪器
tracker = DeepSort(max_age=50, model_path=track_path)  # max_age 是跟踪目标的最大存活帧数

# 打开视频文件或摄像头
video_path = "person.mp4"  # 替换为你的视频文件路径
cap = cv2.VideoCapture(video_path)

# 定义车辆类别 ID（YOLOv8 的 COCO 数据集中，car 的类别 ID 是 2）
vehicle_class_id = 0  # 如果是检测人，改为0

tracked_vehicles = set()  # 用于存储已跟踪的车辆 ID
track_history = {}  # 用于存储每个轨迹的历史位置

while cap.isOpened():
    ret, frame = cap.read()  # frame是当前帧的图像数据，为Numpy数组(heights,weights,channels)
    if not ret:
        break

    # 使用 YOLOv8 进行目标检测
    results = model(frame, stream=True)

    # 提取检测结果 - 优化版
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)  # 类别 ID
            if class_id == vehicle_class_id:  # 只处理车辆
                # 直接获取中心坐标和宽高
                xywh = box.xywh[0].cpu().numpy()
                confidence = float(box.conf)  # 置信度
                detections.append((xywh, confidence))

    # 转换为 NumPy 数组
    if detections:
        boxes = np.array([det[0] for det in detections])
        confidences = np.array([det[1] for det in detections])
    else:
        boxes = np.empty((0, 4))
        confidences = np.empty(0)

    # 使用 DeepSORT 进行目标跟踪
    tracks = tracker.update(boxes, confidences, frame)

    # 处理跟踪结果
    for track in tracks:
        # 提取跟踪框坐标和 ID
        x1, y1, x2, y2 = map(int, track[:4])  # 前4个元素是边界框坐标
        track_id = int(track[4])  # 第5个元素是track_id

        # 绘制矩形框和ID
        label = f"ID-{track_id}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1 - t_size[1] - 2), (x1 + t_size[0] + 2, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, [255, 0, 0], 1)

        # 计算中心点坐标
        center_y = (y1 + y2) // 2

        # 初始化或更新轨迹历史
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append(center_y)

        # 只保留最近5个位置点
        if len(track_history[track_id]) > 5:
            track_history[track_id].pop(0)

    # 显示视频帧
    cv2.imshow("Traffic Counting", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()