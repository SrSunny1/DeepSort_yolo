import cv2
from deep_sort.deep_sort import DeepSort
from ultralytics import YOLO
import numpy as np

# 加载 YOLOv8 模型
model = YOLO("yolov8n.pt")  # 使用预训练的 YOLOv8 模型

track_path = "ckpt.t7"
# 初始化 DeepSORT 跟踪器
tracker = DeepSort(max_age=50, model_path=track_path)

video_path = "person.mp4"
cap = cv2.VideoCapture(video_path)

# 视频处理参数
resize_ratio = 0.5  # 缩小处理分辨率
detect_interval = 2  # 每隔多少帧进行一次检测
frame_skip = 0  # 跳帧处理 (0表示不跳过)

# 定义车辆类别 ID
vehicle_class_id = 0

# 初始化计数器
vehicle_count = 0
tracked_vehicles = set()

# 定义计数线的位置
count_line_y = int(300 * resize_ratio)  # 按比例调整计数线位置

def xyxy_xcyc(x1, y1, x2, y2):
    x_c = x1 + (x2 - x1) / 2
    y_c = y1 + (y2 - y1) / 2
    w = x2 - x1
    h = y2 - y1
    return x_c, y_c, w, h

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 跳帧处理
    frame_count += 1
    if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
        continue

    # 降低分辨率
    small_frame = cv2.resize(frame, (0, 0), fx=resize_ratio, fy=resize_ratio)
    display_frame = small_frame.copy()

    detections = []
    # 每隔detect_interval帧进行一次检测
    if frame_count % detect_interval == 0:
        # 使用 YOLOv8 进行目标检测
        results = model(small_frame, stream=True)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id == vehicle_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x_c, y_c, w, h = xyxy_xcyc(x1, y1, x2, y2)
                    confidence = float(box.conf)
                    detections.append(([x_c, y_c, w, h], confidence, class_id))

    # 准备跟踪输入
    if len(detections) > 0:
        boxes = np.array([det[0] for det in detections])
        confidences = np.array([det[1] for det in detections])
        # 使用 DeepSORT 进行目标跟踪
        tracks = tracker.update(boxes, confidences, small_frame)
    else:
        # 传递空的 NumPy 数组而不是空列表
        tracks = tracker.update(np.empty((0, 4)), np.empty((0,)), small_frame)  # 只进行跟踪

    # 处理跟踪结果
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track[:5])

        # 绘制矩形框和ID (简化绘制)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(display_frame, str(track_id), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 判断车辆是否穿过计数线
        if y1 < count_line_y < y2 and track_id not in tracked_vehicles:
            tracked_vehicles.add(track_id)
            vehicle_count += 1

    # 绘制计数线
    cv2.line(display_frame, (0, count_line_y),
             (display_frame.shape[1], count_line_y), (0, 0, 255), 2)

    # 显示车流量计数
    cv2.putText(display_frame, f"Count: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 显示处理后的帧
    cv2.imshow("Traffic Counting", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()