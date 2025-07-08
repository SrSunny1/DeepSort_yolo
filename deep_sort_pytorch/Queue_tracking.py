import cv2
from deep_sort.deep_sort import DeepSort
from ultralytics import YOLO
import numpy as np
import queue

# 加载 YOLOv8 模型
model = YOLO("yolov8n.pt")  # 使用预训练的 YOLOv8 模型

# 加载权重文件路径
# track_path = r'D:\机器学习—兴趣\车流量统计\Yolov5_DeepSort_Traffic-counter-main\deep_sort_pytorch\deep_sort\deep\checkpoint\ckpt.t7'

track_path = "ckpt.t7"
# 初始化 DeepSORT 跟踪器
tracker = DeepSort(max_age = 50, model_path=track_path)  # max_age 是跟踪目标的最大存活帧数

# 打开视频文件或摄像头
video_path = "videoa.mp4"
cap = cv2.VideoCapture(video_path)

# 定义车辆类别 ID（YOLOv8 的 COCO 数据集中，car 的类别 ID 是 2）
vehicle_class_id = 2

# 初始化计数器
vehicle_count = 0
tracked_vehicles = set()  # 用于存储已跟踪的车辆 ID

# 定义计数线的位置
count_line_y = 300  # 假设计数线在视频的 y=300 位置(此位置需要根据自己视频情况设置位置）

frame_queue = queue.Queue(maxsize = 6)

def xyxy_xcyc(x1, y1, x2, y2):
    x_c = x1+(x2-x1)/2
    y_c = y1+(y2-y1)/2
    w = x2-x1
    h = y2-y1
    return x_c, y_c, w, h

frame_count = 0

while cap.isOpened():
    ret , frame = cap.read()
    if not ret:
        break

    if not frame_queue.empty():  # 队列中有旧帧
        frame_queue.get_nowait() # 丢弃旧帧

    frame_queue.put(frame) # 放入最新帧
    frame = frame_queue.get()  # 获取最新帧
    frame_count += 1
    if frame_count % 3 == 0: # 每三帧处理一次
        # 使用 YOLOv8 进行目标检测
        results = model(frame, stream=True)

        # 提取检测结果
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)  # 类别 ID
                if class_id == vehicle_class_id:  # 只处理车辆
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # 检测框坐标
                    x_c, y_c, w, h = xyxy_xcyc(x1, y1, x2, y2)
                    confidence = float(box.conf)  # 置信度
                    detections.append(([x_c, y_c, w, h], confidence, class_id))

        boxes = np.array([det[0] for det in detections])  # 转换为 NumPy 数组
        confidences = np.array([det[1] for det in detections])  # 转换为 NumPy 数组

        # 使用 DeepSORT 进行目标跟踪
        tracks = tracker.update(boxes, confidences, frame)  # deepsort.update函数的第一个参数Boxes是x_c,y_c,w,h
        # 但是该函数会自己将x_c,y_c,w,h转成x1y1x2y2

        # 处理跟踪结果
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track[:5])  # 提取跟踪框坐标和 ID

            # 绘制矩形框和ID
            label = "ID-{:d}".format(track_id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]  # 返回标签的宽度和高度
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1-t_size[1]-2), (x1 + t_size[0] + 2, y1), (0, 255, 0), -1)  # 绘制背景颜色
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, [255, 0, 0], 1)

            # cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # 判断车辆是否穿过计数线
            if y1 < count_line_y < y2 and track_id not in tracked_vehicles:
                tracked_vehicles.add(track_id)
                vehicle_count += 1

        # 绘制计数线
        cv2.line(frame, (0, count_line_y), (frame.shape[1], count_line_y), (0, 0, 255), 2)

        # 显示车流量计数
        cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示视频帧
    cv2.imshow("Traffic Counting", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()