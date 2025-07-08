import cv2
from deep_sort.deep_sort import DeepSort
from ultralytics import YOLO
import numpy as np

# 加载 YOLOv8 模型
model = YOLO("yolov8n.pt")  # 使用预训练的 YOLOv8 模型

# 加载权重文件路径
track_path = "ckpt.t7"
# 初始化 DeepSORT 跟踪器
tracker = DeepSort(max_age = 50, model_path=track_path)  # max_age 是跟踪目标的最大存活帧数

# 打开视频文件或摄像头
video_path = "videoa.mp4"
cap = cv2.VideoCapture(video_path)

# 定义类别 ID（YOLOv8 的 COCO 数据集中,不同事物对应id，如果是自己训练的模型，就要对应自己yaml文件中的id）
vehicle_class_id = 2
truck_id = 7
person_id = 0

# 初始化计数器
left_count = 0
right_count = 0
left_vehicles = set()  # 用于存储已跟踪的车辆 ID
right_vehicles = set()

# 定义计数线的位置
left_y = 170
right_y = 300

def xyxy_xcyc(x1, y1, x2, y2):
    x_c = x1+(x2-x1)/2
    y_c = y1+(y2-y1)/2
    w = x2-x1
    h = y2-y1
    return x_c, y_c, w, h


while cap.isOpened():
    ret, frame = cap.read()  # frame是当前帧的图像数据，为Numpy数组(heights,weights,channels)
    if not ret:
        break

    # 使用 YOLOv8 进行目标检测
    results = model(frame, stream=True)

    # 提取检测结果
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)  # 类别 ID
            if class_id == vehicle_class_id or class_id == truck_id:  # 只处理车辆
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
        x_centre = frame.shape[1]//2-90
        # 判断车辆是否穿过计数线
        if y1 < left_y < y2 and x2 < x_centre and track_id not in left_vehicles:
            left_vehicles.add(track_id)
            left_count += 1
        if y1 < right_y < y2 and x_centre < x1 and track_id not in right_vehicles:
            right_vehicles.add(track_id)
            right_count += 1

    x_left = int(frame.shape[1]/2-90)
    x_right = int(frame.shape[1]/2-55)

    # 绘制计数线
    cv2.line(frame, (0, left_y), (x_left, left_y), (0, 0, 255), 2)  # 左线
    cv2.line(frame, (x_right, right_y), (frame.shape[1], right_y), (0, 0, 255), 2)  # 右线
    # cv2.line(frame, (frame.shape[1]//2-90, 0), (frame.shape[1]//2-90, frame.shape[0]), (0, 255, 0), 2)

    # 显示车流量计数
    cv2.putText(frame, f"Left Count: {left_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Right Count: {right_count}", (x_right, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # 显示视频帧
    cv2.imshow("Traffic Counting", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()