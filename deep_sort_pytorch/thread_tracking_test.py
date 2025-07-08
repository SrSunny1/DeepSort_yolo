import cv2
from deep_sort.deep_sort import DeepSort
from ultralytics import YOLO
import numpy as np
import queue
from threading import Thread, Event

# 加载 YOLOv8 模型
model = YOLO("yolov8n.pt")

# 加载权重文件路径
track_path = "ckpt.t7"
# 初始化 DeepSORT 跟踪器
tracker = DeepSort(max_age=50, model_path=track_path)

# 打开视频文件或摄像头
video_path = "videoa.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 定义车辆类别 ID
vehicle_class_id = 2

# 初始化计数器
vehicle_count = 0
tracked_vehicles = set()

# 定义计数线的位置
count_line_y = 300

frame_queue = queue.Queue(maxsize=2)
stop_event = Event()  # 用于停止捕获线程

def xyxy_xcyc(x1, y1, x2, y2):
    x_c = x1 + (x2 - x1) / 2
    y_c = y1 + (y2 - y1) / 2
    w = x2 - x1
    h = y2 - y1
    return x_c, y_c, w, h

def capture_thread():
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # 视频读取完毕或出错，退出循环
                break
            try:
                # 如果队列满，移除最旧的帧
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put(frame)
            except queue.Empty:
                pass
        # 视频处理完毕，添加 None 作为结束标志
        frame_queue.put(None)
    except Exception as e:
        print(f"捕获线程出错: {e}")


# 启动捕获线程
capture_thread = Thread(target=capture_thread, daemon=True)
capture_thread.start()

frame_count = 0

try:
    while True:
        try:
            # 从队列获取帧
            frame = frame_queue.get()
            # 检查是否是结束标志
            if frame is None:
                break
            frame_count += 1
            if frame_count % 2 == 0:
                # 使用 YOLOv8 进行目标检测
                try:
                    results = model(frame, stream=True)
                    # 提取检测结果
                    detections = []
                    for result in results:
                        for box in result.boxes:
                            class_id = int(box.cls)
                            if class_id == vehicle_class_id:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                x_c, y_c, w, h = xyxy_xcyc(x1, y1, x2, y2)
                                confidence = float(box.conf)
                                detections.append(([x_c, y_c, w, h], confidence, class_id))
                    print(f"检测到 {len(detections)} 个车辆目标")

                    # 确保有检测结果再进行跟踪
                    if detections:
                        boxes = np.array([det[0] for det in detections])
                        confidences = np.array([det[1] for det in detections])
                        # 使用 DeepSORT 进行目标跟踪
                        try:
                            tracks = tracker.update(boxes, confidences, frame)
                            print(f"DeepSORT 跟踪到 {len(tracks)} 个目标")
                            # 处理跟踪结果
                            for track in tracks:
                                x1, y1, x2, y2, track_id = map(int, track[:5])
                                # 绘制矩形框和ID
                                label = "ID-{:d}".format(track_id)
                                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.rectangle(frame, (x1, y1 - t_size[1] - 2), (x1 + t_size[0] + 2, y1), (0, 255, 0), -1)
                                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, [255, 0, 0], 1)

                                # 判断车辆是否穿过计数线
                                if y1 < count_line_y < y2 and track_id not in tracked_vehicles:
                                    tracked_vehicles.add(track_id)
                                    vehicle_count += 1
                        except Exception as e:
                            print(f"DeepSORT 跟踪出错: {e}")
                except Exception as e:
                    print(f"目标检测出错: {e}")

            # 绘制计数线和车辆计数
            cv2.line(frame, (0, count_line_y), (frame.shape[1], count_line_y), (0, 0, 255), 2)
            cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 显示视频帧
            cv2.imshow("Traffic Counting", frame)

            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except queue.Empty:
            pass
except Exception as e:
    print(f"主线程出错: {e}")
finally:
    # 清理资源
    try:
        stop_event.set()  # 通知捕获线程停止
        # 等待捕获线程结束
        if capture_thread.is_alive():
            capture_thread.join(timeout=1.0)
        # 释放视频资源
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"资源释放出错: {e}")