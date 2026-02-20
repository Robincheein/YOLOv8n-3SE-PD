import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

from ultralytics import YOLO
import cv2
import os
import numpy as np
import time

# 禁用OpenCV的GUI功能（避免显示窗口）
cv2.setUseOptimized(True)

model = YOLO('yolov8n.pt')  # 模型路径
output_dir = 'video_output'
output_file_name = 'detected_video_road_yolov8.mp4'

video_path = 'dataset/road.mp4'  # 视频路径
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file!")
    exit()

# 输出设置
os.makedirs(output_dir, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(os.path.join(output_dir, output_file_name), fourcc, fps, (width, height))


# 改进的违规跟踪系统
class ViolationTracker:
    def __init__(self, max_age=300):  # 300帧（约10秒）内出现的同一人视为重复
        self.known_violations = {}  # track_id: (last_seen_time, bbox, count)
        self.unique_violations = set()  # 真正唯一的违规ID
        self.max_age = max_age
        self.current_frame = 0

    def update(self, boxes, frame):
        self.current_frame += 1
        current_violations = set()

        if boxes is not None:
            for box in boxes:
                if box.id is not None:
                    track_id = int(box.id[0])
                    cls = int(box.cls[0])

                    if cls == 1:  # without_helmet
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])

                        # 检查是否是已知的违规
                        is_new = True
                        for known_id, (last_seen, bbox, count) in list(self.known_violations.items()):
                            # 计算IOU（交并比）来判断是否是同一人
                            iou = self.calculate_iou((x1, y1, x2, y2), bbox)
                            if iou > 0.3:  # IOU阈值，可以调整
                                is_new = False
                                self.known_violations[known_id] = (self.current_frame, (x1, y1, x2, y2), count)
                                current_violations.add(known_id)
                                break

                        if is_new:
                            # 新违规，检查是否在最近出现过
                            matched = False
                            for known_id, (last_seen, bbox, count) in list(self.known_violations.items()):
                                if self.current_frame - last_seen < self.max_age:
                                    iou = self.calculate_iou((x1, y1, x2, y2), bbox)
                                    if iou > 0.2:  # 较低的IOU阈值用于重新识别
                                        matched = True
                                        self.known_violations[known_id] = (self.current_frame, (x1, y1, x2, y2),
                                                                           count + 1)
                                        current_violations.add(known_id)
                                        break

                            if not matched:
                                # 真正的新违规
                                new_id = track_id
                                self.known_violations[new_id] = (self.current_frame, (x1, y1, x2, y2), 1)
                                self.unique_violations.add(new_id)
                                current_violations.add(new_id)

        # 清理过期的记录
        for track_id in list(self.known_violations.keys()):
            last_seen, bbox, count = self.known_violations[track_id]
            if self.current_frame - last_seen > self.max_age:
                del self.known_violations[track_id]

        return current_violations

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IOU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算交集区域
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # 计算并集区域
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def get_unique_count(self):
        return len(self.unique_violations)


# 创建跟踪器
violation_tracker = ViolationTracker()


# 图像预处理函数
def preprocess_frame(frame):
    # 先调整到较大尺寸进行增强
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # 锐化处理
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)

    # 调整对比度和亮度
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

    # 最终调整回原始尺寸
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


print("Starting video processing...")
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # if frame_count % 30 == 0:  # 每30帧打印一次进度
    #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     progress = (frame_count / total_frames) * 100
    #     elapsed = time.time() - start_time
    #     fps = frame_count / elapsed
        # print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), FPS: {fps:.1f}")

    # 预处理帧
    processed_frame = preprocess_frame(frame)

    # 进行检测
    results = model.track(processed_frame, conf=0.4, iou=0.7, persist=True, tracker="botsort.yaml")

    current_violation_ids = set()

    for result in results:
        boxes = result.boxes
        current_violation_ids = violation_tracker.update(boxes, frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                # 获取类别名称
                if hasattr(model, 'names') and model.names is not None:
                    class_name = model.names[cls]
                else:
                    class_name = f"class_{cls}"

                label = f"{class_name} {conf:.2f} ID:{track_id}"

                if cls == 1:  # without_helmet
                    # 检查是否是当前帧的违规（避免显示历史违规）
                    if track_id in current_violation_ids:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Violation: {label}", (x1, max(y1 - 30, 10)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, max(y1 - 30, 10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 添加统计信息
    cv2.putText(frame, f"Violators: {violation_tracker.get_unique_count()}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Frames: {frame_count}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Current violations: {len(current_violation_ids)}", (10, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 写入输出视频
    out.write(frame)

    # 安全限制，避免无限循环
    if frame_count > 1000:  # 最多处理5000帧
        break

# 释放资源
cap.release()
out.release()

end_time = time.time()
total_time = end_time - start_time

print(f"Video processing completed!")
print(f"Total frames processed: {frame_count}, Time taken: {total_time:.1f} seconds")
print(f"Average FPS: {frame_count / total_time:.1f}")
print(f"Total unique violators detected: {violation_tracker.get_unique_count()}")

# 打印详细的违规统计
# print("\nViolation statistics:")
# for track_id, (last_seen, bbox, count) in violation_tracker.known_violations.items():
#     if track_id in violation_tracker.unique_violations:
#         print(f"ID {track_id}: Appeared {count} times")
