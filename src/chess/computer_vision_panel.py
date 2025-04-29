import collections
import threading
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

def create_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kf


# Define a color for each piece type (BGR format)
piece_colors = {
    # BLACK pieces (cooler colors)
    'black-bishop': (128, 0, 128),  # Deep Purple
    'black-king': (0, 0, 180),  # Darker Blue
    'black-knight': (0, 128, 255),  # Sky Blue
    'black-pawn': (0, 255, 255),  # Bright Cyan
    'black-queen': (0, 0, 128),  # Navy Blue
    'black-rook': (0, 255, 128),  # Aquamarine

    # WHITE pieces (warmer / brighter colors)
    'white-bishop': (0, 255, 0),  # Pure Green
    'white-king': (255, 255, 255),  # White
    'white-knight': (0, 165, 255),  # Orange
    'white-pawn': (0, 255, 255),  # Bright Cyan
    'white-queen': (255, 0, 0),  # Bright Red
    'white-rook': (0, 215, 255)  # Gold
}


class ComputerVisionPanel:
    def __init__(self, display_size: tuple[int, int], model_path: str, video_capture_device: int,
                 capture_orientation: str):
        assert capture_orientation in ('landscape', 'portrait'), \
            "orientation must be `landscape` or `portrait`"
        self.orientation = 1 if capture_orientation.__eq__('portrait') else 0

        self.model = YOLO(model_path, 'detect')
        self.cur_prediction = None
        self.cap = cv2.VideoCapture(video_capture_device)
        self.running = False
        self.motion_thresh = 10.0

        self.display_size = display_size

        self.lock = threading.Lock()
        self.latest_annotation = [deque(maxlen=1) for _ in range(3)]  # 0: frame, 1: rect, 2: text

        # launch threads
        threading.Thread(target=self._inference_loop, daemon=True).start()

    def pull_for_render(self):
        """
        Always returns a 2-tuple; components may be None. Handle accordingly
        """
        if len(self.latest_annotation[0]) > 0:
            with self.lock:
                return self.latest_annotation[0].popleft(), self.latest_annotation[1].popleft()

    def quit(self):
        self.running = False

    def _maybe_to_portrait(self, frame: np.ndarray):
        if self.orientation == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.display_size is not None:
            width, height = self.display_size
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        return frame

    def _inference_loop(self):
        history = collections.defaultdict(lambda: collections.deque(maxlen=10))
        kalman_filters = {}
        piece_locations = []

        self.running = True

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success or frame is None or frame.size == 0:
                print("Empty frame – retrying in 100 ms…")
                time.sleep(0.1)
                continue

            frame = self._maybe_to_portrait(frame)

            results = self.model.predict(source=frame, conf=0.7, iou=0.5, save=False, stream=True)
            annotated_frame = frame.copy()

            for r in results:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())

                    x1, y1, x2, y2 = xyxy
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    center_id = (cx // 20, cy // 20)
                    history[center_id].append(cls)

                    counts = collections.Counter(history[center_id])
                    most_common_cls, count = counts.most_common(1)[0]

                    if count / len(history[center_id]) >= 0.7:
                        stable_cls = most_common_cls
                    else:
                        stable_cls = history[center_id][-1]

                    label = self.model.names[stable_cls]

                    width = x2 - x1
                    height = y2 - y1

                    if center_id not in kalman_filters:
                        kalman_filters[center_id] = create_kalman_filter()
                        kalman_filters[center_id].statePre = np.array([[cx], [cy], [0], [0]], np.float32)

                    kf = kalman_filters[center_id]
                    prediction = kf.predict()

                    measurement = np.array([[np.float32(cx)], [np.float32(cy)]], np.float32)
                    corrected = kf.correct(measurement)

                    cx_smoothed, cy_smoothed = int(corrected[0][0]), int(corrected[1][0])

                    x1_smoothed = cx_smoothed - width // 2
                    y1_smoothed = cy_smoothed - height // 2
                    x2_smoothed = cx_smoothed - width // 2
                    y2_smoothed = cy_smoothed - height // 2

                    piece_data = {
                        'label': label,
                        'x1': int(x1_smoothed),
                        'y1': int(y1_smoothed),
                        'width': int(width),
                        'height': int(height)
                    }
                    piece_locations.append(piece_data)
                    self.cur_prediction = piece_locations

                    color = piece_colors.get(label, (0, 255, 0))

                    with self.lock:
                        self.latest_annotation[0].append(frame)
                        self.latest_annotation[1].append(cv2.rectangle(annotated_frame, (x1_smoothed, y1_smoothed), (x2_smoothed, y2_smoothed), color, 2))
                        self.latest_annotation[2].append(cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1_smoothed, y1_smoothed - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2))

            if not self.running:
                break

        self.cap.release()
        cv2.destroyAllWindows()
