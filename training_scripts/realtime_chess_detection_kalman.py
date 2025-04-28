from ultralytics import YOLO
import cv2
import numpy as np
import collections

# Load your trained model
model = YOLO("C:/Users/amith/Documents/School/ITCS 4152/chess_model/best.pt")

# Open Elgato virtual camera (adjust index if needed)
cap = cv2.VideoCapture(1)

# Define a color for each piece type (BGR format)
piece_colors = {
    # BLACK pieces (cooler colors)
    'black-bishop': (128, 0, 128),    # Deep Purple
    'black-king': (0, 0, 180),         # Darker Blue
    'black-knight': (0, 128, 255),     # Sky Blue
    'black-pawn': (0, 255, 255),       # Bright Cyan
    'black-queen': (0, 0, 128),        # Navy Blue
    'black-rook': (0, 255, 128),       # Aquamarine

    # WHITE pieces (warmer / brighter colors)
    'white-bishop': (0, 255, 0),       # Pure Green
    'white-king': (255, 255, 255),     # White
    'white-knight': (0, 165, 255),     # Orange
    'white-pawn': (0, 255, 255),       # Bright Cyan (shared intentionally for visibility)
    'white-queen': (255, 0, 0),        # Bright Red
    'white-rook': (0, 215, 255)        # Gold
}

# History memory for classification smoothing
history = collections.defaultdict(lambda: collections.deque(maxlen=10))

# Kalman filters for bounding box smoothing
kalman_filters = {}

# Initialize a new Kalman filter
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

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("❌ Failed to grab frame.")
        break

    results = model.predict(source=frame, conf=0.7, iou=0.5, save=False, stream=True)
    annotated_frame = frame.copy()
    piece_locations = []

    for r in results:
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            label = model.names[cls]

            x1, y1, x2, y2 = xyxy
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Group detections by rough center location
            center_id = (cx // 20, cy // 20)
            history[center_id].append(cls)

            counts = collections.Counter(history[center_id])
            most_common_cls, count = counts.most_common(1)[0]

            # Require at least 70% agreement before changing label
            if count / len(history[center_id]) >= 0.7:
                stable_cls = most_common_cls
            else:
                stable_cls = history[center_id][-1]

            label = model.names[stable_cls]

            width = x2 - x1
            height = y2 - y1

            # Initialize Kalman filter for this center if not exists
            if center_id not in kalman_filters:
                kalman_filters[center_id] = create_kalman_filter()
                kalman_filters[center_id].statePre = np.array([[cx], [cy], [0], [0]], np.float32)

            # Kalman predict
            kf = kalman_filters[center_id]
            predicted = kf.predict()

            # Measurement update (correct with real center)
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
            corrected = kf.correct(measurement)

            # Use smoothed center
            cx_smoothed, cy_smoothed = int(corrected[0][0]), int(corrected[1][0])

            # Adjust x1, y1 based on new center
            x1_smoothed = cx_smoothed - width // 2
            y1_smoothed = cy_smoothed - height // 2
            x2_smoothed = cx_smoothed + width // 2
            y2_smoothed = cy_smoothed + height // 2

            # Store the piece location (optional for further processing)
            piece_data = {
                'label': label,
                'x1': int(x1_smoothed),
                'y1': int(y1_smoothed),
                'width': int(width),
                'height': int(height)
            }
            piece_locations.append(piece_data)

            # Choose color based on label
            color = piece_colors.get(label, (0, 255, 0))  # default green if not found

            # Draw box and label
            cv2.rectangle(annotated_frame, (x1_smoothed, y1_smoothed), (x2_smoothed, y2_smoothed), color, 2)
            cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1_smoothed, y1_smoothed - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show window
    cv2.imshow("Stabilized Chess Detection", annotated_frame)

    # Print detected pieces
    print("🧠 Detected pieces this frame:")
    for piece in piece_locations:
        print(piece)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
