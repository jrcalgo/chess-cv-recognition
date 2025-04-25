from ultralytics import YOLO
import cv2
import collections

# Load your trained model
model = YOLO("C:/Users/amith/Documents/School/ITCS 4152/chess_model/best.pt")

# Open Elgato virtual camera (adjust index if needed)
cap = cv2.VideoCapture(2)

# History for smoothing predictions
history = collections.defaultdict(lambda: collections.deque(maxlen=5))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("❌ Failed to grab frame.")
        break

    # Run YOLOv11 detection
    results = model.predict(source=frame, conf=0.4, iou=0.5, save=False, stream=True)

    # Annotated frame copy
    annotated_frame = frame.copy()

    # Container for piece bounding box info
    piece_locations = []

    for r in results:
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            label = model.names[cls]

            x1, y1, x2, y2 = xyxy
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Smoothing based on position buckets
            center_id = (cx // 20, cy // 20)
            history[center_id].append(cls)
            stable_cls = max(set(history[center_id]), key=history[center_id].count)
            label = model.names[stable_cls]

            width = x2 - x1
            height = y2 - y1

            # Store only top-left corner + width/height
            piece_data = {
                'label': label,
                'x1': int(x1),
                'y1': int(y1),
                'width': int(width),
                'height': int(height)
            }
            piece_locations.append(piece_data)

            # Draw bounding box and label on frame
            cv2.rectangle(annotated_frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show live window
    cv2.imshow("Stabilized Chess Detection", annotated_frame)

    # Print piece locations for this frame
    print("🧠 Detected pieces this frame:")
    for piece in piece_locations:
        print(piece)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
