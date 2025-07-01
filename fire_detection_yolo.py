import cv2
from ultralytics import YOLO

# Use the more accurate model: 'yolov8s.pt'
MODEL_PATH = "yolov8s.pt"
model = YOLO(MODEL_PATH)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO inference
    results = model(frame)

    # Draw bounding boxes for detected fire and smoke
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = result.names[int(box.cls[0])]
            if label.lower() == "fire":
                color = (0, 0, 255)  # Red for fire
            elif label.lower() == "smoke":
                color = (0, 255, 255)  # Yellow for smoke
            else:
                color = (255, 255, 255)  # White for unknown
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label.upper()} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Webcam - YOLOv8 Fire & Smoke Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 