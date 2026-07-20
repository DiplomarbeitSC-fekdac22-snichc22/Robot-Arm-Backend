import uuid

from ultralytics import YOLO

from .camera import make_jpeg, read_camera, send_frame
from .config import MODEL_PATH, YOLO_CONF


print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("YOLO model loaded.")
print("Model classes:", model.names)


def run_detection(frame):
    return model.predict(
        source=frame,
        imgsz=320,
        conf=YOLO_CONF,
        iou=0.45,
        max_det=10,
        verbose=False,
    )[0]


def parse_detections(result):
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "id": str(uuid.uuid4()),
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 3),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            },
            "center": {
                "x": round((x1 + x2) / 2, 2),
                "y": round((y1 + y2) / 2, 2),
            },
        })

    return detections


def detection_stream():
    while True:
        frame = read_camera()

        result = run_detection(frame)
        frame_with_boxes = result.plot()

        jpeg = make_jpeg(frame_with_boxes)

        if jpeg is None:
            continue

        yield send_frame(jpeg)
