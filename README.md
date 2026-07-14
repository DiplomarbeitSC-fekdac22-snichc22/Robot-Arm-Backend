# Robot Arm Backend

FastAPI backend for the robot arm vision system.

## Features

- Raspberry Pi Camera Module 3 support
- Raw MJPEG camera stream
- YOLO detection stream
- JSON detection endpoint
- Cropped detected object images

## Endpoints

```text
GET /health
GET /video
GET /detect
GET /detections
GET /objects