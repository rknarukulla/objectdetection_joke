import json

import cv2
from flask import Flask, Response, jsonify
from ultralytics import YOLO

# Initialize Flask app and YOLO model
app = Flask(__name__)
model = YOLO('yolov8n.pt')  # Load YOLOv8 model


def get_frame():
    global latest_detection_data
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)
        latest_detection_data = json.loads(results[0].tojson())
        # Render detections on the frame
        frame_with_detections = results[0].plot()

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame_with_detections)
        frame_bytes = buffer.tobytes()

        # # Yield the frame

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/video_feed')
def video_feed():
    # Stream the video with detections
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/data_feed')
def data_feed():
    global latest_detection_data
    return jsonify(latest_detection_data)


@app.route('/names')
def objects_info():
    global latest_detection_data
    names = {}
    for obj in latest_detection_data:
        name = obj["name"]
        if names.get(name) is None:
            names[name] = 1
        else:
            names[name] += 1

    return jsonify(names)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
