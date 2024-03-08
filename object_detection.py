import json

import cv2
from flask import Response, jsonify
from ultralytics import YOLOWorld

import constants as c
from utils import format_names


class ObjectDetection:

    def __init__(self):
        self.latest_detection_data = []
        self.model = YOLOWorld(c.MODEL_OBJECT_DETECTION)  # Load YOLOv8 model
        # Define custom classes
        # model.set_classes(["person with phone", "colors"])
        # print(model.names)

    def get_frame(self):
        # global latest_detection_data
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection
            results = self.model(frame)
            self.latest_detection_data = json.loads(results[0].tojson())
            # Render detections on the frame
            frame_with_detections = results[0].plot()

            # Encode the frame to JPEG
            ret, buffer = cv2.imencode(".jpg", frame_with_detections)
            frame_bytes = buffer.tobytes()

            # # Yield the frame

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

        cap.release()

        #

    def video_feed(self):

        # Stream the video with detections
        return Response(
            self.get_frame(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def data_feed(self):

        return jsonify(self.latest_detection_data)

    def objects_info(self):
        names = format_names(self.latest_detection_data)

        return jsonify(names)
