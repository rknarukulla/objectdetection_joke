# Joke generation with detected objects as context

This is an example of how to use the object detection and LLMs together.

This application uses objects detected by the object detection model as a context to generate a joke for the audience.

- It uses YOLO world model for objection and Mistral LLM for the joke generation.

- `app.py` is the main application to run.
- Please sure to download a quantized Mistral model and update it's path in the `constants.py` file.