import logging

from flask import Flask, render_template, jsonify

from llm_offline_qna import LLMChain
from object_detection import ObjectDetection

object_detection_app = Flask(__name__)

obj = ObjectDetection()
llm_obj = LLMChain()


@object_detection_app.route("/video_feed")
def video_feed_app():
    return obj.video_feed()


@object_detection_app.route("/data_feed")
def data_feed_app():
    return obj.data_feed()


@object_detection_app.route("/names")
def names_app():
    return obj.objects_info()


@object_detection_app.route("/")
def dashboard():
    # Render and serve the dashboard.html template
    return render_template("dashboard.html")


# TODO: fix the API error
@object_detection_app.route("/joke")
def get_llm_response():

    names_json = obj.objects_info()
    logging.info(names_json)
    context = (
        f"Observations are given as json where the key is name and value is count of items or persons. "
        f"The json is {names_json}. Use this context as for surrounding environment",
    )

    question = "Tell a joke to the audience"

    out = llm_obj.generate_answer(question, context)
    return jsonify(out)


if __name__ == "__main__":
    object_detection_app.run(host="0.0.0.0", port=8000, debug=True)
