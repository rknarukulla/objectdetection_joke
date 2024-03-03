import requests
from flask import Flask, request, jsonify
from mistral_llm import load_llm_model
import logging


logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

llm_chain = load_llm_model()


@app.route("/generate", methods=["POST"])
def generate_text():
    """Generate text based on the given context and question."""
    # Parse input data

    data = request.get_json()
    context_p = data.get("context")
    question_p = data.get("question")

    # Ensure both context and question are provided
    if not context_p or not question_p:
        return (
            jsonify(
                {"error": "Both 'context' and 'question' fields are required."}
            ),
            400,
        )

    response = llm_chain.invoke({"question": question_p, "context": context_p})

    return jsonify({"response": response["text"]})


@app.route("/generate_auto", methods=["POST"])
def get_names():
    response = requests.get("http://localhost:5000/names")
    names_json = response.json()

    context = data.get("context")

    # Add names JSON to context
    context += "\n\n" + names_json


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5100, debug=True)

"""
Example input: 

{ 
"context": "Observations are given as json where the key is name and value is count of items or persons. The 
json is {person:3,laptop:1,chair:2}. Use this context as for surrounding environment", 

"question": "Tell a joke to the audience" 
}


"""
