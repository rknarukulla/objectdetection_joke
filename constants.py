# Initialize the model and template outside of the request handling function
import os

MODEL_PATH = "llm_model/TheBloke/mistral-7b-instruct-v0.1.Q5_0/mistral-7b-instruct-v0.1.Q5_0.gguf"
MODEL_FILE = "mistral-7b-instruct-v0.1.Q5_0.gguf"

LLM_CONFIG_SIMPLE = {"max_new_tokens": 100, "temperature": 0.9}

# Some basic configurations for the model
# "context_length": 4096,
LLM_CONFIG = {
    "max_new_tokens": 100,
    "repetition_penalty": 1.1,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 0.9,
    "stream": True,
    "threads": int(os.cpu_count() / 2),
}

TEMPLATE = """<s>[INST] Imagine you are in a room where you've noticed several items and people around you, 
These observations will be provided as context. Based on this observation, craft a message that answers the question 
making sure to incorporate the presence of these items and individuals creatively. Ensure your response is both 
creative and formal, and do not refer to any process of detection or mention the technical aspect of how you came to 
know about these items.

Context: {context}
Question: {question}

[/INST] </s>"""

INPUT_VARIABLES = ["question", "context"]

MODEL_OBJECT_DETECTION = "yolov8s-world.pt"
