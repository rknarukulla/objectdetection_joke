import requests
from flask import Flask, request, jsonify
from langchain.chains import LLMChain
from langchain_community.llms.ctransformers import CTransformers
from langchain_core.prompts import PromptTemplate
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Initialize the model and template outside of the request handling function
model_path = "llm_model/TheBloke/mistral-7b-instruct-v0.1.Q5_0/mistral-7b-instruct-v0.1.Q5_0.gguf"
config = {'max_new_tokens': 100, 'temperature': 0.9}

llm = CTransformers(model=model_path, model_file="mistral-7b-instruct-v0.1.Q5_0.gguf", config=config)

template = """<s>[INST] Imagine you are in a room where you've noticed several items and people around you, 
These observations will be provided as context. Based on this observation, craft a message that answers the question 
making sure to incorporate the presence of these items and individuals creatively. Ensure your response is both 
creative and formal, and do not refer to any process of detection or mention the technical aspect of how you came to 
know about these items.

Context: {context}
Question: {question}

[/INST] </s>"""

# Pre-define the prompt template to avoid recreating it on every request
prompt_template = PromptTemplate(template=template, input_variables=["question", "context"])
llm_chain = LLMChain(prompt=prompt_template, llm=llm)
logging.info("LLM model loaded, Warming up...")
llm_chain.invoke({"question": "say Hi", "context": "check"})
logging.info("LLM warmup complete")

@app.route('/generate', methods=['POST'])
def generate_text():
    """Generate text based on the given context and question.
    """
    # Parse input data

    data = request.get_json()
    context_p = data.get('context')
    question_p = data.get('question')

    # Ensure both context and question are provided
    if not context_p or not question_p:
        return jsonify({"error": "Both 'context' and 'question' fields are required."}), 400

    response = llm_chain.invoke({"question": question_p, "context": context_p})

    return jsonify({"response": response['text']})


@app.route('/generate_auto', methods=['POST'])
def get_names():

    response = requests.get("http://localhost:5000/names")
    names_json = response.json()

    context = data.get('context')

    # Add names JSON to context
    context += "\n\n" + names_json

    

@app.route('/health')
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)


"""
Example input: 

{ 
"context": "Observations are given as json where the key is name and value is count of items or persons. The 
json is {person:3,laptop:1,chair:2}. Use this context as for surrounding environment", 

"question": "Tell a joke to the audience" 
}


"""