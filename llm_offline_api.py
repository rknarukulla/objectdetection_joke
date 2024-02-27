from flask import Flask, request, jsonify
from langchain.chains import LLMChain
from langchain_community.llms.ctransformers import CTransformers
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)

# Assuming your model is loaded here (adjust according to your actual model loading code)
model_path = "llm_model/TheBloke/mistral-7b-instruct-v0.1.Q5_0/mistral-7b-instruct-v0.1.Q5_0.gguf"
config = {'max_new_tokens': 100, 'temperature': 0}
llm = CTransformers(model=model_path, model_file="mistral-7b-instruct-v0.1.Q5_0.gguf", config=config)

template = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in less than 20 words from the context
Answer the question below from context below :
{context}
{question} [/INST] </s>
"""


@app.route('/generate', methods=['POST'])
def generate_text():
    # Parse input data
    data = request.json
    context_p = data.get('context')
    question_p = data.get('question')

    # Ensure both context and question are provided
    if not context_p or not question_p:
        return jsonify({"error": "Both 'context' and 'question' fields are required."}), 400

    prompt = PromptTemplate(template=template, input_variables=["question", "context"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    response = llm_chain.invoke({"question": question_p, "context": context_p})

    return jsonify({"response": response['text']})


if __name__ == '__main__':
    app.run(debug=True)
