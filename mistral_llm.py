#pip install -U transformers accelerate ctransformers langchain torch pydantic

from langchain.chains import LLMChain
from langchain_community.llms.ctransformers import CTransformers
from langchain_core.prompts import PromptTemplate

# Load your model (this step is highly dependent on your model's format and quantization)
model_path = "llm_model/TheBloke/mistral-7b-instruct-v0.1.Q5_0/mistral-7b-instruct-v0.1.Q5_0.gguf"

config = {'max_new_tokens': 100, 'temperature': 0}
llm = CTransformers(model=model_path,model_file="mistral-7b-instruct-v0.1.Q5_0.gguf", config=config)

template = """<s>[INST] You are a helpful, respectful and honest assistant. Answer exactly in less than 20 words from the context
Answer the question below from context below :
{context}
{question} [/INST] </s>
"""

#### Prompt
question_p = """Tell me a joke"""
context_p = """you are looking at an image that has 3 people standing, 2 chairs and one laptom. Use this context as for surrounding environment"""

prompt = PromptTemplate(template=template, input_variables=["question","context"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
response = llm_chain.invoke({"question":question_p,"context":context_p})

print(response['text'])