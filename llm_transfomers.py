import os

from ctransformers import AutoTokenizer
from langchain_community.llms import CTransformers
from langchain.agents import AgentOutputParser, initialize_agent
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish


MODEL_PATH = 'path to downloaded .gguf file'

# Some basic configurations for the model
config = {
    "max_new_tokens": 2048,
    "context_length": 4096,
    "repetition_penalty": 1.1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 0.9,
    "stream": True,
    "threads": int(os.cpu_count() / 2)
}

MODEL_PATH = "llm_model/TheBloke/mistral-7b-instruct-v0.1.Q5_0/mistral-7b-instruct-v0.1.Q5_0.gguf"
# We use Langchain's CTransformers llm class to load our quantized model
llm = CTransformers(model=MODEL_PATH,
                    config=config)



# Tokenizer for Mistral-7B-Instruct from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=None,
    llm=llm,
    verbose=True,
    early_stopping_method="generate",
    memory=memory,
    agent_kwargs={"output_parser": parser}
)

def generate_text(prompt):

    try:
        response = agent(prompt)
    except Exception as e:
        response = str(e)
        if response.startswith("Could not parse LLM output: `"):
            response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
            print(response)


# Replace "Your prompt here" with an actual prompt


if __name__ == '__main__':
    # Call the function and print the joke
    prompt = "tellme a joke"
    print(generate_text(prompt))
