import os

from ctransformers import AutoTokenizer
from langchain_community.llms import CTransformers
from langchain.agents import AgentOutputParser, initialize_agent
from langchain.tools import Memory

import constants
import constants as c

MODEL_PATH = "path to downloaded .gguf file"


MODEL_PATH = constants.MODEL_PATH
# We use Langchain's CTransformers llm class to load our quantized model
llm = CTransformers(model=MODEL_PATH, config=c.LLM_CONFIG)

# Tokenizer for Mistral-7B-Instruct from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

#
# agent = initialize_agent(
#     agent="chat-conversational-react-description",
#     tools=None,
#     llm=llm,
#     verbose=True,
#     early_stopping_method="generate",
#     memory=memory,
#     agent_kwargs={"output_parser": parser},
# )


def generate_text(prompt):

    try:
        response = agent(prompt)
    except Exception as e:
        response = str(e)
        if response.startswith("Could not parse LLM output: `"):
            response = response.removeprefix(
                "Could not parse LLM output: `"
            ).removesuffix("`")
            print(response)


# Replace "Your prompt here" with an actual prompt


if __name__ == "__main__":
    # Call the function and print the joke
    prompt = "tellme a joke"
    print(generate_text(prompt))
