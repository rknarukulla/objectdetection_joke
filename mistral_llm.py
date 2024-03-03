from langchain.chains import LLMChain
from langchain_community.llms.ctransformers import CTransformers
from langchain_core.prompts import PromptTemplate
import logging

import constants as c


def load_llm_model(
    model_path: str = c.MODEL_PATH,
    model_file: str = c.MODEL_FILE,
    config=c.LLM_CONFIG,
    template=c.TEMPLATE,
    input_variables=c.INPUT_VARIABLES,
):

    llm = CTransformers(model=model_path, model_file=model_file, config=config)
    # Pre-define the prompt template to avoid recreating it on every request
    prompt_template = PromptTemplate(
        template=template, input_variables=input_variables
    )
    llm_model_chain = LLMChain(prompt=prompt_template, llm=llm)
    logging.info("LLM model loaded, Warming up...")
    llm_model_chain.invoke({"question": "say Hi", "context": "check"})
    logging.info("LLM warmup complete")

    return llm_model_chain
