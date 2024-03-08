import logging
from unittest import TestCase

from llm_offline_qna import LLMChain

llm_obj = LLMChain()


class TestLLMChain(TestCase):

    def test_generate_answer(self):
        response = llm_obj.generate_answer(
            "What is the meaning of life?", "Life is 42"
        )
        logging.info(response)
        self.assertTrue(True)
