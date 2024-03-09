from mistral_llm import load_llm_model


class LLMChain:

    def __init__(self):

        self.llm_model = load_llm_model()

    def generate_answer(self, question, context):
        """Generate answer based on the given question and context."""  # Ensure both context and question are provided
        if context is None or question is None:
            raise ValueError("Both context and question are required")
        response = self.llm_model.invoke(
            {"question": question, "context": context}
        )
        return response["text"]
