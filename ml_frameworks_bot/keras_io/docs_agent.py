import instructor
import weave
from instructor import Instructor
from litellm import completion

from ..schema import CodeSnippets
from .retriever import KerasIORetreiver


class KerasDocumentationAgent(weave.Model):
    llm_name: str
    retriever: KerasIORetreiver
    _llm_client: Instructor

    def __init__(self, llm_name: str, retriever: KerasIORetreiver):
        super().__init__(llm_name=llm_name, retriever=retriever)
        self._llm_client = instructor.from_litellm(completion)

    @weave.op()
    def extract_code_snippets(
        self, code_snippet: str, max_retries: int = 3
    ) -> CodeSnippets:
        return weave.op()(self._llm_client.chat.completions.create)(
            model=self.llm_name,
            max_retries=max_retries,
            response_model=CodeSnippets,
            messages=[
                {
                    "role": "system",
                    "content": """
You are an experienced machine learning engineer expert in python and Keras.
You are suppossed to extract all the Keras operations from a given snippet of code.
                    """,
                },
                {
                    "role": "user",
                    "content": code_snippet,
                },
            ],
        )

    @weave.op()
    def predict(self, code_snippet: str, max_retries: int = 3) -> CodeSnippets:
        return self.extract_code_snippets(code_snippet, max_retries)
