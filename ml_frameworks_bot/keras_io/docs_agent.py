from typing import Optional

import instructor
import weave
from instructor import Instructor
from litellm import completion

from ..schema import KerasOperations
from ..utils import weave_op_wrapper
from .retriever import KerasIORetreiver


class KerasDocumentationAgent(weave.Model):
    llm_name: str
    template_retriever: KerasIORetreiver
    guides_retriever: KerasIORetreiver
    example_retriever: KerasIORetreiver
    _llm_client: Instructor

    def __init__(
        self,
        llm_name: str,
        template_retriever: KerasIORetreiver,
        guides_retriever: KerasIORetreiver,
        example_retriever: KerasIORetreiver,
    ):
        super().__init__(
            llm_name=llm_name,
            template_retriever=template_retriever,
            guides_retriever=guides_retriever,
            example_retriever=example_retriever,
        )
        self._llm_client = instructor.from_litellm(completion)

    @weave.op()
    def extract_keras_operations(
        self, code_snippet: str, seed: Optional[int] = None, max_retries: int = 3
    ) -> KerasOperations:
        keras_operations: KerasOperations = weave_op_wrapper(
            name="Instructor.chat.completions.create"
        )(self._llm_client.chat.completions.create)(
            model=self.llm_name,
            max_retries=max_retries,
            response_model=KerasOperations,
            seed=seed,
            messages=[
                {
                    "role": "system",
                    "content": """
You are an experienced machine learning engineer expert in python and Keras.
You are suppossed to think step-by-step about all the unique Keras operations,
layers, and functions from a given snippet of code.

Here are some rules:
1. All functions and classes that are imported from `keras` should be considered to
    be Keras operations.
2. `impport` statements don't count as separate statements.
3. If there are nested Keras operations, you should extract all the operations that
    are present inside the parent operation.
4. You should simply return the names of the ops and not the entire statement itself.
""",
                },
                {
                    "role": "user",
                    "content": code_snippet,
                },
            ],
        )
        unique_keras_ops = KerasOperations(
            snippets=list(set(keras_operations.operations))
        )
        return unique_keras_ops

    @weave.op()
    def predict(
        self, code_snippet: str, seed: Optional[int] = None, max_retries: int = 3
    ) -> KerasOperations:
        return self.extract_code_snippets(
            code_snippet=code_snippet, seed=seed, max_retries=max_retries
        )
