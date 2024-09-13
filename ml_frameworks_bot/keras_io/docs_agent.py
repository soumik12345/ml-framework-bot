from typing import List, Optional

import instructor
import weave
from instructor import Instructor
from litellm import completion
from llama_index.core.schema import BaseNode
from pydantic import BaseModel
from rich.progress import track

from ..schema import KerasOperations
from ..utils import weave_op_wrapper
from .retriever import KerasIORetreiver


class KerasOpWithAPIReference(BaseModel):
    keras_op: str
    api_reference: str


class KerasDocumentationAgent(weave.Model):
    llm_name: str
    api_reference_retriever: KerasIORetreiver
    _llm_client: Instructor

    def __init__(self, llm_name: str, api_reference_retriever: KerasIORetreiver):
        super().__init__(
            llm_name=llm_name, api_reference_retriever=api_reference_retriever
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
5. Ensure that the names of the ops consist of the entire `keras` namespace.
""",
                },
                {
                    "role": "user",
                    "content": code_snippet,
                },
            ],
        )
        unique_keras_ops = KerasOperations(
            operations=list(set(keras_operations.operations))
        )
        return unique_keras_ops

    @weave.op()
    def retrieve_api_references(self, keras_ops: KerasOperations) -> List[BaseNode]:
        ops_with_api_reference = []
        for keras_op in track(
            keras_ops.operations, description="Retrieving api references:"
        ):
            api_reference: BaseNode = self.api_reference_retriever.predict(
                query=f"API reference for `{keras_op}`"
            )[0]
            ops_with_api_reference.append(
                KerasOpWithAPIReference(
                    keras_op=keras_op, api_reference=api_reference.text
                )
            )
        return ops_with_api_reference

    @weave.op()
    def predict(
        self, code_snippet: str, seed: Optional[int] = None, max_retries: int = 3
    ) -> List[BaseNode]:
        keras_ops = self.extract_keras_operations(
            code_snippet=code_snippet, seed=seed, max_retries=max_retries
        )
        return self.retrieve_api_references(keras_ops=keras_ops)
