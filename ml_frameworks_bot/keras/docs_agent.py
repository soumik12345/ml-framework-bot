from typing import List, Optional

import instructor
import weave
from instructor import Instructor
from litellm import completion
from packaging.version import Version
from pydantic import BaseModel
from rich.progress import track

from ..schema import KerasOperations
from ..utils import weave_op_wrapper
from .retriever import KerasDocumentationRetriever


class KerasOpWithAPIReference(BaseModel):
    keras_op: str
    api_reference: str
    api_reference_path: str


def postprocess_keras_op_name(keras_op_name: str, keras_version: str) -> str:
    keras_op_name = (
        "keras." + keras_op_name
        if not keras_op_name.startswith("keras.")
        else keras_op_name
    )
    keras_op_name = (
        "tensorflow." + keras_op_name
        if Version(keras_version) < Version("3.0.0")
        and not keras_version.startswith("tensorflow")
        else keras_op_name
    )
    return keras_op_name


class KerasDocumentationAgent(weave.Model):
    llm_name: str
    api_reference_retriever: KerasDocumentationRetriever
    keras_api_version: str
    _llm_client: Instructor

    def __init__(
        self,
        llm_name: str,
        api_reference_retriever: KerasDocumentationRetriever,
        keras_api_version: str = "3.5.0",
    ):
        super().__init__(
            llm_name=llm_name,
            api_reference_retriever=api_reference_retriever,
            keras_api_version=keras_api_version,
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
You are supposed to think step-by-step about all the unique Keras operations,
layers, and functions from a given snippet of code.

Here are some rules:
1. All functions and classes that are imported from `keras` should be considered to
    be Keras operations.
2. `import` statements don't count as separate statements.
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
    def retrieve_api_references(
        self, keras_ops: KerasOperations
    ) -> List[KerasOpWithAPIReference]:
        ops_with_api_reference = []
        for keras_op in track(
            keras_ops.operations, description="Retrieving api references:"
        ):
            query_op_name = keras_op.replace("keras.", "").replace("tensorflow.", "")
            api_reference = self.api_reference_retriever.predict(
                query=f"API reference for `{query_op_name}`"
            )[0]
            ops_with_api_reference.append(
                KerasOpWithAPIReference(
                    keras_op=postprocess_keras_op_name(
                        keras_op_name=keras_op, keras_version=self.keras_api_version
                    ),
                    api_reference=api_reference.text,
                    api_reference_path=api_reference.node.metadata["file_path"],
                )
            )
        return ops_with_api_reference

    @weave.op()
    def predict(
        self, code_snippet: str, seed: Optional[int] = None, max_retries: int = 3
    ) -> List[KerasOpWithAPIReference]:
        keras_ops = self.extract_keras_operations(
            code_snippet=code_snippet, seed=seed, max_retries=max_retries
        )
        return self.retrieve_api_references(keras_ops=keras_ops)
