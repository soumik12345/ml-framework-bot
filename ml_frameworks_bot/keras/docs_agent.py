from typing import Dict, List, Optional, Union

import weave
from llama_index.core.schema import BaseNode
from pydantic import BaseModel
from rich.progress import track

from ..llm_wrapper import LLMClientWrapper
from ..schema import KerasOperations
from .heuristic_retriever import KerasDocumentationHeuristicRetreiver
from .neural_retriever import KerasDocumentationRetreiver


DocumentationRetreiver = Union[
    KerasDocumentationRetreiver, KerasDocumentationHeuristicRetreiver
]


class KerasOpWithAPIReference(BaseModel):
    keras_op: str
    api_reference: str
    api_reference_path: str


class KerasDocumentationAgent(weave.Model):
    op_extraction_llm_client: LLMClientWrapper
    retrieval_augmentation_llm_client: LLMClientWrapper
    api_reference_retriever: DocumentationRetreiver
    use_rich: bool

    def __init__(
        self,
        op_extraction_llm_client: LLMClientWrapper,
        retrieval_augmentation_llm_client: LLMClientWrapper,
        api_reference_retriever: DocumentationRetreiver,
        use_rich: bool = True,
    ):
        super().__init__(
            op_extraction_llm_client=op_extraction_llm_client,
            retrieval_augmentation_llm_client=retrieval_augmentation_llm_client,
            api_reference_retriever=api_reference_retriever,
            use_rich=use_rich,
        )

    @weave.op()
    def extract_keras_operations(
        self, code_snippet: str, seed: Optional[int] = None, max_retries: int = 3
    ) -> KerasOperations:
        keras_operations: KerasOperations = self.op_extraction_llm_client.predict(
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
    def ask_llm_about_op(self, keras_op: str) -> str:
        return self.retrieval_augmentation_llm_client.predict(
            messages=[
                {
                    "role": "user",
                    "content": f"Describe the purpose of `{keras_op}` in less than 100 words",  # noqa: E501
                }
            ],
        )

    @weave.op()
    def retrieve_api_references(
        self, keras_ops: KerasOperations
    ) -> List[KerasOpWithAPIReference]:
        ops_with_api_reference = []
        iterable = keras_ops.operations
        iterable = (
            track(iterable, description="Retrieving api references:")
            if self.use_rich
            else iterable
        )
        for keras_op in iterable:
            is_neural_retriever = isinstance(
                self.api_reference_retriever, KerasDocumentationRetreiver
            )
            if is_neural_retriever:
                purpose_of_op = self.ask_llm_about_op(keras_op)
                api_reference: BaseNode = self.api_reference_retriever.predict(
                    query=f"API reference for `{keras_op}`.\n{purpose_of_op}",
                )[0]
            else:
                api_reference: BaseNode = self.api_reference_retriever.predict(
                    query=keras_op
                )
            ops_with_api_reference.append(
                KerasOpWithAPIReference(
                    keras_op=keras_op,
                    api_reference=api_reference.text,
                    api_reference_path=(
                        api_reference.node.metadata["file_path"]
                        if is_neural_retriever
                        else api_reference.metadata["file_path"]
                    ),
                )
            )
        return ops_with_api_reference

    @weave.op()
    def predict(
        self, code_snippet: str, seed: Optional[int] = None, max_retries: int = 3
    ) -> Dict[str, List[KerasOpWithAPIReference]]:
        keras_ops = self.extract_keras_operations(
            code_snippet=code_snippet, seed=seed, max_retries=max_retries
        )
        return {
            "retrieved_keras_ops_with_references": self.retrieve_api_references(
                keras_ops=keras_ops
            )
        }
