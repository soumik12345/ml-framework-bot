from typing import Dict, List, Optional, Union

import weave
from llama_index.core.schema import BaseNode
from pydantic import BaseModel
from rich.progress import track

from .llm_wrapper import LLMClientWrapper
from .retrieval import HeuristicRetreiver, NeuralRetreiver
from .schema import Operations


DocumentationRetreiver = Union[NeuralRetreiver, HeuristicRetreiver]


class OpWithAPIReference(BaseModel):
    op: str
    api_reference: str
    api_reference_path: str


class TranslationAgent(weave.Model):
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
    def extract_operations(
        self, code_snippet: str, seed: Optional[int] = None, max_retries: int = 3
    ) -> Operations:
        operations: Operations = self.op_extraction_llm_client.predict(
            max_retries=max_retries,
            response_model=Operations,
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
        unique_ops = Operations(operations=list(set(operations.operations)))
        return unique_ops

    @weave.op()
    def ask_llm_about_op(self, op: str) -> str:
        return self.retrieval_augmentation_llm_client.predict(
            messages=[
                {
                    "role": "user",
                    "content": f"Describe the purpose of `{op}` in less than 100 words",  # noqa: E501
                }
            ],
        )

    @weave.op()
    def retrieve_api_references(self, ops: Operations) -> List[OpWithAPIReference]:
        ops_with_api_reference = []
        iterable = ops.operations
        iterable = (
            track(iterable, description="Retrieving api references:")
            if self.use_rich
            else iterable
        )
        for op in iterable:
            is_neural_retriever = isinstance(
                self.api_reference_retriever, NeuralRetreiver
            )
            if is_neural_retriever:
                purpose_of_op = self.ask_llm_about_op(op)
                api_reference: BaseNode = self.api_reference_retriever.predict(
                    query=f"API reference for `{op}`.\n{purpose_of_op}",
                )[0]
            else:
                api_reference: BaseNode = self.api_reference_retriever.predict(query=op)
            ops_with_api_reference.append(
                OpWithAPIReference(
                    op=op,
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
    ) -> Dict[str, List[OpWithAPIReference]]:
        ops = self.extract_operations(
            code_snippet=code_snippet, seed=seed, max_retries=max_retries
        )
        return {"retrieved_ops_with_references": self.retrieve_api_references(ops=ops)}
