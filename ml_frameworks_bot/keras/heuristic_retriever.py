import os
from typing import Optional

import weave
from llama_index.core.schema import BaseNode, TextNode

from .mapping import APIToDocMapping


class KerasDocumentationHeuristicRetreiver(weave.Model):
    repository_local_path: Optional[str]
    repository: str = "https://github.com/keras-team/keras-io"

    def __init__(
        self,
        repository_local_path: Optional[str] = None,
    ):
        super().__init__(
            repository_local_path=repository_local_path,
        )
        self.repository_local_path = repository_local_path

    def load_doc(self, keras_op: str) -> BaseNode:
        # TODO: Gracefully handle KeyError
        with open(
            os.path.join(self.repository_local_path, APIToDocMapping[keras_op]), "r"
        ) as file:
            text = file.read()

        return TextNode(text=text, metadata={"file_path": APIToDocMapping[keras_op]})

    @weave.op()
    def predict(self, query: str) -> BaseNode:
        return self.load_doc(keras_op=query)
