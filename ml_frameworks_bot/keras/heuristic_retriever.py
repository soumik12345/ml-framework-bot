import os
from typing import Optional

import weave
from llama_index.core.schema import BaseNode, TextNode

from ..utils import build_keras_io_sources, fetch_git_repository
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
        repository_owner = self.repository.split("/")[-2]
        repository_name = self.repository.split("/")[-1]
        personal_access_token = os.getenv("PERSONAL_ACCESS_TOKEN")
        fetch_git_repository(
            self.repository_local_path,
            repository_owner,
            repository_name,
            personal_access_token,
        )
        source_directory = os.path.join(self.repository_local_path, "sources")
        if not os.path.exists(source_directory):
            build_keras_io_sources(repository_local_path=self.repository_local_path)

        # FIXME: Gracefully handle KeyError
        with open(
            os.path.join(self.repository_local_path, APIToDocMapping[keras_op]), "r"
        ) as file:
            text = file.read()

        return TextNode(text=text, metadata={"file_path": APIToDocMapping[keras_op]})

    @weave.op()
    def predict(self, query: str) -> BaseNode:
        return self.load_doc(keras_op=query)
