import json
import os
from typing import Optional

import wandb
import weave
from llama_index.core.schema import BaseNode, TextNode


class KerasDocumentationHeuristicRetreiver(weave.Model):
    repository_local_path: Optional[str]
    repository: str = "https://github.com/keras-team/keras-io"
    _api_to_doc_mapping: dict[str, str] = {}

    def __init__(
        self,
        api_to_doc_mapping_file: str,
        repository_local_path: Optional[str] = None,
    ):
        super().__init__(
            repository_local_path=repository_local_path,
        )
        with open(api_to_doc_mapping_file, "r") as file:
            self._api_to_doc_mapping = json.load(file)
        if repository_local_path is None:
            api = wandb.Api()
            artifact = api.artifact("ml-colabs/ml-frameworks-bot/keras3-docs:latest")
            artifact_dir = artifact.download()
            self.repository_local_path = artifact_dir
        else:
            self.repository_local_path = repository_local_path

    def load_doc(self, keras_op: str) -> BaseNode:
        # TODO: Gracefully handle KeyError
        with open(
            os.path.join(
                self.repository_local_path, self._api_to_doc_mapping[keras_op]
            ),
            "r",
        ) as file:
            text = file.read()

        return TextNode(
            text=text, metadata={"file_path": self._api_to_doc_mapping[keras_op]}
        )

    @weave.op()
    def predict(self, query: str) -> BaseNode:
        return self.load_doc(keras_op=query)
