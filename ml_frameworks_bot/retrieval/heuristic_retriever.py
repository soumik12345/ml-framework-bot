import json
import os
from typing import Optional, get_args

import weave
from llama_index.core.schema import BaseNode, TextNode

from ..utils import SUPPORTED_FRAMEWORKS, get_wandb_artifact
from .common import RepositoryMapping


class HeuristicRetreiver(weave.Model):
    framework: str
    repository_local_path: Optional[str]

    def __init__(
        self,
        framework: str,
        repository_local_path: Optional[str] = None,
    ):
        assert framework in get_args(
            SUPPORTED_FRAMEWORKS
        ), f"""{framework} not supported
        Supported frameworks are {", ".join(get_args(SUPPORTED_FRAMEWORKS))}"""

        super().__init__(
            framework=framework,
            repository_local_path=repository_local_path,
        )
        with open(RepositoryMapping[framework]["mapping"], "r") as file:
            self._api_to_doc_mapping = json.load(file)
        if repository_local_path is None:
            self.repository_local_path = get_wandb_artifact(
                artifact_name=RepositoryMapping[framework]["artifact_address"],
                artifact_type="docs",
            )
        else:
            self.repository_local_path = repository_local_path

    def load_doc(self, op: str) -> BaseNode:
        try:
            with open(
                os.path.join(self.repository_local_path, self._api_to_doc_mapping[op]),
                "r",
            ) as file:
                text = file.read()
                return TextNode(
                    text=text, metadata={"file_path": self._api_to_doc_mapping[op]}
                )
        except KeyError:
            raise KeyError(f"API reference for {op} not found")

    @weave.op()
    def predict(self, query: str) -> BaseNode:
        return self.load_doc(op=query)
