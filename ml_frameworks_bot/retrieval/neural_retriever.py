import os
from typing import List, Optional, Tuple, Union

import torch
import weave
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import BaseNode, Document, NodeWithScore, TextNode
from rich.progress import track

import wandb

from ..utils import (
    get_all_file_paths,
    get_wandb_artifact,
    make_embedding_model,
    upload_file_as_artifact,
)
from .common import FrameworkParams, RepositoryMapping


class NeuralRetreiver(weave.Model):
    framework: str
    embedding_model_name: str
    repository_local_path: Optional[str]
    similarity_top_k: int
    torch_dtype: str
    _vector_index: VectorStoreIndex = None
    _retreival_engine: BaseRetriever = None

    def __init__(
        self,
        framework: str,
        embedding_model_name: str,
        torch_dtype: torch.dtype,
        similarity_top_k: int = 10,
        repository_local_path: Optional[str] = None,
        vector_index: Optional[VectorStoreIndex] = None,
    ):
        super().__init__(
            framework=framework,
            embedding_model_name=embedding_model_name,
            similarity_top_k=similarity_top_k,
            repository_local_path=repository_local_path,
            torch_dtype=str(torch_dtype),
        )
        self.repository_local_path = repository_local_path
        self._vector_index = vector_index
        if self.repository_local_path is None and self._vector_index is None:
            raise ValueError(
                "Both `repository_local_path` and `vector_index` cannot be `None`."
            )
        Settings.embed_model = make_embedding_model(
            self.embedding_model_name, model_kwargs={"torch_dtype": torch_dtype}
        )

    @classmethod
    def from_wandb_artifact(
        cls,
        artifact_address: str,
        torch_dtype: torch.dtype = torch.float16,
        similarity_top_k: int = 10,
    ) -> "NeuralRetreiver":
        artifact_dir, metadata = get_wandb_artifact(
            artifact_name=artifact_address,
            artifact_type="vector_index",
            get_metadata=True,
        )
        embedding_model_name = metadata.get("embedding_model_name")
        Settings.embed_model = make_embedding_model(embedding_model_name)
        vector_index = load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir=artifact_dir)
        )
        _cls = cls(
            framework=metadata.get("framework"),
            embedding_model_name=embedding_model_name,
            similarity_top_k=similarity_top_k,
            vector_index=vector_index,
            torch_dtype=torch_dtype,
        )
        return _cls

    def load_documents(
        self,
        return_nodes: bool,
    ) -> List[Union[BaseNode, Document]]:
        # Load documents if not available locally
        if self.repository_local_path is None:
            self.repository_local_path = get_wandb_artifact(
                artifact_name=RepositoryMapping[self.framework]["artifact_address"],
                artifact_type="docs",
            )

        # Determine which files to index
        input_files = []
        for directory in FrameworkParams[self.framework]["included_directories"]:
            input_files += get_all_file_paths(
                os.path.join(self.repository_local_path, directory),
                included_file_extensions=[".md"],
            )

        # Create nodes
        document_nodes = []
        for file_path in track(input_files, description="Loading documents:"):
            # Exclude files with certain postfixes
            exclude_file = False
            for exclusion in FrameworkParams[self.framework]["exclude_file_postfixes"]:
                if file_path.endswith(exclusion):
                    exclude_file = True
                    break

            if not exclude_file:
                with open(file_path, "r") as file:
                    text = file.read()

                if FrameworkParams[self.framework]["chunk_on_separator"]:
                    document_nodes.extend(
                        split_by_separator(
                            split_pattern=FrameworkParams[self.framework][
                                "split_pattern"
                            ],
                            file_path=file_path,
                            text=text,
                        )
                    )
                else:
                    document_nodes.append(
                        (TextNode if return_nodes else Document)(
                            text=text,
                            metadata={
                                "file_path": file_path.replace(
                                    self.repository_local_path + "/", ""
                                )
                            },
                        )
                    )
        return document_nodes

    def index_documents(
        self,
        return_nodes: bool = True,
        vector_index_persist_dir: Optional[str] = None,
        artifact_name: Optional[str] = None,
        artifact_aliases: Optional[List[str]] = [],
    ) -> VectorStoreIndex:
        if self.repository_local_path is not None:
            # Load documents as nodes
            document_nodes = self.load_documents(return_nodes=return_nodes)

            # Index documents
            self._vector_index = VectorStoreIndex(nodes=document_nodes)
            assert (
                len(document_nodes) == len(self._vector_index.docstore.docs)
            ), f"No. of document nodes {len(document_nodes)} != No. of nodes in VectorIndex {len(self._vector_index.docstore.docs)}"  # noqa: E501

            # Persist the index as a wandb artifact
            if vector_index_persist_dir:
                assert (
                    wandb.run is not None
                ), "Attempted to log artifact without wandb run"
                self._vector_index.storage_context.persist(
                    persist_dir=vector_index_persist_dir
                )
                upload_file_as_artifact(
                    path=vector_index_persist_dir,
                    artifact_name=artifact_name,
                    artifact_metadata={
                        "framework": self.framework,
                        "embedding_model_name": self.embedding_model_name,
                        "included_directories": FrameworkParams[self.framework][
                            "included_directories"
                        ],
                        "exclude_file_postfixes": FrameworkParams[self.framework][
                            "exclude_file_postfixes"
                        ],
                        "torch_dtype": self.torch_dtype,
                    },
                    artifact_aliases=artifact_aliases,
                )
        return self._vector_index

    @weave.op()
    def predict(self, query: str) -> List[NodeWithScore]:
        if self._retreival_engine is None:
            self._retreival_engine = self._vector_index.as_retriever(
                similarity_top_k=self.similarity_top_k,
            )
        retreived_nodes = self._retreival_engine.retrieve(query)
        return retreived_nodes


def split_by_separator(
    split_pattern: List[Tuple[str, str]], file_path: str, text: str
) -> List[TextNode]:
    """
    Split the text content of a file on the separator string.

    Args:
        chunk_on_separator (List[List[str]]): Each item is a 2-Tuple, consisting of
            the common substring and the separator string. Each file path containing
            the common substring will be split on the separator string and each
            chunk will be treated as a separate TextNode.
        file_path (str): The path of the file to split.
        text (str): Text content of the file.

    Returns:
        List[TextNode]: List of TextNodes split on the separator string.
    """
    for pattern in split_pattern:
        if file_path.contains(pattern[0]):
            texts = text.split(pattern[1])
            nodes = [TextNode(text=text) for text in texts]
            return nodes
