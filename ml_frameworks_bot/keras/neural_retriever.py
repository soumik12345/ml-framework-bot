import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import weave
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode, Document, NodeWithScore, TextNode
from rich.progress import track

import wandb

from ..utils import (
    get_all_file_paths,
    make_embedding_model,
)


class KerasDocumentationRetreiver(weave.Model):
    embedding_model_name: str
    repository_local_path: Optional[str]
    similarity_top_k: int
    torch_dtype: str
    repository: str = "https://github.com/keras-team/keras-io"
    wandb_artifact_address: Optional[str] = None
    _vector_index: VectorStoreIndex = None
    _retreival_engine: BaseRetriever = None

    def __init__(
        self,
        embedding_model_name: str,
        torch_dtype: torch.dtype,
        similarity_top_k: int = 10,
        repository_local_path: Optional[str] = None,
        vector_index: Optional[VectorStoreIndex] = None,
    ):
        super().__init__(
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
    ):
        api = wandb.Api()
        artifact = api.artifact(artifact_address)
        artifact_dir = artifact.download()
        embedding_model_name = artifact.metadata.get("embedding_model_name")
        Settings.embed_model = make_embedding_model(embedding_model_name)
        vector_index = load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir=artifact_dir)
        )
        _cls = cls(
            embedding_model_name=embedding_model_name,
            similarity_top_k=similarity_top_k,
            vector_index=vector_index,
            torch_dtype=torch_dtype,
        )
        _cls.wandb_artifact_address = artifact_address
        return _cls

    def load_documents(
        self,
        included_directories: List[str] = ["examples", "guides", "templates"],
        exclude_file_postfixes: List[str] = ["index.md"],
        chunk_on_separator: List[Tuple[str, str]] = [["ops", "----"]],
        return_nodes: bool = True,
    ) -> List[Union[BaseNode, Document]]:
        """
        Args:
            chunk_on_separator (List[List[str]]): Each item is a 2-Tuple, consisting of
                the common substring and the separator string. Each file path containing
                the common substring will be split on the separator string and each
                chunk will be treated as a separate TextNode.
        """
        if self.repository_local_path is None:
            api = wandb.Api()
            artifact = api.artifact("ml-colabs/ml-frameworks-bot/keras3-docs:latest")
            artifact_dir = artifact.download()
            self.repository_local_path = artifact_dir
        input_files = []
        for directory in included_directories:
            input_files += get_all_file_paths(
                os.path.join(self.repository_local_path, directory),
                included_file_extensions=[".md"],
            )
        document_nodes = []
        for file_path in track(input_files, description="Loading documents:"):
            exclude_file = False
            for exclusion in exclude_file_postfixes:
                if file_path.endswith(exclusion):
                    exclude_file = True
                    break
            if not exclude_file:
                with open(file_path, "r") as file:
                    text = file.read()

                if chunk_on_separator:
                    for pattern in chunk_on_separator:
                        if file_path.contains(pattern[0]):
                            texts = text.split(pattern[1])
                            nodes = [TextNode(text=text) for text in texts]
                            document_nodes.append(nodes)
                else:
                    document_nodes.append(
                        TextNode(
                            text=text,
                            metadata={
                                "file_path": file_path.replace(
                                    self.repository_local_path + "/", ""
                                )
                            },
                        )
                        if return_nodes
                        else Document(
                            text=text,
                            metadata={
                                "file_path": file_path.replace(
                                    self.repository_local_path + "/", ""
                                )
                            },
                        )
                    )
        return document_nodes

    def chunk_documents(
        self,
        documents: List[Document],
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
    ) -> List[BaseNode]:
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=Settings.embed_model,
        )
        return splitter.get_nodes_from_documents(documents, show_progress=True)

    def index_documents(
        self,
        included_directories: List[str] = ["examples", "guides", "templates"],
        exclude_file_postfixes: List[str] = ["index.md"],
        apply_chunking: bool = False,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        vector_index_persist_dir: Optional[str] = None,
        artifact_name: Optional[str] = None,
        artifact_metadata: Optional[Dict[str, Any]] = {},
        artifact_aliases: Optional[List[str]] = [],
        track_load_documents: bool = False,
        track_chunk_documents: bool = False,
    ) -> VectorStoreIndex:
        if self.repository_local_path is not None:
            load_document_fn = (
                weave.op()(self.load_documents)
                if track_load_documents
                else self.load_documents
            )
            document_nodes = load_document_fn(
                included_directories=included_directories,
                exclude_file_postfixes=exclude_file_postfixes,
                return_nodes=not apply_chunking,
            )
            chunk_document_fn = (
                weave.op()(self.chunk_documents)
                if track_chunk_documents
                else self.chunk_documents
            )
            document_nodes = (
                chunk_document_fn(
                    documents=document_nodes,
                    buffer_size=buffer_size,
                    breakpoint_percentile_threshold=breakpoint_percentile_threshold,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                if apply_chunking
                else document_nodes
            )
            self._vector_index = VectorStoreIndex(nodes=document_nodes)
            assert (
                len(document_nodes) == len(self._vector_index.docstore.docs)
            ), f"No. of document nodes {len(document_nodes)} != No. of nodes in VectorIndex {len(self._vector_index.docstore.docs)}"  # noqa: E501
            if vector_index_persist_dir:
                self._vector_index.storage_context.persist(
                    persist_dir=vector_index_persist_dir
                )
                if wandb.run and artifact_name:
                    artifact_metadata = {
                        **artifact_metadata,
                        **{
                            "embedding_model_name": self.embedding_model_name,
                            "apply_chunking": apply_chunking,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "buffer_size": buffer_size,
                            "breakpoint_percentile_threshold": breakpoint_percentile_threshold,  # noqa: E501
                            "included_directories": included_directories,
                            "torch_dtype": self.torch_dtype,
                        },
                    }
                    artifact_aliases.append("latest")
                    artifact = wandb.Artifact(
                        name=artifact_name,
                        type="vector_index",
                        metadata=artifact_metadata,
                    )
                    artifact.add_dir(local_path=vector_index_persist_dir)
                    wandb.log_artifact(artifact, aliases=artifact_aliases)
        return self._vector_index

    @weave.op()
    def predict(self, query: str) -> List[NodeWithScore]:
        if self._retreival_engine is None:
            self._retreival_engine = self._vector_index.as_retriever(
                similarity_top_k=self.similarity_top_k,
            )
        retreived_nodes = self._retreival_engine.retrieve(query)
        return retreived_nodes
