import os
from typing import Any, Dict, List, Optional

import weave
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode, Document, NodeWithScore

import wandb

from ..utils import fetch_git_repository, get_all_file_paths, make_embedding_model


class KerasIORetreiver(weave.Model):
    embedding_model_name: str
    repository_local_path: Optional[str]
    similarity_top_k: int
    repository: str = "https://github.com/keras-team/keras-io"
    wandb_artifact_address: Optional[str] = None
    _vector_index: VectorStoreIndex = None
    _retreival_engine: BaseRetriever = None

    def __init__(
        self,
        embedding_model_name: str,
        similarity_top_k: int = 10,
        repository_local_path: Optional[str] = None,
        vector_index: Optional[VectorStoreIndex] = None,
    ):
        super().__init__(
            embedding_model_name=embedding_model_name,
            similarity_top_k=similarity_top_k,
            repository_local_path=repository_local_path,
        )
        self.repository_local_path = repository_local_path
        self._vector_index = vector_index
        if self.repository_local_path is None and self._vector_index is None:
            raise ValueError(
                "Both `repository_local_path` and `vector_index` cannot be `None`."
            )
        Settings.embed_model = make_embedding_model(self.embedding_model_name)

    @classmethod
    def from_wandb_artifact(cls, artifact_address: str, similarity_top_k: int = 10):
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
        )
        _cls.wandb_artifact_address = artifact_address
        return _cls

    def load_documents(
        self,
        included_directories: List[str] = ["examples", "guides", "templates"],
        num_workers: Optional[int] = None,
    ) -> List[Document]:
        repository_owner = self.repository.split("/")[-2]
        repository_name = self.repository.split("/")[-1]
        personal_access_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        fetch_git_repository(
            self.repository_local_path,
            repository_owner,
            repository_name,
            personal_access_token,
        )
        input_files = []
        for directory in included_directories:
            input_files += get_all_file_paths(
                os.path.join(self.repository_local_path, directory),
                included_file_extensions=[".md"],
            )
        reader = SimpleDirectoryReader(input_files=input_files)
        return reader.load_data(num_workers=num_workers, show_progress=True)

    def chunk_documents(
        self,
        documents: List[Document],
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
    ) -> List[BaseNode]:
        splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=Settings.embed_model,
        )
        return splitter.get_nodes_from_documents(documents, show_progress=True)

    def index_documents(
        self,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        included_directories: List[str] = ["examples", "guides", "templates"],
        num_workers: Optional[int] = None,
        build_index_from_documents: bool = True,
        vector_index_persist_dir: Optional[str] = None,
        artifact_name: Optional[str] = None,
        artifact_metadata: Optional[Dict[str, Any]] = {},
    ) -> VectorStoreIndex:
        if self.repository_local_path is not None:
            documents = self.load_documents(
                included_directories=included_directories, num_workers=num_workers
            )
            nodes = self.chunk_documents(
                documents,
                buffer_size=buffer_size,
                breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            )
            self._vector_index = (
                VectorStoreIndex.from_documents(
                    documents, show_progress=True, node_parser=nodes
                )
                if build_index_from_documents
                else VectorStoreIndex(nodes=nodes, show_progress=True)
            )
            if vector_index_persist_dir:
                self._vector_index.storage_context.persist(
                    persist_dir=vector_index_persist_dir
                )
                if wandb.run and artifact_name:
                    artifact_metadata["embedding_model_name"] = (
                        self.embedding_model_name
                    )
                    artifact = wandb.Artifact(
                        name=artifact_name,
                        type="vector_index",
                        metadata=artifact_metadata,
                    )
                    artifact.add_dir(local_path=vector_index_persist_dir)
                    artifact.save()
        return self._vector_index

    @weave.op()
    def predict(self, query: str) -> List[NodeWithScore]:
        if self._retreival_engine is None:
            self._retreival_engine = self._vector_index.as_retriever(
                similarity_top_k=self.similarity_top_k
            )
        retreived_nodes = self._retreival_engine.retrieve(query)
        return retreived_nodes
