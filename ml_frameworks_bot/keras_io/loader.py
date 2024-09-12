import os
from typing import Any, List, Optional

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel

from ..utils import fetch_git_repository, get_all_file_paths


class KerasIOLoader(BaseModel):
    repository_local_path: str
    embedding_model_name: str
    repository: str = "https://github.com/keras-team/keras-io"
    _embedding_model: BaseEmbedding = None
    _vector_index: Any = None

    def __init__(self, repository_local_path: str, embedding_model_name: str):
        super().__init__(
            repository_local_path=repository_local_path,
            embedding_model_name=embedding_model_name,
        )
        self.repository_local_path = repository_local_path
        self._embedding_model = (
            OpenAIEmbedding(model_name=embedding_model_name)
            if embedding_model_name
            in [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ]
            else HuggingFaceEmbedding(model_name=embedding_model_name)
        )

    def load_documents(self, num_workers: Optional[int] = None) -> List[Document]:
        repository_owner = self.repository.split("/")[-2]
        repository_name = self.repository.split("/")[-1]
        personal_access_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        fetch_git_repository(
            self.repository_local_path,
            repository_owner,
            repository_name,
            personal_access_token,
        )
        input_files = get_all_file_paths(
            os.path.join(self.repository_local_path, "examples"),
            included_file_extensions=[".md"],
        )
        input_files += get_all_file_paths(
            os.path.join(self.repository_local_path, "guides"),
            included_file_extensions=[".md"],
        )
        input_files += get_all_file_paths(
            os.path.join(self.repository_local_path, "templates"),
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
            embed_model=self._embedding_model,
        )
        return splitter.get_nodes_from_documents(documents, show_progress=True)

    def load(
        self,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        num_workers: Optional[int] = None,
        build_index_from_documents: bool = True,
        vector_index_persist_dir: Optional[str] = None,
    ) -> VectorStoreIndex:
        documents = self.load_documents(num_workers=num_workers)
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
        return self._vector_index
