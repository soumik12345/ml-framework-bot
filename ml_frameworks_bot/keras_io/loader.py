import os
from typing import Any, List, Optional

import weave
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from ..utils import fetch_git_repository, get_all_file_paths


class KerasIOLoader(weave.Model):
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
        self._embedding_model = OpenAIEmbedding(model=embedding_model_name)

    @weave.op()
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
            os.path.join(self.repository_local_path, "examples")
        )
        input_files += get_all_file_paths(
            os.path.join(self.repository_local_path, "guides")
        )
        input_files += get_all_file_paths(
            os.path.join(self.repository_local_path, "templates")
        )
        reader = SimpleDirectoryReader(input_files=input_files)
        return reader.load_data(num_workers=num_workers, show_progress=True)

    @weave.op()
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
        return splitter.get_nodes_from_documents(documents)

    @weave.op()
    def predict(
        self,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        num_workers: Optional[int] = None,
    ) -> List[BaseNode]:
        documents = self.load_documents(num_workers=num_workers)
        nodes = self.chunk_documents(
            documents,
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        )
        return nodes

    @weave.op()
    def load(
        self,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        num_workers: Optional[int] = None,
    ) -> List[BaseNode]:
        return self.predict(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            num_workers=num_workers,
        )
