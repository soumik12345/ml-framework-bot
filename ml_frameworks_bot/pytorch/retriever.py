import os
import weave
import torch

from typing import Optional

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.base.base_retriever import BaseRetriever

from ml_frameworks_bot.utils import (
    make_embedding_model,
    fetch_git_repository,
    build_pytorch_sources,
)


class PyTorchRetriever(weave.Model):
    embedding_model_name: str
    repository_local_path: Optional[str]
    similarity_top_k: int
    torch_dtype: str
    repository: str = "https://github.com/pytorch/pytorch"
    wandb_artifact_address: Optional[str] = None
    _vector_index: VectorStoreIndex = None
    _retrieval_engine: BaseRetriever = None

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

    def load_documents(
        self,
    ) -> None:
        repository_owner = self.repository.split("/")[-2]
        repository_name = self.repository.split("/")[-1]
        personal_access_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        fetch_git_repository(
            self.repository_local_path,
            repository_owner,
            repository_name,
            personal_access_token,
        )
        source_directory = os.path.join(self.repository_local_path, "sources")
        if not os.path.exists(source_directory):
            build_pytorch_sources(repository_local_path=self.repository_local_path)
