import os
from typing import Any, Callable, Dict, List, Optional

import wandb
import weave
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


def get_all_file_paths(
    directory: str, included_file_extensions: List[str]
) -> List[str]:
    file_paths = []

    def recurse_folder(folder):
        for entry in os.scandir(folder):
            if entry.is_dir():
                recurse_folder(entry.path)
            else:
                for file_ext in included_file_extensions:
                    if entry.path.endswith(file_ext):
                        file_paths.append(entry.path)

    recurse_folder(directory)

    return file_paths


def make_embedding_model(
    embedding_model_name: str, model_kwargs: Optional[Dict[str, Any]] = {}
) -> BaseEmbedding:
    return (
        OpenAIEmbedding(model_name=embedding_model_name)
        if embedding_model_name
        in [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]
        else HuggingFaceEmbedding(
            model_name=embedding_model_name,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
    )


def weave_op_wrapper(name: str) -> Callable[[Callable], Callable]:
    def wrapper(fn: Callable) -> Callable:
        op = weave.op()(fn)
        op.name = name
        return op

    return wrapper


def get_wandb_artifact(
    artifact_name: str,
    artifact_type: str,
    get_metadata: bool = False,
) -> str:
    if wandb.run:
        artifact = wandb.use_artifact(artifact_name, type=artifact_type)
        artifact_dir = artifact.download()
    else:
        api = wandb.Api()
        artifact = api.artifact(artifact_name)
        artifact_dir = artifact.download()
    if get_metadata:
        return artifact_dir, artifact.metadata
    return artifact_dir
