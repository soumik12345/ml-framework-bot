import os
import subprocess
from glob import glob
from typing import Callable, List

import torch
import weave
from git import Repo
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


def fetch_git_repository(
    repository_local_path: str,
    username: str,
    repo_name: str,
    personal_access_token: str,
) -> None:
    os.makedirs(repository_local_path, exist_ok=True)
    if len(glob(os.path.join(repository_local_path, "*"))) == 0:
        repository_url = f"https://{personal_access_token}:x-oauth-basic@github.com/{username}/{repo_name}"
        repository = Repo.clone_from(repository_url, repository_local_path)
    else:
        repository = Repo(repository_local_path)
    repository.remotes.origin.pull()


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


def make_embedding_model(embedding_model_name: str) -> BaseEmbedding:
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
            model_kwargs={"torch_dtype": torch.float16},
        )
    )


def weave_op_wrapper(name: str) -> Callable[[Callable], Callable]:
    def wrapper(fn: Callable) -> Callable:
        op = weave.op()(fn)
        op.name = name
        return op

    return wrapper


def build_keras_io_sources(repository_local_path: str):
    working_directory = os.getcwd()
    os.chdir(repository_local_path)
    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    subprocess.run(["pip", "install", "keras-nlp==0.14.4"])
    os.chdir("scripts")
    subprocess.run(["python", "autogen.py", "make"])
    os.chdir(working_directory)