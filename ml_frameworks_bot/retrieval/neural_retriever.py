import os
from typing import Optional

import torch
import weave
from sentence_transformers import SentenceTransformer
from rich.progress import track

from ..utils import get_torch_backend


from ..utils import (
    get_all_file_paths,
    get_wandb_artifact,
)
from .common import FrameworkParams, RepositoryMapping


def split_by_separator(
    split_pattern: list[tuple[str, str]], file_path: str, text: str
) -> list[dict[str, str]]:
    for pattern in split_pattern:
        if file_path.contains(pattern[0]):
            texts = text.split(pattern[1])
            return [{"file_path": file_path, "text": text} for text in texts]


class NeuralRetreiver(weave.Model):
    framework: str
    embedding_model_name: str
    _model: Optional[SentenceTransformer] = None

    def model_post_init(self, __context):
        self._model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
            device=get_torch_backend(),
        )

    def load_documents(self) -> list[dict[str, str]]:
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
                directory=os.path.join(self.repository_local_path, directory),
                included_file_extensions=FrameworkParams[self.framework][
                    "included_file_extensions"
                ],
            )

        # Create nodes
        documents = []
        for file_path in track(input_files, description="Loading documents:"):
            # Exclude files with certain postfixes
            exclude_file = False
            if "exclude_file_postfixes" in FrameworkParams[self.framework]:
                for exclusion in FrameworkParams[self.framework][
                    "exclude_file_postfixes"
                ]:
                    if file_path.endswith(exclusion):
                        exclude_file = True
                        break

            if not exclude_file:
                with open(file_path, "r") as file:
                    text = file.read()

                if FrameworkParams[self.framework]["chunk_on_separator"]:
                    documents.extend(
                        split_by_separator(
                            split_pattern=FrameworkParams[self.framework][
                                "split_pattern"
                            ],
                            file_path=file_path,
                            text=text,
                        )
                    )
                else:
                    documents.append({
                        "file_path": file_path.replace(
                            self.repository_local_path + "/", ""
                        ),
                        "text": text,
                    })
        return documents
