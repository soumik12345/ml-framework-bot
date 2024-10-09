import pytest

import wandb


@pytest.fixture
def keras3_docs() -> str:
    api = wandb.Api()
    artifact = api.artifact("ml-colabs/ml-frameworks-bot/keras3-docs:latest")
    artifact_dir = artifact.download()
    return artifact_dir
