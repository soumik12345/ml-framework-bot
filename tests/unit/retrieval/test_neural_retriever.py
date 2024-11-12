import pytest
from dotenv import load_dotenv

from ml_frameworks_bot.retrieval import NeuralRetreiver


load_dotenv()


@pytest.mark.parametrize(
    "artifact_address, query, expected_path",
    [
        (
            "ml-colabs/ml-frameworks-bot/keras3_api_reference:latest",
            "keras.layers.Dense",
            "sources/api/layers/core_layers/dense.md",
        ),
    ],
    ids=["keras3"],
)
def test_keras_retriever(artifact_address: str, query: str, expected_path: str) -> None:
    retriever = NeuralRetreiver.from_wandb_artifact(artifact_address=artifact_address)
    retrieved_nodes = retriever.predict(query=f"Fetch the API referece for `{query}`")
    assert retrieved_nodes[0].node.metadata["file_path"] == expected_path
