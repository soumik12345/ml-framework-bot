from dotenv import load_dotenv

from ml_frameworks_bot.keras import KerasDocumentationRetreiver

load_dotenv()


def test_retriever():
    retriever = KerasDocumentationRetreiver.from_wandb_artifact(
        artifact_address="ml-colabs/ml-frameworks-bot/keras3_api_reference:latest"
    )
    retrieved_nodes = retriever.predict(
        query="Fetch the API referece for `keras.layers.Dense`"
    )
    assert (
        retrieved_nodes[0].node.metadata["file_path"]
        == "keras_docs/sources/api/layers/core_layers/dense.md"
    )
