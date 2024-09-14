from dotenv import load_dotenv
from ml_frameworks_bot.keras_io import KerasIORetreiver

load_dotenv()


def test_retriever():
    retriever = KerasIORetreiver.from_wandb_artifact(
        artifact_address="ml-colabs/ml-frameworks-bot/keras3_api_reference:latest"
    )
    retrieved_nodes = retriever.predict(
        query="Fetch the API referece for `keras.layers.Dense`"
    )

    assert (
        retrieved_nodes[0].to_dict()["node"]["metadata"]["file_path"]
        == "keras_docs/sources/api/layers/core_layers/dense.md"
    )
