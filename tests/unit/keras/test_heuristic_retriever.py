from dotenv import load_dotenv

from ml_frameworks_bot.keras import KerasDocumentationHeuristicRetreiver

load_dotenv()


def test_keras_retriever():
    retriever = KerasDocumentationHeuristicRetreiver(
        repository_local_path="artifacts/keras_docs"
    )
    retrieved_nodes = retriever.predict(keras_op="keras.layers.Dense")

    assert (
        retrieved_nodes.metadata["file_path"]
        == "sources/api/layers/core_layers/dense.md"
    )
