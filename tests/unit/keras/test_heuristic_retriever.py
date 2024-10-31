from dotenv import load_dotenv

from ml_frameworks_bot.keras import KerasDocumentationHeuristicRetreiver


load_dotenv()


def test_keras_retriever(keras3_docs):
    retriever = KerasDocumentationHeuristicRetreiver(
        repository_local_path=keras3_docs,
        api_to_doc_mapping_file="mappings/keras3.json",
    )
    retrieved_nodes = retriever.predict(query="keras.layers.Dense")

    assert (
        retrieved_nodes.metadata["file_path"]
        == "sources/api/layers/core_layers/dense.md"
    )
