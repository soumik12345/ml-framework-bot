import asyncio

import weave
from dotenv import load_dotenv

from ml_frameworks_bot.keras import KerasDocumentationAgent, KerasDocumentationRetreiver
from ml_frameworks_bot.keras.metrics import KerasDocumentationAgentJudge

load_dotenv()


def test_keras_retriever():
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


def test_keras_docs_agent(repository_local_path="keras_docs"):
    load_dotenv()
    weave.init(project_name="ml-colabs/ml-frameworks-bot")
    api_reference_retriever = KerasDocumentationRetreiver.from_wandb_artifact(
        artifact_address="ml-colabs/ml-frameworks-bot/keras3_api_reference:latest"
    )
    keras_docs_agent = KerasDocumentationAgent(
        llm_name="o1-preview",
        api_reference_retriever=api_reference_retriever,
        use_rich_progressbar=False,
    )
    evaluation = weave.Evaluation(
        dataset=weave.ref("keras_evaluation_dataset:v0").get(),
        scorers=[
            KerasDocumentationAgentJudge(repository_local_path=repository_local_path)
        ],
    )
    summary = asyncio.run(evaluation.evaluate(keras_docs_agent))
    assert (
        summary["KerasDocumentationAgentJudge"]["api_reference_retrieval_accuracy"][
            "mean"
        ]
        > 0.8
    )
    assert (
        summary["KerasDocumentationAgentJudge"]["op_extraction_accuracy"]["mean"] > 0.8
    )
