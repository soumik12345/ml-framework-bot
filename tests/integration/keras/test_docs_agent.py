import asyncio

import weave
from dotenv import load_dotenv

from ml_frameworks_bot.keras import (
    KerasDocumentationAgent,
    KerasDocumentationHeuristicRetreiver,
    KerasDocumentationRetreiver,
)
from ml_frameworks_bot.keras.metrics import KerasDocumentationAgentJudge
from ml_frameworks_bot.llm_wrapper import LLMClientWrapper


load_dotenv()


def test_keras_docs_agent_neural_retriever(keras3_docs):
    load_dotenv()
    weave.init(project_name="ml-colabs/ml-frameworks-bot")
    api_reference_retriever = KerasDocumentationRetreiver.from_wandb_artifact(
        artifact_address="ml-colabs/ml-frameworks-bot/keras3_api_reference:latest"
    )
    keras_docs_agent = KerasDocumentationAgent(
        op_extraction_llm_client=LLMClientWrapper(
            model_name="claude-3-5-sonnet-20240620"
        ),
        retrieval_augmentation_llm_client=LLMClientWrapper(model_name="gpt-4o"),
        api_reference_retriever=api_reference_retriever,
        use_rich=False,
    )
    evaluation = weave.Evaluation(
        dataset=weave.ref("keras_evaluation_dataset:v1").get(),
        scorers=[KerasDocumentationAgentJudge(repository_local_path=keras3_docs)],
    )
    summary = asyncio.run(evaluation.evaluate(keras_docs_agent))
    assert (
        summary["KerasDocumentationAgentJudge"]["api_reference_retrieval_accuracy"][
            "mean"
        ]
        > 0.6
    )
    assert (
        summary["KerasDocumentationAgentJudge"]["op_extraction_accuracy"]["mean"] > 0.6
    )


def test_keras_docs_agent_heuristic_retriever(keras3_docs):
    load_dotenv()
    weave.init(project_name="ml-colabs/ml-frameworks-bot")
    api_reference_retriever = KerasDocumentationHeuristicRetreiver(
        repository_local_path=keras3_docs,
        api_to_doc_mapping_file="mappings/keras3.json",
    )
    keras_docs_agent = KerasDocumentationAgent(
        op_extraction_llm_client=LLMClientWrapper(
            model_name="claude-3-5-sonnet-20240620"
        ),
        retrieval_augmentation_llm_client=LLMClientWrapper(model_name="gpt-4o"),
        api_reference_retriever=api_reference_retriever,
        use_rich=False,
    )
    evaluation = weave.Evaluation(
        dataset=weave.ref("keras_evaluation_dataset:v1").get(),
        scorers=[KerasDocumentationAgentJudge(repository_local_path=keras3_docs)],
    )
    summary = asyncio.run(evaluation.evaluate(keras_docs_agent))
    assert (
        summary["KerasDocumentationAgentJudge"]["api_reference_retrieval_accuracy"][
            "mean"
        ]
        == 1.0
    )
    assert (
        summary["KerasDocumentationAgentJudge"]["op_extraction_accuracy"]["mean"] > 0.8
    )
