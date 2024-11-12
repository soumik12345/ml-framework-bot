import asyncio

import weave
from dotenv import load_dotenv

from ml_frameworks_bot.llm_wrapper import LLMClientWrapper
from ml_frameworks_bot.metrics import DocumentationAgentJudge
from ml_frameworks_bot.retrieval import HeuristicRetreiver, NeuralRetreiver
from ml_frameworks_bot.translation_agent import TranslationAgent


load_dotenv()


def test_agent_neural_retriever(keras3_docs):
    load_dotenv()
    weave.init(project_name="ml-colabs/ml-frameworks-bot")
    api_reference_retriever = NeuralRetreiver.from_wandb_artifact(
        artifact_address="ml-colabs/ml-frameworks-bot/keras3_api_reference:latest"
    )
    agent = TranslationAgent(
        op_extraction_llm_client=LLMClientWrapper(
            model_name="claude-3-5-sonnet-20240620"
        ),
        retrieval_augmentation_llm_client=LLMClientWrapper(model_name="gpt-4o"),
        api_reference_retriever=api_reference_retriever,
        use_rich=False,
    )
    evaluation = weave.Evaluation(
        dataset=weave.ref("keras_evaluation_dataset:v2").get(),
        scorers=[DocumentationAgentJudge(repository_local_path=keras3_docs)],
    )
    summary = asyncio.run(evaluation.evaluate(agent))
    assert (
        summary["DocumentationAgentJudge"]["api_reference_retrieval_accuracy"]["mean"]
        > 0.6
    )
    assert summary["DocumentationAgentJudge"]["op_extraction_accuracy"]["mean"] > 0.6


def test_agent_heuristic_retriever(keras3_docs):
    load_dotenv()
    weave.init(project_name="ml-colabs/ml-frameworks-bot")
    api_reference_retriever = HeuristicRetreiver(
        framework="keras3",
        repository_local_path=keras3_docs,
    )
    agent = TranslationAgent(
        op_extraction_llm_client=LLMClientWrapper(
            model_name="claude-3-5-sonnet-20240620"
        ),
        retrieval_augmentation_llm_client=LLMClientWrapper(model_name="gpt-4o"),
        api_reference_retriever=api_reference_retriever,
        use_rich=False,
    )
    evaluation = weave.Evaluation(
        dataset=weave.ref("keras_evaluation_dataset:v2").get(),
        scorers=[DocumentationAgentJudge(repository_local_path=keras3_docs)],
    )
    summary = asyncio.run(evaluation.evaluate(agent))
    assert (
        summary["DocumentationAgentJudge"]["api_reference_retrieval_accuracy"]["mean"]
        == 1.0
    )
    assert summary["DocumentationAgentJudge"]["op_extraction_accuracy"]["mean"] > 0.8
