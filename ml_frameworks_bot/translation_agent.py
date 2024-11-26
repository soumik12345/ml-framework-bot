from typing import get_args

import weave

from .llm_wrapper import LLMClientWrapper
from .op_extraction import DocumentationRetreiver, OpExtractor
from .retrieval import HeuristicRetreiver
from .utils import SUPPORTED_FRAMEWORKS


class TranslationAgent(weave.Model):
    model: str

    @weave.op()
    def predict(
        self,
        code_snippet: str,
        target_framework: str,
    ):
        # identify source framework
        framework_identification_model = FrameworkIdentificationModel(model=self.model)
        source_framework = framework_identification_model.predict(snippet=code_snippet)

        # initialise retriever
        source_retriever: DocumentationRetreiver = HeuristicRetreiver(
            framework=source_framework,
        )

        # Op Extraction
        source_op_extractor = OpExtractor(
            op_extraction_llm_client=LLMClientWrapper(model_name=self.model),
            retrieval_augmentation_llm_client=LLMClientWrapper(model_name="gpt-4o"),
            api_reference_retriever=source_retriever,
            use_rich=False,
        )
        source_ops = source_op_extractor.predict(code_snippet=code_snippet)  # noqa: F841
        pass


class FrameworkIdentificationModel(weave.Model):
    model: str

    # ruff: noqa: E501
    @weave.op()
    def predict(self, code_snippet: str) -> str:
        return LLMClientWrapper(self.model).predict(
            messages=[
                {
                    "role": "system",
                    "content": f"""
                        You are an experienced machine learning engineer expert in python,
                        and the following frameworks: {", ".join(get_args(SUPPORTED_FRAMEWORKS))}.
                        You will be provided with a code snippet and you need to identify from the aforementioned
                        frameworks the code snippet belongs to. You must follow the following rules:
                        1. If keras is imported using `from tensorflow import keras`, then the framework is keras2.
                            Also, if tensorflow is imported using `import tensorflow as tf` and then keras is used as
                            `tf.keras`, then the framework is keras2.
                        2. If keras is imported using `import keras` without it being imported
                            from tensorflow, then the framework is keras3.
                    """,
                },
                {
                    "role": "user",
                    "content": code_snippet,
                },
            ],
            response_model=SUPPORTED_FRAMEWORKS,
        )
