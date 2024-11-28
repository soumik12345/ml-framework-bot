from typing import get_args

import litellm
import weave
from pydantic import BaseModel
from .op_extraction import DocumentationRetreiver, OpExtractor
from .retrieval import HeuristicRetreiver
from .utils import SUPPORTED_FRAMEWORKS, get_structured_output_from_completion


class SupportedFrameworks(BaseModel):
    frameworks: SUPPORTED_FRAMEWORKS


class FrameworkIdentificationModel(weave.Model):
    model_name: str

    # ruff: noqa: E501
    @weave.op()
    def predict(self, code_snippet: str) -> str:
        completion = litellm.completion(
            model=self.model_name,
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
            response_format=SupportedFrameworks,
        )
        return get_structured_output_from_completion(
            completion,
            response_format=SupportedFrameworks,
        ).frameworks



class TranslationAgent(weave.Model):
    model_name: str
    verbose: bool = True

    @weave.op()
    def predict(
        self,
        code_snippet: str,
        # target_framework: str,
    ):
        # identify source framework
        framework_identification_model = FrameworkIdentificationModel(model_name=self.model_name)
        source_framework = framework_identification_model.predict(code_snippet=code_snippet)

        # initialise retriever
        source_retriever: DocumentationRetreiver = HeuristicRetreiver(
            framework=source_framework,
        )

        # Op Extraction
        source_op_extractor = OpExtractor(
            model_name=self.model_name,
            api_reference_retriever=source_retriever,
            verbose=self.verbose,
        )
        source_ops = source_op_extractor.predict(code_snippet=code_snippet)  # noqa: F841
        return source_ops
