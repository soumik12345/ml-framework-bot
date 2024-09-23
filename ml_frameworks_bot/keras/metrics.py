from typing import Dict, List, Optional

import weave

from ml_frameworks_bot.keras.docs_agent import KerasOpWithAPIReference


class KerasDocumentationAgentJudge(weave.Scorer):
    repository_local_path: str

    @weave.op()
    async def score(
        self,
        model_output: Optional[Dict[str, List[KerasOpWithAPIReference]]],
        keras_ops: List[str],
        keras_api_reference_path: List[str],
    ):
        retrieved_keras_ops_with_references = model_output[
            "retrieved_keras_ops_with_references"
        ]
        num_correct_ops_extracted, num_api_reference_correct = 0, 0
        for retrieved_keras_op_node in retrieved_keras_ops_with_references:
            if retrieved_keras_op_node.keras_op in keras_ops:
                num_correct_ops_extracted += 1
                base_retrieved_api_reference_path = keras_api_reference_path[
                    keras_ops.index(retrieved_keras_op_node.keras_op)
                ].replace("sources/", "")
                if (
                    base_retrieved_api_reference_path
                    == retrieved_keras_op_node.api_reference_path.replace(
                        f"{self.repository_local_path}/sources/", ""
                    )
                ):
                    num_api_reference_correct += 1
        return {
            "num_correct_ops_extracted": num_correct_ops_extracted,
            "num_api_reference_correct": num_api_reference_correct,
            "op_extraction_accuracy": num_correct_ops_extracted / len(keras_ops),
            "api_reference_retrieval_accuracy": num_api_reference_correct
            / len(keras_ops),
        }
