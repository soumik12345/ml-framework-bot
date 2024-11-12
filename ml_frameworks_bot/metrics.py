from typing import Dict, List, Optional

import weave

from .translation_agent import OpWithAPIReference


class DocumentationAgentJudge(weave.Scorer):
    repository_local_path: str

    @weave.op()
    async def score(
        self,
        model_output: Optional[Dict[str, List[OpWithAPIReference]]],
        ops: List[str],
        api_reference_path: List[str],
    ) -> Dict[str, int | float]:
        retrieved_ops_with_references = model_output["retrieved_ops_with_references"]
        num_correct_ops_extracted, num_api_reference_correct = 0, 0
        for retrieved_op_node in retrieved_ops_with_references:
            if retrieved_op_node.op in ops:
                num_correct_ops_extracted += 1
                base_retrieved_api_reference_path = api_reference_path[
                    ops.index(retrieved_op_node.op)
                ]
                if (
                    base_retrieved_api_reference_path
                    == retrieved_op_node.api_reference_path
                ):
                    num_api_reference_correct += 1
        return {
            "num_correct_ops_extracted": num_correct_ops_extracted,
            "num_api_reference_correct": num_api_reference_correct,
            "op_extraction_accuracy": num_correct_ops_extracted / len(ops),
            "api_reference_retrieval_accuracy": num_api_reference_correct / len(ops),
        }
