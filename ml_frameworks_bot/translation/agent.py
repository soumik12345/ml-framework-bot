import instructor
import weave
from instructor import Instructor
from litellm import completion

from ..keras import KerasDocumentationAgent


class Keras3MigrationAgent(weave.Model):
    llm_name: str
    source_framework_docs_agent: KerasDocumentationAgent
    target_framework_docs_agent: KerasDocumentationAgent
    _llm_client: Instructor

    def __init__(
        self,
        llm_name: str,
        source_framework_docs_agent: KerasDocumentationAgent,
        target_framework_docs_agent: KerasDocumentationAgent,
    ):
        super().__init__(
            llm_name=llm_name,
            source_framework_docs_agent=source_framework_docs_agent,
            target_framework_docs_agent=target_framework_docs_agent,
        )
        self._llm_client = instructor.from_litellm(completion)
