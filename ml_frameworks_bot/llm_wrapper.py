from typing import TYPE_CHECKING, Any, Union, get_args

import anthropic
import weave
from instructor import Instructor

if TYPE_CHECKING:
    from anthropic import Anthropic
    from instructor import Instructor
    from openai import OpenAI


OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o1-preview",
    "o1-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]


class LLMClientWrapper(weave.Model):
    model_name: str
    _llm_client: Union["OpenAI", "Anthropic"] = None
    _structured_llm_client: "Instructor" = None

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        import instructor

        if self.model_name in OPENAI_MODELS:
            from openai import OpenAI

            self._llm_client = OpenAI()
            self._structured_llm_client = instructor.from_openai(self._llm_client)
        elif self.model_name in get_args(get_args(anthropic.types.model.Model)[-1]):
            from anthropic import Anthropic

            self._llm_client = Anthropic()
            self._structured_llm_client = instructor.from_anthropic(self._llm_client)
        else:
            raise ValueError(f"{self.model_name} is not supported!")

    @weave.op()
    def predict(self, **kwargs) -> Any:
        if self.model_name in OPENAI_MODELS:
            return (
                self._structured_llm_client.chat.completions.create(
                    model=self.model_name, **kwargs
                )
                if "response_model" in kwargs
                else self._llm_client.chat.completions.create(
                    model=self.model_name, **kwargs
                )
                .choices[0]
                .message.content
            )
        elif self.model_name in get_args(get_args(anthropic.types.model.Model)[-1]):
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = 1024
            if "seed" in kwargs:
                kwargs.pop("seed")
            system_prompt = None
            for idx, message in enumerate(kwargs["messages"]):
                if message["role"] == "system":
                    system_prompt = kwargs["messages"].pop(idx)["content"]
                if system_prompt is not None:
                    kwargs["system"] = system_prompt
            return (
                self._structured_llm_client.chat.completions.create(
                    model=self.model_name, **kwargs
                )
                if "response_model" in kwargs
                else self._llm_client.messages.create(model=self.model_name, **kwargs)
                .content[0]
                .text
            )
