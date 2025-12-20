import os
from abc import abstractmethod

import backoff
import litellm
from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
)


def engine_factory(model, **kwargs):
    if model in ["gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview", "gpt-4-turbo"]:
        assert (
            os.getenv("OPENAI_API_KEY") is not None
        ), "must set OPENAI_API_KEY in the environment"
        return OpenAIEngine(model=model, **kwargs)
    elif model in [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-2.0-flash-exp",
    ]:
        assert (
            os.getenv("GEMINI_API_KEY") is not None
        ), "must set GEMINI_API_KEY in the environment"
        model = f"gemini/{model}"
        return GeminiEngine(model=model, **kwargs)
    elif model in ["claude-3-5-sonnet-latest", "claude-3-opus-20240229"]:
        assert (
            os.getenv("ANTHROPIC_API_KEY") is not None
        ), "must set ANTHROPIC_API_KEY in the environment"
        model = f"anthropic/{model}"
        return ClaudeEngine(model=model, **kwargs)
    else:
        Exception(f"Unsupported model: {model}")


class Engine:
    def __init__(self, rate_limit=-1, model=None, temperature=0):
        self.rate_limit = rate_limit
        self.model = model
        self.temperature = temperature
        print(f"Initializing model {self.model}")

    @abstractmethod
    def generate(self):
        pass


class GeminiEngine(Engine):
    def __init__(self, **kwargs) -> None:
        """
        Init a Gemini engine
        """
        super().__init__(**kwargs)

    def generate(self, conversation, stop=None, max_tokens=4096) -> str:
        response = litellm.completion(
            model=self.model,
            messages=conversation,
            temperature=self.temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        return response.choices[0].message.content


class ClaudeEngine(Engine):
    def __init__(self, **kwargs) -> None:
        """
        Init a Claude engine
        """
        super().__init__(**kwargs)

    def generate(self, conversation, stop=None, max_tokens=4096) -> str:
        if stop is not None:
            # Anthropic won't support '\n' as a stop token
            stop = [s for s in stop if s != "\n"]
            response = litellm.completion(
                model=self.model,
                messages=conversation,
                temperature=self.temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
        else:
            response = litellm.completion(
                model=self.model,
                messages=conversation,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
        return response.choices[0].message.content


class OpenAIEngine(Engine):
    def __init__(self, **kwargs) -> None:
        """
        Init an OpenAI GPT engine
        """
        super().__init__(**kwargs)
        # self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
    )
    def generate(self, conversation, stop=None, max_tokens=4096) -> str:
        if stop is not None:
            response = litellm.completion(
                model=self.model,
                messages=conversation,
                temperature=self.temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
        else:
            response = litellm.completion(
                model=self.model,
                messages=conversation,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
        return response.choices[0].message.content

    def generate_format(
        self, conversation, response_format, max_tokens=4096, frequency_penalty=0
    ):
        # TODO: probably will be deprecated.
        response = litellm.completion(
            model=self.model,
            messages=conversation,
            temperature=self.temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            frequency_penalty=frequency_penalty,
        )

        return response
