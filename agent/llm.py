"""ChatOpenAI subclass that records the exact LLM payload in Langfuse."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from agent.tracing import span, tracing_enabled


def _serialize_result(result: ChatResult, api: str) -> dict[str, Any]:
    return {
        "format": api,
        "generations": [
            {
                "text": generation.text,
                "message": generation.message.model_dump(mode="json"),
                "generation_info": generation.generation_info,
            }
            for generation in result.generations
        ],
        "llm_output": result.llm_output,
    }


class TracedChatOpenAI(ChatOpenAI):
    """ChatOpenAI plus a child span with the exact request and generated output.

    The Langfuse LangChain callback already creates the ``ChatOpenAI``
    generation span. This subclass adds a nested ``llm_call`` span whose input is
    the real ``_get_request_payload`` dict (Responses or Chat Completions) and
    whose output is the parsed message and token usage.
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not tracing_enabled():
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        api = "responses" if self._use_responses_api(payload) else "chat_completions"
        with span("llm_call", input=payload, metadata={"api": api}) as observation:
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            if observation is not None:
                observation.update(output=_serialize_result(result, api))
            return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if not tracing_enabled():
            yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return

        payload = self._get_request_payload(messages, stop=stop, stream=True, **kwargs)
        api = "responses" if self._use_responses_api(payload) else "chat_completions"
        with span("llm_call", input=payload, metadata={"api": api}) as observation:
            merged: ChatGenerationChunk | None = None
            for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                merged = chunk if merged is None else merged + chunk
                yield chunk
            if observation is not None and merged is not None:
                observation.update(
                    output={
                        "format": api,
                        "message": merged.message.model_dump(mode="json"),
                        "usage": merged.message.usage_metadata,
                    }
                )
