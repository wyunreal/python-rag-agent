"""ReAct agent built on LangChain's ``create_agent``.

``create_agent`` is the prebuilt ReAct loop in LangChain 1.x: it decides when to
call a tool, feeds the result back to the model and repeats until it can answer.
We only provide the model, the retrieval tool and the legal system prompt.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from agent.config import Settings
from agent.llm import TracedChatOpenAI
from agent.prompts import build_system_prompt
from agent.tools import make_tools


def build_model(settings: Settings) -> ChatOpenAI:
    """Build the chat model for the selected OpenAI-compatible API.

    ``responses`` uses ``POST /v1/responses`` in stateless mode: the full
    message history travels in every request and no ``previous_response_id`` is
    sent. ``chat_completions`` keeps the classic ``/v1/chat/completions`` path.
    """
    use_responses = settings.api_mode == "responses"
    return TracedChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.require_api_key(),
        temperature=0,
        max_retries=2,
        timeout=180,
        use_responses_api=use_responses,
        use_previous_response_id=False,
        include=["reasoning.encrypted_content"] if use_responses else None,
    )


def build_agent(settings: Settings | None = None) -> CompiledStateGraph:
    settings = settings or Settings.from_env()
    return create_agent(
        model=build_model(settings),
        tools=make_tools(settings),
        system_prompt=build_system_prompt(),
    )
