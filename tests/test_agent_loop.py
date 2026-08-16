from __future__ import annotations

import asyncio
import dataclasses

import httpx2
import pytest
from openai import APIConnectionError, RateLimitError

from tests.fake_surface import FakeSurface
from wordlehands.agent.loop import _create_completion_with_retry, run_discovery
from wordlehands.config import settings
from wordlehands.escalation.manager import EscalationManager
from wordlehands.evidence.logger import EvidenceLogger


@pytest.fixture
def evidence(tmp_path):
    return EvidenceLogger(tmp_path / "run")


async def _instant_sleep(_seconds: float) -> None:
    return None


def _connection_error() -> APIConnectionError:
    req = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    return APIConnectionError(request=req)


def _rate_limit_error() -> RateLimitError:
    req = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx2.Response(429, request=req)
    return RateLimitError("rate limited", response=resp, body=None)


class _Completions:
    def __init__(self, exceptions: list[Exception]):
        self._exceptions = exceptions
        self.call_count = 0

    async def create(self, **kwargs):
        self.call_count += 1
        if self._exceptions:
            raise self._exceptions.pop(0)
        return "the real response"


class _Chat:
    def __init__(self, exceptions: list[Exception]):
        self.completions = _Completions(exceptions)


class _FlakyClient:
    """Fails with the given exceptions in order, then succeeds."""

    def __init__(self, exceptions: list[Exception]):
        self.chat = _Chat(exceptions)

    @property
    def call_count(self) -> int:
        return self.chat.completions.call_count


@pytest.mark.asyncio
async def test_retries_transient_errors_and_eventually_succeeds(evidence, monkeypatch):
    monkeypatch.setattr("wordlehands.agent.loop.asyncio.sleep", _instant_sleep)
    client = _FlakyClient([_connection_error(), _rate_limit_error()])

    result = await _create_completion_with_retry(client, evidence, max_attempts=4, model="gpt-4.1", messages=[])

    assert result == "the real response"
    assert client.call_count == 3  # 2 failures + 1 success

    retry_events = [json_line for json_line in evidence.run_dir.joinpath("log.jsonl").read_text().splitlines()]
    assert sum("llm_call_retry" in line for line in retry_events) == 2


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts(evidence, monkeypatch):
    monkeypatch.setattr("wordlehands.agent.loop.asyncio.sleep", _instant_sleep)
    client = _FlakyClient([_connection_error(), _connection_error(), _connection_error()])

    with pytest.raises(APIConnectionError):
        await _create_completion_with_retry(client, evidence, max_attempts=2, model="gpt-4.1", messages=[])

    assert client.call_count == 2  # stopped at max_attempts, did not keep retrying forever


# --- escalation hand-off / resume-into-reasoning -----------------------------


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id_: str, name: str, arguments: str):
        self.id = id_
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, tool_calls: list[_FakeToolCall]):
        self.tool_calls = tool_calls
        self.content = None

    def model_dump(self, exclude_none: bool = True) -> dict:
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ],
        }


class _FakeChoice:
    def __init__(self, message: _FakeMessage):
        self.message = message


class _FakeChatResponse:
    def __init__(self, message: _FakeMessage):
        self.choices = [_FakeChoice(message)]


class _ScriptedCompletions:
    def __init__(self, responses: list[_FakeChatResponse]):
        self._responses = list(responses)

    async def create(self, **kwargs):
        return self._responses.pop(0)


class _ScriptedChat:
    def __init__(self, responses: list[_FakeChatResponse]):
        self.completions = _ScriptedCompletions(responses)


class _ScriptedClient:
    """A fake OpenAI client that returns pre-scripted responses in order,
    standing in for the model's actual decisions across two loop turns: one
    before an escalation, one after a human resumes control."""

    def __init__(self, responses: list[_FakeChatResponse]):
        self.chat = _ScriptedChat(responses)


@pytest.mark.asyncio
async def test_escalation_hands_off_to_human_and_resumes_llm_reasoning(evidence, monkeypatch):
    surface = FakeSurface(mode="success")
    escalation_manager = EscalationManager(surface, evidence)

    escalate_call = _FakeToolCall("call_1", "escalate", '{"reason": "test stuck"}')
    finish_call = _FakeToolCall("call_2", "finish_goal", '{"outcome": "won", "summary": "done"}')
    client = _ScriptedClient(
        [_FakeChatResponse(_FakeMessage([escalate_call])), _FakeChatResponse(_FakeMessage([finish_call]))]
    )

    fake_settings = dataclasses.replace(settings, openai_api_key="fake-key-for-test")
    monkeypatch.setattr("wordlehands.agent.loop.settings", fake_settings)
    monkeypatch.setattr("wordlehands.agent.loop.AsyncOpenAI", lambda api_key: client)

    escalate_events: list[tuple[str, str]] = []

    async def _act_as_human_and_resume():
        while escalation_manager.control_owner != "human":
            await asyncio.sleep(0.01)
        await escalation_manager.record_human_action("type_text shard", True, "typed 5 chars")
        await escalation_manager.resume()

    resume_task = asyncio.create_task(_act_as_human_and_resume())

    tool_runner = await run_discovery(
        "goal",
        "https://hellowordl.net/",
        surface,
        evidence,
        max_steps=5,
        escalation_manager=escalation_manager,
        on_escalate=lambda reason, url: escalate_events.append((reason, url)),
        operator_url="http://127.0.0.1:9999",
    )
    await resume_task

    # The loop didn't just stop at the escalation — it kept going afterward
    # and let the model finish the goal on the second scripted turn.
    assert tool_runner.finished is True
    assert tool_runner.finish_result == {"outcome": "won", "summary": "done"}
    assert escalate_events == [("test stuck", "http://127.0.0.1:9999")]
    assert len(escalation_manager.human_actions) == 1
    assert escalation_manager.control_owner == "automation"
