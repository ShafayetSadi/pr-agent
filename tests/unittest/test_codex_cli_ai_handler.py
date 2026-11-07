import json
import stat
import sys
import textwrap

import pytest

import pr_agent.algo.ai_handlers.codex_cli_ai_handler as codex_handler
import pr_agent.algo.ai_handlers.litellm_ai_handler as litellm_handler
import pr_agent.algo.token_budget as token_budget
from pr_agent.algo.ai_handlers.codex_cli_ai_handler import (
    CODEX_LOCKDOWN_ARGS,
    CodexCLIAIHandler,
    CodexCLIError,
    is_codex_model,
)
from pr_agent.algo.run_details import get_run_details, init_run_details
from pr_agent.algo.token_budget import FallbackEligibleError, get_max_tokens
from pr_agent.config_security import is_repo_host_only_key

# Stand-in for `codex exec`: records its argv, stdin and environment, then behaves per FAKE_CODEX_MODE.
_FAKE_CLI = textwrap.dedent("""\
    import json, os, pathlib, sys, time
    args = sys.argv[1:]
    record = {"args": args, "stdin": sys.stdin.read(), "cwd": os.getcwd(),
              "env_openai_key": os.environ.get("OPENAI_API_KEY"),
              "env_keep": os.environ.get("FAKE_CODEX_KEEP")}
    pathlib.Path(os.environ["FAKE_CODEX_RECORD"]).write_text(json.dumps(record))
    mode = os.environ.get("FAKE_CODEX_MODE", "ok")
    if mode == "sleep":
        time.sleep(30)
    if mode == "fail":
        print(json.dumps({"type": "turn.failed", "error": {"message": "usage limit reached"}}))
        sys.exit(1)
    out = args[args.index("--output-last-message") + 1]
    if mode == "ok":
        pathlib.Path(out).write_text("## Review\\nLooks good.\\n")
    print(json.dumps({"type": "thread.started", "thread_id": "t"}))
    print(json.dumps({"type": "item.completed", "item": {"type": "agent_message",
                                                          "text": "" if mode == "empty" else "from events"}}))
    print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 40,
                                                          "cache_write_input_tokens": 0, "output_tokens": 7}}))
""")


class _Settings:
    def __init__(self, values):
        self.values = {k.upper(): v for k, v in values.items()}

    def get(self, key, default=None):
        return self.values.get(key.upper(), default)


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    script = tmp_path / "fake_codex.py"
    script.write_text(_FAKE_CLI)
    launcher = tmp_path / "codex"
    launcher.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    record = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_CODEX_RECORD", str(record))
    monkeypatch.setenv("FAKE_CODEX_MODE", "ok")

    def configure(**values):
        settings = _Settings({"codex.cli_path": str(launcher), **values})
        monkeypatch.setattr(codex_handler, "get_settings", lambda: settings)

    configure()
    return configure, lambda: json.loads(record.read_text())


def test_is_codex_model():
    assert is_codex_model("codex/gpt-5.3-codex")
    assert is_codex_model("codex/")
    assert not is_codex_model("gpt-5.3-codex")
    assert not is_codex_model("openai/codex/x")
    assert not is_codex_model(None)


@pytest.mark.asyncio
async def test_runs_locked_down_codex_exec_and_records_usage(fake_cli, monkeypatch):
    configure, read_record = fake_cli
    configure(**{"codex.reasoning_effort": "High"})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("FAKE_CODEX_KEEP", "kept")
    init_run_details()

    response, finish_reason = await CodexCLIAIHandler().chat_completion(
        model="codex/gpt-5.3-codex", system="SYSTEM RULES", user="USER DIFF")

    assert (response, finish_reason) == ("## Review\nLooks good.", "stop")
    record = read_record()
    args = record["args"]
    assert args[0] == "exec"
    for flag in CODEX_LOCKDOWN_ARGS:
        assert flag in args
    assert args[args.index("--model") + 1] == "gpt-5.3-codex"
    assert 'model_reasoning_effort="high"' in args
    assert args[args.index("--cd") + 1] == record["cwd"]
    assert args[-1] == "-"
    assert "SYSTEM RULES" in record["stdin"] and "USER DIFF" in record["stdin"]
    assert record["stdin"].index("SYSTEM RULES") < record["stdin"].index("USER DIFF")
    assert record["env_openai_key"] is None
    assert record["env_keep"] == "kept"

    details = get_run_details()
    assert details.num_ai_calls == 1
    assert (details.prompt_tokens, details.completion_tokens, details.total_tokens) == (100, 7, 107)
    assert details.cache_read_tokens == 40
    assert details.known_cost_call_count == 0


@pytest.mark.asyncio
async def test_bare_prefix_uses_cli_default_model_and_ignores_bad_effort(fake_cli):
    configure, read_record = fake_cli
    configure(**{"codex.reasoning_effort": "max"})

    await CodexCLIAIHandler().chat_completion(model="codex/", system="s", user="u")

    args = read_record()["args"]
    assert "--model" not in args
    assert not any(arg.startswith("model_reasoning_effort") for arg in args)


@pytest.mark.asyncio
async def test_falls_back_to_last_agent_message_when_output_file_missing(fake_cli, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "no_file")
    response, _ = await CodexCLIAIHandler().chat_completion(model="codex/m", system="s", user="u")
    assert response == "from events"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, message", [("fail", "usage limit reached"), ("empty", "empty response")])
async def test_failures_raise_fallback_eligible_error(fake_cli, monkeypatch, mode, message):
    monkeypatch.setenv("FAKE_CODEX_MODE", mode)
    with pytest.raises(CodexCLIError, match=message) as excinfo:
        await CodexCLIAIHandler().chat_completion(model="codex/m", system="s", user="u")
    assert isinstance(excinfo.value, FallbackEligibleError)


@pytest.mark.asyncio
async def test_timeout_kills_cli(fake_cli, monkeypatch):
    configure, _ = fake_cli
    configure(**{"codex.timeout_seconds": "0.5"})
    monkeypatch.setenv("FAKE_CODEX_MODE", "sleep")
    with pytest.raises(CodexCLIError, match="timed out"):
        await CodexCLIAIHandler().chat_completion(model="codex/m", system="s", user="u")


@pytest.mark.asyncio
async def test_missing_cli_reports_setup_hint(tmp_path, monkeypatch):
    settings = _Settings({"codex.cli_path": str(tmp_path / "missing-codex")})
    monkeypatch.setattr(codex_handler, "get_settings", lambda: settings)
    with pytest.raises(CodexCLIError, match="codex login"):
        await CodexCLIAIHandler().chat_completion(model="codex/m", system="s", user="u")


@pytest.mark.parametrize("value", [0, -5, "soon", None])
def test_invalid_timeout_uses_default(monkeypatch, value):
    monkeypatch.setattr(codex_handler, "get_settings", lambda: _Settings({"codex.timeout_seconds": value}))
    assert codex_handler._codex_settings()[1] == codex_handler.DEFAULT_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_litellm_handler_delegates_codex_models(monkeypatch):
    calls = []

    class _FakeCodex:
        async def chat_completion(self, model, system, user, temperature, img_path):
            calls.append((model, system, user))
            return "codex answer", "stop"

    monkeypatch.setattr(litellm_handler, "CodexCLIAIHandler", _FakeCodex)
    handler = litellm_handler.LiteLLMAIHandler.__new__(litellm_handler.LiteLLMAIHandler)

    assert await handler.chat_completion("codex/gpt-5.3-codex", "s", "u") == ("codex answer", "stop")
    assert await handler.chat_completion("codex/", "s2", "u2") == ("codex answer", "stop")
    assert calls == [("codex/gpt-5.3-codex", "s", "u"), ("codex/", "s2", "u2")]


def test_max_tokens_use_the_named_model(monkeypatch):
    settings = type("", (), {
        "config": type("", (), {"custom_model_max_tokens": 0, "max_model_tokens": 0})(),
        "litellm": type("", (), {"custom_llm_provider": ""})(),
    })()
    monkeypatch.setattr(token_budget, "get_settings", lambda: settings)
    assert get_max_tokens("codex/gpt-5.3-codex") == get_max_tokens("gpt-5.3-codex") == 400000


@pytest.mark.parametrize("key", ["cli_path", "timeout_seconds", "reasoning_effort"])
def test_codex_section_is_host_only(key):
    assert is_repo_host_only_key("codex", key)
