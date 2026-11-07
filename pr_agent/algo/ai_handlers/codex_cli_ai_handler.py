"""Run chat completions through the local Codex CLI so they bill against its ChatGPT sign-in."""

import asyncio
import contextlib
import json
import os
import tempfile
from pathlib import Path

from pr_agent.algo.ai_handlers.base_ai_handler import BaseAiHandler
from pr_agent.algo.run_details import record_ai_call
from pr_agent.algo.token_budget import FallbackEligibleError
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

CODEX_MODEL_PREFIX = "codex/"
DEFAULT_CLI_PATH = "codex"
DEFAULT_TIMEOUT_SECONDS = 600
CODEX_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh"})
_ERROR_DETAIL_CHARS = 2000

# The prompt carries untrusted pull-request content, so Codex runs as a plain text model:
# no shell, MCP servers, apps, plugins, sub-agents, browser, hooks or web search, no user
# config.toml or execpolicy rules, and nothing persisted. Unknown -c keys only warn, so
# feature names an older or newer CLI lacks do not break the call.
CODEX_LOCKDOWN_ARGS = (
    "--ignore-user-config",
    "--ignore-rules",
    "--ephemeral",
    "--skip-git-repo-check",
    "--sandbox", "read-only",
    "-c", "features.shell_tool=false",
    "-c", "features.unified_exec=false",
    "-c", "features.apps=false",
    "-c", "features.plugins=false",
    "-c", "features.multi_agent=false",
    "-c", "features.browser_use=false",
    "-c", "features.computer_use=false",
    "-c", "features.image_generation=false",
    "-c", "features.hooks=false",
    "-c", 'web_search="disabled"',
)

# Credentials PR-Agent may export for other handlers. Leaving them in the child environment
# could switch Codex from the ChatGPT sign-in to metered API billing or another endpoint.
_STRIPPED_ENV_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE")

_PROMPT_PREAMBLE = (
    "You are being used as a plain text-completion model. You have no tools and no workspace: "
    "answer directly from the content below, in exactly the format the instructions request."
)


class CodexCLIError(FallbackEligibleError):
    """Report a failed Codex CLI run; fallback models may still answer."""


def is_codex_model(model) -> bool:
    """Return whether a model name selects the Codex CLI handler."""
    return isinstance(model, str) and model.startswith(CODEX_MODEL_PREFIX)


def _codex_settings():
    settings = get_settings()
    cli_path = str(settings.get("CODEX.CLI_PATH", "") or "").strip() or DEFAULT_CLI_PATH
    try:
        timeout = float(settings.get("CODEX.TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        timeout = 0
    if timeout <= 0:
        get_logger().warning(f"Invalid codex.timeout_seconds; using {DEFAULT_TIMEOUT_SECONDS}")
        timeout = DEFAULT_TIMEOUT_SECONDS
    effort = str(settings.get("CODEX.REASONING_EFFORT", "") or "").strip().lower()
    if effort and effort not in CODEX_REASONING_EFFORTS:
        get_logger().warning(f"Ignoring unsupported codex.reasoning_effort {effort!r}")
        effort = ""
    return cli_path, timeout, effort


def _render_prompt(messages: list[dict]) -> str:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return f"{_PROMPT_PREAMBLE}\n\n<instructions>\n{system}\n</instructions>\n\n{user}"


def _parse_events(stdout: str):
    """Return (last agent message, usage dict, error message) from `codex exec --json` output."""
    message, usage, error = "", None, ""
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        item = event.get("item")
        if kind == "item.completed" and isinstance(item, dict) and item.get("type") == "agent_message":
            message = item.get("text") or message
        elif kind == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
        elif kind == "turn.failed" and isinstance(event.get("error"), dict):
            error = str(event["error"].get("message") or error)
        elif kind == "error":
            error = str(event.get("message") or error)
    return message, usage, error


def _to_usage(codex_usage: dict | None) -> dict | None:
    if not codex_usage:
        return None
    prompt_tokens = int(codex_usage.get("input_tokens") or 0)
    completion_tokens = int(codex_usage.get("output_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cache_read_input_tokens": int(codex_usage.get("cached_input_tokens") or 0),
        "cache_creation_input_tokens": int(codex_usage.get("cache_write_input_tokens") or 0),
    }


class CodexCLIAIHandler(BaseAiHandler):
    """Send prompts to `codex exec`, which authenticates with the host's `codex login` session.

    Select it with a model named "codex/<model>", for example "codex/gpt-5.3-codex"; "codex/"
    alone uses the CLI's default model. Temperature and images are not supported.
    """

    def __init__(self):
        super().__init__()

    @property
    def deployment_id(self):
        return None

    async def chat_completion(self, model: str, system: str, user: str, temperature: float = 0.2, img_path: str = None):
        if img_path:
            get_logger().warning("The Codex CLI handler does not support image inputs; ignoring the image")
        cli_path, timeout, effort = _codex_settings()
        codex_model = model.removeprefix(CODEX_MODEL_PREFIX).strip()
        prompt = _render_prompt(self.build_request_messages(model, system, user))
        env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV_VARS}

        with tempfile.TemporaryDirectory(prefix="pr-agent-codex-") as workdir:
            output_path = Path(workdir) / "last_message.txt"
            args = [cli_path, "exec", *CODEX_LOCKDOWN_ARGS, "--json", "--cd", workdir,
                    "--output-last-message", str(output_path)]
            if codex_model:
                args += ["--model", codex_model]
            if effort:
                args += ["-c", f'model_reasoning_effort="{effort}"']
            args.append("-")

            stdout, stderr, returncode = await self._run(args, prompt, workdir, env, timeout, cli_path)
            message, codex_usage, event_error = _parse_events(stdout)
            if returncode != 0:
                detail = (event_error or stderr.strip() or stdout.strip())[-_ERROR_DETAIL_CHARS:]
                raise CodexCLIError(f"Codex CLI exited with status {returncode}: {detail}")
            try:
                content = output_path.read_text(encoding="utf-8")
            except OSError:
                content = ""
        content = content.strip() or message.strip()
        if not content:
            raise CodexCLIError(f"Codex CLI returned an empty response{': ' + event_error if event_error else ''}")

        usage = _to_usage(codex_usage)
        get_logger().info("AI response", response=content, model=model, finish_reason="stop", usage=usage)
        # Subscription calls have no per-token price, so they count as unpriced calls.
        record_ai_call(usage, model=model)
        return content, "stop"

    @staticmethod
    async def _run(args, prompt, workdir, env, timeout, cli_path):
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, cwd=workdir, env=env,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except OSError as e:
            raise CodexCLIError(
                f"Could not start the Codex CLI ({cli_path!r}): {e}. Install it, run `codex login`, "
                "or set codex.cli_path."
            ) from e
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), timeout)
        except (TimeoutError, asyncio.CancelledError) as e:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            if isinstance(e, asyncio.CancelledError):
                raise
            raise CodexCLIError(f"Codex CLI timed out after {timeout:g} seconds") from e
        return stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace"), proc.returncode
