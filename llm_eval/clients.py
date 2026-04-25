from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI

from .settings import FrameworkConfig


@dataclass(frozen=True)
class GenerationResult:
    content: str
    usage: dict[str, int]
    error: str | None = None
    fatal: bool = False
    http_status_code: int | None = None


class OpenAICompatibleClient:
    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.model.api_key,
            base_url=config.model.base_url,
            timeout=config.model.timeout_seconds,
        )

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        last_error = "Unknown generation error"
        last_status_code: int | None = None
        for attempt in range(3):
            try:
                kwargs = {
                    "model": self.config.model.model_name,
                    "messages": messages,
                }
                if self.config.run.thinking_enabled and self.config.run.reasoning_effort:
                    kwargs["reasoning_effort"] = self.config.run.reasoning_effort
                if self.config.run.thinking_enabled:
                    kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

                raw_response = self.client.chat.completions.with_raw_response.create(**kwargs)
                response = raw_response.parse()
                content = (response.choices[0].message.content or "").strip()
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
                if content:
                    return GenerationResult(content=content, usage=usage, http_status_code=raw_response.status_code)
                last_error = "Model returned an empty completion"
            except Exception as exc:
                last_error = str(exc).replace("\n", " ")
                last_status_code = getattr(exc, "status_code", None)
                if self._is_fatal_error(last_error):
                    return GenerationResult(content="", usage=_empty_usage(), error=last_error, fatal=True, http_status_code=last_status_code)
                time.sleep(1 + attempt)

        return GenerationResult(content="", usage=_empty_usage(), error=last_error, fatal=False, http_status_code=last_status_code)

    def _is_fatal_error(self, message: str) -> bool:
        lowered = message.lower()
        fatal_markers = ("invalid api key", "authentication", "unauthorized", "incorrect api key")
        return any(marker in lowered for marker in fatal_markers)


def _empty_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
