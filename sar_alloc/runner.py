"""Real API client. Credentials come only from the process environment."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
REASONING_EFFORTS = ("none", "low", "medium", "high")

def _env(name: str) -> str:
    return os.getenv(name, "").strip()


class LLMClientError(RuntimeError):
    """Raised when the real LLM client cannot be configured or called."""


@dataclass
class OpenAICompatClient:
    base_url: str
    api_key: str
    model: str
    reasoning_effort: str = "none"
    use_system_proxy: bool = False
    _client: Any = field(default=None, init=False, repr=False)
    _requests: int = field(default=0, init=False, repr=False)
    _prompt_tokens: int = field(default=0, init=False, repr=False)
    _completion_tokens: int = field(default=0, init=False, repr=False)
    _total_tokens: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("Missing LLM_BASE_URL.")
        if not self.api_key:
            raise ValueError("Missing LLM_API_KEY.")
        if not self.model:
            raise ValueError("Missing model.")
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                "reasoning_effort must be one of: "
                + ", ".join(REASONING_EFFORTS)
            )
        try:
            from openai import DefaultHttpxClient, OpenAI  # type: ignore
        except ImportError as exc:
            raise LLMClientError("The 'openai' package is required for real LLM calls.") from exc
        # 重试由实验入口统一管理。
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=0,
            http_client=DefaultHttpxClient(
                # 正式默认值直接连接环境变量指定的 API。
                trust_env=bool(self.use_system_proxy),
            ),
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_sec: float = 60.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        for index, message in enumerate(messages):
            if not isinstance(message.get("content"), str):
                raise LLMClientError(f"LLM request message[{index}].content must be a string.")
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout_sec,
        }
        kwargs["extra_body"] = {
            "thinking": {
                "type": (
                    "disabled" if self.reasoning_effort == "none" else "enabled"
                )
            }
        }
        if self.reasoning_effort != "none":
            kwargs["reasoning_effort"] = self.reasoning_effort
        if extra is not None:
            kwargs.update(extra)
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMClientError(f"LLM request failed: {type(exc).__name__}; status={getattr(exc, 'status_code', None)}") from exc
        self._requests += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self._completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            self._total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
        if not response.choices:
            raise LLMClientError("LLM response has no choices")
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise LLMClientError(f"LLM response content is not a string. model={self.model}, content_type={type(content).__name__}")
        return content

    def usage_summary(self) -> Dict[str, int]:
        return {
            "requests": self._requests, "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens, "total_tokens": self._total_tokens,
        }


def build_llm_client(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str,
    reasoning_effort: str = "none",
    use_system_proxy: Optional[bool] = None,
) -> OpenAICompatClient:
    final_base_url = base_url.strip() if isinstance(base_url, str) else _env("LLM_BASE_URL")
    final_api_key = api_key.strip() if isinstance(api_key, str) else _env("LLM_API_KEY")
    final_model = str(model).strip()
    final_reasoning_effort = str(reasoning_effort).strip().lower() or "none"
    proxy_setting = _env("LLM_USE_SYSTEM_PROXY").strip().lower()
    final_use_system_proxy = (
        bool(use_system_proxy)
        if use_system_proxy is not None
        else proxy_setting in {"1", "true", "yes", "on"}
    )
    return OpenAICompatClient(
        base_url=final_base_url,
        api_key=final_api_key,
        model=final_model,
        reasoning_effort=final_reasoning_effort,
        use_system_proxy=final_use_system_proxy,
    )
