from dataclasses import dataclass, field
import json
import logging
import os
import re
from typing import Any
import httpx

from app.core.config import settings

logger = logging.getLogger("growtube.llm")


@dataclass
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    extra_content: dict[str, Any] | None = None
    raw_tool_call: dict[str, Any] | None = None


@dataclass
class LLMToolResponse:
    thought: str = ""
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    raw_content: str = ""
    tokens_used: int = 0
    error: str | None = None
    error_type: str | None = None  # "connection_error", "not_found", "timeout", "server_error"
    raw_message: dict[str, Any] | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None


class OllamaLLMClient:
    """Client for local Ollama instance running Qwen (qwen2.5:3b) with structured tool calling."""

    def __init__(self):
        self.provider = "ollama"
        self.base_url = (getattr(settings, "OLLAMA_BASE_URL", None) or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = (getattr(settings, "OLLAMA_MODEL", None) or getattr(settings, "LLM_MODEL", None) or os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")).strip()

    @property
    def is_configured(self) -> bool:
        """Ollama is configured whenever base_url and model are set."""
        return bool(self.base_url and self.model)

    def check_model_status(self) -> tuple[bool, str]:
        """Check whether local Ollama is reachable and the requested model is pulled."""
        try:
            with httpx.Client(timeout=4.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                if resp.status_code != 200:
                    return False, f"Ollama returned HTTP status {resp.status_code}"
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                
                # Check for exact name or prefix (e.g. qwen2.5:3b or qwen2.5:latest)
                model_base = self.model.split(":")[0]
                has_model = any(self.model == m or m.startswith(f"{model_base}:") for m in models)
                if has_model:
                    return True, f"Model '{self.model}' is installed and ready on Ollama."
                return False, f"Model '{self.model}' is not installed in Ollama. Available: {models}. Run: 'ollama pull {self.model}'."
        except httpx.ConnectError:
            return False, f"Cannot connect to local Ollama at {self.base_url}. Ensure Ollama is running ('ollama serve' or 'ollama run {self.model}')."
        except httpx.TimeoutException:
            return False, f"Connection to Ollama at {self.base_url} timed out."
        except Exception as e:
            return False, f"Error inspecting Ollama: {e}"

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> str | None:
        """Execute chat completion via local Ollama OpenAI-compatible chat API."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices and "message" in choices[0]:
                        return choices[0]["message"].get("content", "").strip()
                else:
                    logger.warning("Ollama returned status %s: %s", resp.status_code, resp.text[:200])
                    return None
        except httpx.ConnectError:
            logger.warning("Cannot connect to local Ollama server at %s.", self.base_url)
            return None
        except httpx.TimeoutException:
            logger.warning("Ollama request timed out after 300s for model %s.", self.model)
            return None
        except Exception as e:
            logger.warning("Ollama LLM request failed: %s", e)
            return None

    def chat_completion_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> LLMToolResponse:
        """Execute chat completion with tool calling via Ollama's OpenAI-compatible endpoint."""
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
                if resp.status_code != 200:
                    err_msg = f"Ollama tools endpoint returned status {resp.status_code}: {resp.text[:200]}"
                    logger.warning(err_msg)
                    return LLMToolResponse(
                        error=err_msg,
                        error_type="server_error" if resp.status_code >= 500 else "client_error",
                    )

                data = resp.json()
                usage = data.get("usage", {}).get("total_tokens", 0)
                choices = data.get("choices", [])
                if not choices:
                    return LLMToolResponse(error="No choices returned from Ollama.", error_type="empty_response")

                msg = choices[0].get("message", {})
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning") or msg.get("thinking") or ""
                finish_reason = choices[0].get("finish_reason")

                tool_calls: list[LLMToolCall] = []

                # 1. Native Ollama tool calls
                raw_tool_calls = msg.get("tool_calls", [])
                for tc in raw_tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            args_dict = json.loads(raw_args)
                        except Exception:
                            args_dict = {}
                    else:
                        args_dict = raw_args or {}

                    tool_calls.append(
                        LLMToolCall(
                            id=tc.get("id", f"call_{len(tool_calls)+1}"),
                            name=name,
                            arguments=args_dict,
                            extra_content=tc.get("extra_content"),
                            raw_tool_call=tc,
                        )
                    )

                # 2. Text-based tool call extraction if model produced raw JSON instead of native call
                if not tool_calls and content:
                    extracted = self._extract_tool_calls_from_text(content)
                    if extracted:
                        tool_calls.extend(extracted)
                        msg["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments),
                                },
                            }
                            for tc in extracted
                        ]

                return LLMToolResponse(
                    thought=reasoning or content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    raw_content=content,
                    tokens_used=usage,
                    raw_message=msg,
                )
        except httpx.ConnectError:
            err_msg = (
                f"Cannot connect to local Ollama server at {self.base_url}. "
                f"Please ensure Ollama is running ('ollama serve' or 'ollama run {self.model}')."
            )
            logger.warning(err_msg)
            return LLMToolResponse(error=err_msg, error_type="connection_error")
        except httpx.TimeoutException:
            err_msg = f"Local Ollama generation timed out after 300s for model '{self.model}'."
            logger.warning(err_msg)
            return LLMToolResponse(error=err_msg, error_type="timeout")
        except Exception as e:
            logger.warning(f"Error in chat_completion_with_tools: {e}")
            return LLMToolResponse(
                error=f"Error communicating with Ollama: {e!s}",
                error_type="unknown_error",
            )

    def _extract_tool_calls_from_text(self, text: str) -> list[LLMToolCall]:
        """Extract tool call intentions from LLM text responses formatted in JSON."""
        tool_calls = []

        # Find all json code blocks
        blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        candidates = blocks if blocks else [text]

        for cand in candidates:
            cand = cand.strip()
            start = cand.find("{")
            end = cand.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = cand[start : end + 1]
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict):
                        # Case 1: {"name": "...", "arguments": {...}}
                        if "name" in data and ("arguments" in data or "args" in data):
                            tool_calls.append(
                                LLMToolCall(
                                    id=f"text_call_{len(tool_calls)+1}",
                                    name=data["name"],
                                    arguments=data.get("arguments") or data.get("args") or {},
                                )
                            )
                        # Case 2: {"action": "tool_call", "tool": "...", "arguments": {...}}
                        elif data.get("action") == "tool_call" and ("tool" in data or "name" in data):
                            tool_name = data.get("tool") or data.get("name")
                            tool_calls.append(
                                LLMToolCall(
                                    id=f"text_call_{len(tool_calls)+1}",
                                    name=tool_name,
                                    arguments=data.get("arguments", {}),
                                )
                            )
                except Exception:
                    pass

        return tool_calls


# Backwards compatibility alias
OpenRouterLLMClient = OllamaLLMClient
