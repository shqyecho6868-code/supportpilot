"""Bridge between the agent and the MCP server.

The gateway is the only place tool calls are executed. It:
  * injects the authenticated user id into MCP request ``_meta``,
  * hides orchestrator-owned parameters (the idempotency key) from the model,
  * declares which tools need human confirmation before they run,
  * screens every result with the prompt-injection guard,
  * records every call as a Langfuse tool observation.
"""

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any

from mcp import Client
from mcp.server.mcpserver import MCPServer

from supportpilot.guard import screen_tool_result
from supportpilot.mcp_server import IDEMPOTENCY_KEY_PARAM, USER_ID_META_KEY
from supportpilot.observability import get_tracer

# Tools that change customer data. The graph pauses for confirmation before running them.
CONFIRMATION_REQUIRED = frozenset({"update_address"})

# Tools the order agent may call. search_policy is called by the knowledge agent itself
# (single-shot retrieval), so the order agent does not see it.
ORDER_AGENT_TOOLS = frozenset({"get_order", "list_my_orders", "update_address", "create_ticket"})

# The single-agent architecture (an evaluation baseline) gives one agent every tool.
SINGLE_AGENT_TOOLS = ORDER_AGENT_TOOLS | {"search_policy"}

# Parameters the orchestrator fills in; the model never sees or sets them.
ORCHESTRATOR_PARAMS: dict[str, frozenset[str]] = {"update_address": frozenset({IDEMPOTENCY_KEY_PARAM})}


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool
    # Injection-guard pattern names that matched the raw result (empty if clean).
    injection_flags: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        try:
            return {"data": json.loads(self.content), "is_error": self.is_error}
        except json.JSONDecodeError:
            return {"data": self.content, "is_error": self.is_error}


def _strip_params(schema: dict[str, Any], hidden: frozenset[str]) -> dict[str, Any]:
    schema = copy.deepcopy(schema)
    schema.pop("title", None)
    for name in hidden:
        schema.get("properties", {}).pop(name, None)
    if "required" in schema:
        schema["required"] = [r for r in schema["required"] if r not in hidden]
    return schema


class ToolGateway:
    def __init__(self, server: MCPServer, *, guard: bool = True):
        self._server = server
        self._guard = guard
        self._model_tools: list[dict[str, Any]] | None = None

    async def model_tools(self, names: frozenset[str] | None = None) -> list[dict[str, Any]]:
        """Tool definitions in Anthropic Messages API format, derived from the MCP server.

        ``names`` restricts the list to those tools (order is kept stable for prompt caching).
        """
        tools = await self._all_model_tools()
        return tools if names is None else [t for t in tools if t["name"] in names]

    async def _all_model_tools(self) -> list[dict[str, Any]]:
        if self._model_tools is None:
            async with Client(self._server) as client:
                listed = await client.list_tools()
            self._model_tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": _strip_params(
                        tool.input_schema, ORCHESTRATOR_PARAMS.get(tool.name, frozenset())
                    ),
                }
                for tool in sorted(listed.tools, key=lambda t: t.name)  # stable order keeps prompt cache warm
            ]
        return self._model_tools

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        user_id: str,
        idempotency_key: str | None = None,
    ) -> ToolResult:
        # Drop anything the model tried to pass for orchestrator-owned params (or a user id).
        hidden = ORCHESTRATOR_PARAMS.get(name, frozenset())
        args = {k: v for k, v in arguments.items() if k not in hidden and k != "user_id"}
        if IDEMPOTENCY_KEY_PARAM in hidden:
            if not idempotency_key:
                raise ValueError(f"{name} requires an idempotency key")
            args[IDEMPOTENCY_KEY_PARAM] = idempotency_key

        with get_tracer().span(f"tool:{name}", as_type="tool", input=args) as obs:
            started = time.monotonic()
            async with Client(self._server) as client:
                result = await client.call_tool(name, args, meta={USER_ID_META_KEY: user_id})
            text = "\n".join(getattr(block, "text", "") for block in result.content)
            flags: tuple[str, ...] = ()
            if self._guard and not result.is_error:
                text, verdict = screen_tool_result(name, text)
                flags = verdict.matches
            obs.update(
                output=text[:8000],
                level="ERROR" if result.is_error else ("WARNING" if flags else None),
                metadata={
                    "is_error": bool(result.is_error),
                    "injection_flags": list(flags),
                    "seconds": round(time.monotonic() - started, 3),
                },
            )
        return ToolResult(content=text, is_error=bool(result.is_error), injection_flags=flags)
