"""OpenAI function-call exchange over Antigravity's structured text output."""

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from models import ChatCompletionRequest


TOOL_RESULT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        },
    },
    "required": ["content", "tool_calls"],
}


def build_tool_prompt(
    conversation: str, system_prompt: Optional[str], request: ChatCompletionRequest
) -> str:
    tools = [tool.model_dump(exclude_none=True) for tool in request.tools or []]
    choice = request.tool_choice or ("auto" if tools else "none")
    instruction = (
        "You are the model in an OpenAI Chat Completions tool exchange. "
        "The tools below belong to the calling program. Do not execute them yourself. "
        "Return a JSON object with exactly two fields: content (string) and "
        "tool_calls (array of objects with name and arguments object). "
        "To request a tool, put its name and arguments in tool_calls and leave content empty. "
        "The calling program will execute it and send a tool result in a later request. "
        "If no tool is needed, return an empty tool_calls array and put your answer in content. "
        f"Tool choice: {json.dumps(choice)}. "
        f"Parallel tool calls allowed: {request.parallel_tool_calls}. "
        f"Available program tools: {json.dumps(tools)}."
    )
    return f"{instruction}\n\nSystem instructions: {system_prompt or ''}\n\nConversation:\n{conversation}"


def parse_tool_result(
    output: Dict[str, Any], request: ChatCompletionRequest
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    raw_calls = output.get("tool_calls")
    content = output.get("content")
    if not isinstance(raw_calls, list) or not isinstance(content, str):
        raise ValueError("Antigravity returned an invalid tool response")

    if not raw_calls:
        try:
            nested = json.loads(content)
        except json.JSONDecodeError:
            nested = None
        if (
            isinstance(nested, dict)
            and set(nested) == {"content", "tool_calls"}
            and isinstance(nested["content"], str)
            and isinstance(nested["tool_calls"], list)
        ):
            content = nested["content"]
            raw_calls = nested["tool_calls"]

    allowed = {tool.function.name for tool in request.tools or []}
    choice = request.tool_choice or ("auto" if allowed else "none")
    if choice == "none" and raw_calls:
        raise ValueError("Antigravity returned tool calls when tool_choice is none")
    if choice == "required" and not raw_calls:
        raise ValueError("Antigravity did not return a required tool call")
    if not request.parallel_tool_calls and len(raw_calls) > 1:
        raise ValueError("Antigravity returned parallel tool calls when disabled")

    forced_name = None
    if isinstance(choice, dict):
        forced_name = choice.get("function", {}).get("name")
        if not forced_name or forced_name not in allowed:
            raise ValueError("tool_choice names a function that was not offered")
        if len(raw_calls) != 1:
            raise ValueError("Antigravity did not return the forced function call")

    calls = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            raise ValueError("Antigravity returned an invalid tool call")
        name = raw.get("name")
        arguments = raw.get("arguments")
        if name not in allowed:
            raise ValueError(f"Antigravity returned a tool that was not offered: {name}")
        if forced_name and name != forced_name:
            raise ValueError("Antigravity returned a different function than tool_choice")
        if not isinstance(arguments, dict):
            raise ValueError("Antigravity returned non-object tool arguments")
        calls.append({
            "id": f"call_{uuid.uuid4().hex}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, separators=(",", ":"))},
        })
    return (None if calls else content), calls
