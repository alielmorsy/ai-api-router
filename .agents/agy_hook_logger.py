"""Capture safe Antigravity tool metadata for the wrapper's server log."""

import json
import os
import sys
from pathlib import Path


def main() -> None:
    event = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        payload = json.load(sys.stdin)
        tool = payload.get("toolCall") or {}
        outcome = (
            "started" if event == "session" else
            "invoking" if event == "invocation" else
            "attempted" if event == "pre" else
            "error" if payload.get("error") else "completed"
        )
        summary = {
            "event": event,
            "tool": tool.get("name"),
            "outcome": outcome,
            "conversation_id": payload.get("conversationId"),
            "model": payload.get("modelName"),
            "step": payload.get("stepIdx"),
        }
        log_path = os.environ.get("AGY_WRAPPER_HOOK_LOG")
        if log_path:
            with Path(log_path).open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(summary) + "\n")
    except Exception as exc:
        print(f"agy wrapper hook logging failed: {exc}", file=sys.stderr)
    print(json.dumps({"decision": "allow"} if event == "pre" else {}))


if __name__ == "__main__":
    main()
