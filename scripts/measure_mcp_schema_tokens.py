#!/usr/bin/env python3
"""Probe an MCP stdio server for tools/list and count schema bytes and o200k_base tokens."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any


def count_tools_schema(tools: list[dict[str, Any]], encoding: str = "o200k_base") -> dict[str, Any]:
    """Return tool count, serialized byte size, and o200k_base token count (None if tiktoken absent)."""
    payload = json.dumps(tools)
    size_bytes = len(payload.encode("utf-8"))
    tokens: int | None = None
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding(encoding)
        tokens = len(enc.encode(payload))
    except ImportError:
        tokens = None
    return {
        "tools": len(tools),
        "bytes": size_bytes,
        "encoding": encoding,
        "tokens": tokens,
    }


def _read_response(proc: subprocess.Popen[str], target_id: int) -> dict[str, Any]:
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"MCP server closed stdout before answering id={target_id}")
        text = line.strip()
        if not text:
            continue
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(msg, dict) and msg.get("id") == target_id:
            if "error" in msg:
                raise RuntimeError(f"MCP error for id={target_id}: {msg['error']}")
            return msg


def probe_mcp_server(command: list[str], env: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Run MCP initialize + notifications/initialized + tools/list over stdio and return tools array."""
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=env,
    )
    try:
        assert proc.stdin is not None
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-schema-probe", "version": "1.0"},
            },
        }
        proc.stdin.write(json.dumps(init_req) + "\n")
        proc.stdin.flush()
        _read_response(proc, 1)

        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n")
        proc.stdin.flush()

        resp = _read_response(proc, 2)
        return list(resp.get("result", {}).get("tools", []))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure MCP tools/list context token overhead.")
    parser.add_argument("--bytes-only", action="store_true", help="Skip requiring tiktoken")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON output")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="MCP stdio command, e.g. -- deja mcp",
    )
    args = parser.parse_args(argv)

    cmd = [part for part in args.command if part != "--"] if args.command else []
    if not cmd and os.environ.get("SERVER"):
        cmd = json.loads(os.environ["SERVER"])
    if not cmd:
        parser.error("Provide an MCP server command (e.g. -- deja mcp) or SERVER='[...]'")

    tools = probe_mcp_server(cmd)
    stats = count_tools_schema(tools)
    if stats["tokens"] is None and not args.bytes_only:
        print(
            "error: tiktoken is not installed; install tiktoken for o200k_base counts or pass --bytes-only",
            file=sys.stderr,
        )
        return 2

    if args.as_json:
        print(json.dumps(stats))
    elif stats["tokens"] is None:
        print(f"tools={stats['tools']} bytes={stats['bytes']}")
    else:
        print(f"tools={stats['tools']} bytes={stats['bytes']} {stats['encoding']}_tokens={stats['tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
