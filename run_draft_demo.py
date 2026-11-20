"""Real-HTTP trial run of every draft-decoding scenario (no Docker required).

Mirrors run_demo.py for the speculative decoding path: node-b proposes draft
batches of up to 4 tokens, node-a verifies each batch, and this script prints
the per-batch verification record (how many tokens were appended after each
batch), the fallback/incomplete events, and the equivalence check against the
healthy primary-only baseline.

Usage:  python run_draft_demo.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time

import httpx
import uvicorn

from common.deterministic import expected_token_ids
from gateway.app import GatewayConfig, create_app as create_gateway
from node.app import NodeConfig, create_app as create_node

PROMPT = "请写一段关于秋天的短文"
MAX_TOKENS = 12
PORT_A, PORT_B, PORT_G = 8911, 8912, 8918
READ_TIMEOUT = 0.5
EMIT_INTERVAL = 0.02

DRAFT_SCENARIOS = [
    "draft_all_match",
    "draft_replace",
    "draft_timeout",
    "draft_error",
    "draft_primary_fail",
    "draft_version_mismatch",
]


def serve(app, port):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server


async def consume(client: httpx.AsyncClient, scenario: str):
    events: list[tuple[str, dict]] = []
    async with client.stream(
        "POST", f"http://127.0.0.1:{PORT_G}/api/chat",
        json={"prompt": PROMPT, "max_tokens": MAX_TOKENS,
              "scenario": scenario, "draft": True},
    ) as resp:
        event = None
        data: list[str] = []
        async for raw in resp.aiter_lines():
            line = raw.rstrip("\r")
            if line == "":
                if event is not None or data:
                    events.append((event or "message", json.loads("\n".join(data))))
                event, data = None, []
            elif line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
    return events


def report(scenario: str, events, healthy_ids):
    print(f"\n{'=' * 78}\n草稿场景: {scenario}\n{'=' * 78}")
    tokens = [e for n, e in events if n == "token"]
    for name, e in events:
        if name == "meta":
            print(f"[meta] request={e['request_id'][:8]}… 上限={e['max_tokens']} "
                  f"每批≤{e['draft_batch_size']}")
        elif name == "draft_batch":
            kind = "整批匹配" if not e["replaced"] else "批内替换"
            print(f"[batch {e['batch']}] 起始 index={e['start_index']} 草稿 {e['draft_count']} 个，"
                  f"{kind}：匹配 {e['matched']}、替换 {e['replaced']}、未采纳 {e['rejected']}"
                  f"（丢弃 {e['discarded']}）→ 本批追加 {e['appended']} 个，已确认 {e['confirmed_count']}")
        elif name == "draft_fallback":
            print(f"[draft_fallback] reason={e['reason']}，已确认 {e['accepted_count']} 个，"
                  f"首节点从已确认位置续写")
        elif name == "incomplete":
            print(f"[incomplete] reason={e['reason']} 保留 {e['accepted_count']} 个词元")
    by_node: dict[str, list[str]] = {}
    for t in tokens:
        by_node.setdefault(t["node"], []).append(t["text"])
    for node, texts in by_node.items():
        print(f"[tokens:{node}] ({len(texts)}) {' '.join(texts)}")
    ids = [t["id"] for t in tokens]
    final = "done" in {n for n, _ in events}
    if final:
        ok = ids == healthy_ids
        print(f"[核对] 完整回复词元数={len(ids)}/{MAX_TOKENS}，与首节点独立生成逐词元一致: {ok}")
        assert ok, "draft result diverged from healthy baseline"
    else:
        n = len(ids)
        ok = ids == healthy_ids[:n]
        print(f"[核对] 未完成，保留前缀 {n}/{MAX_TOKENS} 个词元，与基准同位置前缀一致: {ok}")
        assert ok
    return tokens


async def main():
    node_a = create_node(NodeConfig("node-a", emit_interval=EMIT_INTERVAL))
    node_b = create_node(NodeConfig("node-b", emit_interval=EMIT_INTERVAL))
    gateway = create_gateway(GatewayConfig(
        node_a_url=f"http://127.0.0.1:{PORT_A}",
        node_b_url=f"http://127.0.0.1:{PORT_B}",
        read_timeout=READ_TIMEOUT,
    ))
    servers = [serve(node_a, PORT_A), serve(node_b, PORT_B), serve(gateway, PORT_G)]
    async with httpx.AsyncClient() as client:
        for port in (PORT_A, PORT_B, PORT_G):
            for _ in range(50):
                try:
                    r = await client.get(
                        f"http://127.0.0.1:{port}/info" if port != PORT_G
                        else f"http://127.0.0.1:{port}/api/scenarios")
                    if r.status_code == 200:
                        break
                except httpx.HTTPError:
                    await asyncio.sleep(0.1)

    healthy_ids = expected_token_ids(PROMPT, 0, MAX_TOKENS)
    print(f"确定性基准词元 id 序列: {healthy_ids}")
    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            for scenario in DRAFT_SCENARIOS:
                report(scenario, await consume(client, scenario), healthy_ids)
    finally:
        for s in servers:
            s.should_exit = True
    print(f"\n全部草稿场景试跑通过，耗时 {time.time() - started:.1f}s。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as exc:
        print(f"\n试跑失败: {exc}", file=sys.stderr)
        sys.exit(1)
