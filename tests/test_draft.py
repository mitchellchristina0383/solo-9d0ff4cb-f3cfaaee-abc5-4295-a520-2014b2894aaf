"""End-to-end behavior tests for draft (speculative) decoding.

Drives the same real uvicorn harness as test_scenarios.py: node-b proposes
draft batches of up to 4 tokens, node-a verifies each batch, and the gateway
streams the accepted tokens to the browser client.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from common.deterministic import (
    draft_token_id,
    expected_token_ids,
    token_text,
)


PROMPT = "请写一段关于秋天的短文"
MAX_TOKENS = 12
BATCH = 4


async def collect_sse(client: httpx.AsyncClient, scenario: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    async with client.stream(
        "POST", "/api/chat",
        json={"prompt": PROMPT, "max_tokens": MAX_TOKENS,
              "scenario": scenario, "draft": True},
    ) as resp:
        assert resp.status_code == 200
        event = None
        data_lines: list[str] = []
        async for raw in resp.aiter_lines():
            line = raw.rstrip("\r")
            if line == "":
                if event is not None or data_lines:
                    events.append((event or "message", json.loads("\n".join(data_lines))))
                event, data_lines = None, []
            elif line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if event is not None or data_lines:
            events.append((event or "message", json.loads("\n".join(data_lines))))
    return events


def tokens_of(events):
    return [(e["index"], e["id"], e["text"], e["node"], e.get("accepted_as"))
            for name, e in events if name == "token"]


def batches_of(events):
    return [e for name, e in events if name == "draft_batch"]


def names_of(events):
    return [name for name, _ in events]


# --- scenarios ---------------------------------------------------------------

async def _all_match(gateway_client):
    events = await collect_sse(gateway_client, "draft_all_match")
    names = names_of(events)
    assert names[0] == "meta"
    tokens = tokens_of(events)

    # 12 tokens / 4 per batch -> exactly 3 batches, each fully matched.
    batches = batches_of(events)
    assert [b["batch"] for b in batches] == [1, 2, 3]
    assert all(b["draft_count"] == BATCH for b in batches)
    assert all(b["matched"] == BATCH and b["replaced"] == 0
               and b["rejected"] == 0 and b["appended"] == BATCH
               for b in batches)
    assert [b["confirmed_count"] for b in batches] == [4, 8, 12]

    assert len(tokens) == MAX_TOKENS
    assert [t[0] for t in tokens] == list(range(MAX_TOKENS))
    # Every token was a verified draft proposal from node-b.
    assert all(t[3] == "node-b" and t[4] == "draft_match" for t in tokens)
    # The decisive property: identical to a primary-only healthy generation.
    assert [t[1] for t in tokens] == expected_token_ids(PROMPT, 0, MAX_TOKENS)
    done = next(e for n, e in events if n == "done")
    assert done["draft"] is True and done["draft_batches"] == 3
    assert done["identical_to_healthy"] is True
    assert "incomplete" not in names


async def _replace_within_batch(gateway_client):
    events = await collect_sse(gateway_client, "draft_replace")
    tokens = tokens_of(events)
    batches = batches_of(events)

    # Divergence starts at absolute index 2, which sits in batch 1:
    # batch 1 = indices 0..3 -> 2 matches (0,1), replacement at index 2,
    # one remaining draft token (index 3) discarded. "rejected" counts every
    # draft token that was not accepted as a verified match.
    b1 = batches[0]
    assert b1["start_index"] == 0
    assert b1["matched"] == 2 and b1["replaced"] == 1
    assert b1["rejected"] == 2 and b1["discarded"] == 1 and b1["appended"] == 3
    # Every later batch is rejected at its first draft token: no draft is
    # accepted, the primary's replacement is appended and the other drafts
    # are discarded. The final batch is trimmed to the remaining budget.
    for b in batches[1:]:
        assert b["matched"] == 0 and b["replaced"] == 1
        assert b["rejected"] == b["draft_count"]
        assert b["discarded"] == b["draft_count"] - 1
        assert b["appended"] == 1

    # Sequence of confirmed positions after each batch: 3, 4, 5, ..., 12.
    assert [b["confirmed_count"] for b in batches] == list(range(3, MAX_TOKENS + 1))

    assert len(tokens) == MAX_TOKENS
    assert [t[0] for t in tokens] == list(range(MAX_TOKENS))
    # First two tokens: accepted drafts; index 2 onwards: primary replacements.
    assert [t[3] for t in tokens[:2]] == ["node-b", "node-b"]
    assert all(t[3] == "node-a" and t[4] == "primary_replacement"
               for t in tokens[2:])
    # Rejected draft tokens never reach the browser and the final stream is
    # still token-for-token identical to the healthy primary generation.
    assert [t[1] for t in tokens] == expected_token_ids(PROMPT, 0, MAX_TOKENS)
    assert names_of(events)[-1] == "done"


async def _draft_timeout_falls_back_to_primary(gateway_client):
    events = await collect_sse(gateway_client, "draft_timeout")
    names = names_of(events)
    assert "draft_fallback" in names
    fb = next(e for n, e in events if n == "draft_fallback")
    assert fb["reason"] == "draft_timeout"
    assert fb["accepted_count"] == 4  # first batch fully confirmed

    batches = batches_of(events)
    assert len(batches) == 1 and batches[0]["appended"] == 4

    tokens = tokens_of(events)
    assert len(tokens) == MAX_TOKENS
    assert [t[3] for t in tokens] == ["node-b"] * 4 + ["node-a"] * 8
    assert [t[1] for t in tokens] == expected_token_ids(PROMPT, 0, MAX_TOKENS)
    assert names_of(events)[-1] == "done"


async def _draft_error_falls_back_to_primary(gateway_client):
    events = await collect_sse(gateway_client, "draft_error")
    fb = next(e for n, e in events if n == "draft_fallback")
    assert fb["reason"] == "draft_error"
    assert fb["accepted_count"] == 8  # two full batches confirmed

    tokens = tokens_of(events)
    assert len(tokens) == MAX_TOKENS
    assert [t[3] for t in tokens] == ["node-b"] * 8 + ["node-a"] * 4
    assert [t[1] for t in tokens] == expected_token_ids(PROMPT, 0, MAX_TOKENS)
    assert "incomplete" not in names_of(events)


async def _primary_failure_keeps_prefix(gateway_client):
    # Primary /verify is injected to fail at absolute index 5, i.e. while
    # verifying the second batch (starts at 4): batch 1 confirmed 4 tokens.
    events = await collect_sse(gateway_client, "draft_primary_fail")
    tokens = tokens_of(events)
    assert len(tokens) == 4
    assert all(t[3] == "node-b" for t in tokens)
    incomplete = next(e for n, e in events if n == "incomplete")
    assert incomplete["reason"] in ("verify_error", "verify_timeout")
    assert incomplete["accepted_count"] == 4
    assert incomplete["prefix_text"] == " ".join(
        token_text(PROMPT, i) for i in range(4))
    assert [t[1] for t in tokens] == expected_token_ids(PROMPT, 0, 4)
    assert "done" not in names_of(events)


async def _version_mismatch_disables_draft(gateway_client):
    events = await collect_sse(gateway_client, "draft_version_mismatch")
    fb = next(e for n, e in events if n == "draft_fallback")
    assert fb["reason"] == "version_mismatch"
    assert fb["backup_identity"]["tokenizer_version"] == "tok-v2"

    # No draft batches ran; the primary produced the whole reply itself.
    assert batches_of(events) == []
    tokens = tokens_of(events)
    assert len(tokens) == MAX_TOKENS
    assert all(t[3] == "node-a" for t in tokens)
    assert [t[1] for t in tokens] == expected_token_ids(PROMPT, 0, MAX_TOKENS)
    assert names_of(events)[-1] == "done"


async def _shared_budget(gateway_client):
    for scenario in ("draft_all_match", "draft_replace",
                     "draft_timeout", "draft_error"):
        events = await collect_sse(gateway_client, scenario)
        done = next(e for n, e in events if n == "done")
        assert done["token_count"] == MAX_TOKENS
        assert len(tokens_of(events)) == MAX_TOKENS
        indices = [t[0] for t in tokens_of(events)]
        assert indices == list(range(MAX_TOKENS))  # no gaps, no duplicates


async def _last_batch_is_trimmed(gateway_client):
    # A budget that is not a multiple of 4: the final batch must be trimmed.
    events: list[tuple[str, dict]] = []
    async with gateway_client.stream(
        "POST", "/api/chat",
        json={"prompt": PROMPT, "max_tokens": 10,
              "scenario": "draft_all_match", "draft": True},
    ) as resp:
        assert resp.status_code == 200
        event = None
        data_lines: list[str] = []
        async for raw in resp.aiter_lines():
            line = raw.rstrip("\r")
            if line == "":
                if event is not None or data_lines:
                    events.append((event or "message", json.loads("\n".join(data_lines))))
                event, data_lines = None, []
            elif line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if event is not None or data_lines:
            events.append((event or "message", json.loads("\n".join(data_lines))))
    batches = batches_of(events)
    assert [b["draft_count"] for b in batches] == [4, 4, 2]
    assert batches[-1]["appended"] == 2
    done = next(e for n, e in events if n == "done")
    assert done["token_count"] == 10


# --- node-level endpoint contracts ------------------------------------------

async def _draft_and_verify_node_contracts(registry):
    client = registry.client(registry.node_b_url)
    async with client:
        prefix = expected_token_ids(PROMPT, 0, 4)
        # Healthy draft: reproduces the primary stream.
        r = await client.post("/draft", json={
            "request_id": "d1", "prompt": PROMPT, "start": 4,
            "max_tokens": 4, "accepted_ids": prefix,
        })
        assert r.status_code == 200
        data = r.json()
        assert [t["id"] for t in data["tokens"]] == expected_token_ids(PROMPT, 4, 8)
        assert data["diverged"] is False

        # Divergent draft: the proposal differs from the primary at and after
        # draft_diverge_at, and stays deterministic.
        r = await client.post("/draft", json={
            "request_id": "d2", "prompt": PROMPT, "start": 4,
            "max_tokens": 4, "accepted_ids": prefix, "draft_diverge_at": 5,
        })
        data = r.json()
        assert data["diverged"] is True
        ids = [t["id"] for t in data["tokens"]]
        # index 4 still healthy, index 5+ use the divergent pure function.
        assert ids[0] == expected_token_ids(PROMPT, 4, 5)[0]
        assert ids[1:] == [draft_token_id(PROMPT, i) for i in range(5, 8)]
        assert all(ids[k] != expected_token_ids(PROMPT, 4, 8)[k] for k in (1, 2, 3))

        # Batch size is capped at 4.
        r = await client.post("/draft", json={
            "request_id": "d3", "prompt": PROMPT, "start": 0, "max_tokens": 9,
        })
        assert r.status_code == 422

        # Prefix verification is enforced on /draft too.
        r = await client.post("/draft", json={
            "request_id": "d4", "prompt": PROMPT, "start": 2,
            "max_tokens": 2, "accepted_ids": [1, 1],
        })
        assert r.status_code == 409

    # Primary /verify marks matches against its own deterministic tokens.
    client = registry.client(registry.node_a_url)
    async with client:
        good = expected_token_ids(PROMPT, 0, 4)
        tampered = good[:2] + [good[2] ^ 0xFFFF if (good[2] ^ 0xFFFF) != good[2]
                               else (good[2] + 1) % 80] + good[3:]
        r = await client.post("/verify", json={
            "request_id": "v1", "prompt": PROMPT, "start": 0,
            "draft_ids": tampered,
        })
        assert r.status_code == 200
        tokens = r.json()["tokens"]
        assert [t["match"] for t in tokens] == [True, True, False, True]
        assert [t["id"] for t in tokens] == good
        assert tokens[2]["draft_id"] == tampered[2]

        r = await client.post("/verify", json={
            "request_id": "v2", "prompt": PROMPT, "start": 2,
            "draft_ids": good[2:4], "accepted_ids": [1, 1],
        })
        assert r.status_code == 409


# --- sync wrappers -----------------------------------------------------------

def test_draft_all_match_full_batches(gateway_client):
    asyncio.run(_all_match(gateway_client))


def test_draft_replace_within_batch(gateway_client):
    asyncio.run(_replace_within_batch(gateway_client))


def test_draft_timeout_falls_back_to_primary_streaming(gateway_client):
    asyncio.run(_draft_timeout_falls_back_to_primary(gateway_client))


def test_draft_error_falls_back_to_primary_streaming(gateway_client):
    asyncio.run(_draft_error_falls_back_to_primary(gateway_client))


def test_draft_primary_failure_keeps_prefix_incomplete(gateway_client):
    asyncio.run(_primary_failure_keeps_prefix(gateway_client))


def test_draft_version_mismatch_runs_primary_only(gateway_client):
    asyncio.run(_version_mismatch_disables_draft(gateway_client))


def test_draft_shared_budget_and_strict_order(gateway_client):
    asyncio.run(_shared_budget(gateway_client))


def test_draft_final_batch_trimmed_to_remaining_budget(gateway_client):
    asyncio.run(_last_batch_is_trimmed(gateway_client))


def test_draft_and_verify_node_contracts(registry):
    asyncio.run(_draft_and_verify_node_contracts(registry))
