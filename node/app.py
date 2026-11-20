"""Reproducible local mock LLM node.

Two instances of this same app are deployed as node-a and node-b.

The decoded token stream is a pure function of (prompt, token index), so the
backup node reproduces the healthy stream exactly when asked to continue from
the gateway's already-accepted prefix.

Fault injection is driven per request by mock headers (only the gateway uses
them), making every scenario repeatable:

  X-Mock-Model-Override / X-Mock-Tokenizer-Override
      Pretend the node runs a different model / tokenizer version.
  X-Mock-Stall-Seconds
      Emit stall_before tokens, then stop producing anything (the gateway's
      read timeout fires). A detached background task keeps "generating" and
      logs the would-be late packets to prove the gateway ignores them.
  X-Mock-Fail-At
      Fail with HTTP 500 before emitting the token at this absolute index
      (used to make the backup continuation itself fail).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from common.deterministic import (
    draft_token_id,
    draft_token_text,
    expected_token_ids,
    load_vocab,
    token_id,
    token_text,
)

logger = logging.getLogger("mocknode")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")


@dataclass(frozen=True)
class NodeConfig:
    node_id: str = "node"
    model_id: str = "mock-llm-7b"
    tokenizer_version: str = "tok-v1"
    emit_interval: float = 0.12


class GenerateRequest(BaseModel):
    request_id: str
    prompt: str = Field(min_length=1)
    start: int = Field(ge=0)
    max_tokens: int = Field(gt=0)
    accepted_ids: list[int] = Field(default_factory=list)
    prefix_text: str = ""
    stall_before: int | None = Field(default=None, ge=0)
    fail_at: int | None = Field(default=None, ge=0)


class DraftRequest(BaseModel):
    """One speculative batch: propose up to ``max_tokens`` draft tokens that
    continue after the already-confirmed prefix."""
    request_id: str
    prompt: str = Field(min_length=1)
    start: int = Field(ge=0)
    max_tokens: int = Field(gt=0, le=4)
    accepted_ids: list[int] = Field(default_factory=list)
    prefix_text: str = ""
    # Absolute index from which the draft model diverges from the primary
    # (None = healthy draft, proposals reproduce the primary's stream).
    draft_diverge_at: int | None = Field(default=None, ge=0)
    stall_before: int | None = Field(default=None, ge=0)
    fail_at: int | None = Field(default=None, ge=0)


class VerifyRequest(BaseModel):
    """Primary node verifies one draft batch and returns its own tokens for
    the same absolute positions."""
    request_id: str
    prompt: str = Field(min_length=1)
    start: int = Field(ge=0)
    draft_ids: list[int] = Field(min_length=1, max_length=4)
    accepted_ids: list[int] = Field(default_factory=list)
    stall_before: int | None = Field(default=None, ge=0)
    fail_at: int | None = Field(default=None, ge=0)


def sse_line(event: str, data: str) -> bytes:
    lines = "".join(f"data: {line}\n" for line in data.split("\n"))
    return f"event: {event}\n{lines}\n".encode()


def create_app(config: NodeConfig | None = None) -> FastAPI:
    cfg = config or NodeConfig(
        node_id=os.environ.get("NODE_ID", "node"),
        model_id=os.environ.get("MODEL_ID", "mock-llm-7b"),
        tokenizer_version=os.environ.get("TOKENIZER_VERSION", "tok-v1"),
        emit_interval=float(os.environ.get("EMIT_INTERVAL", "0.12")),
    )
    # In-memory record of packets emitted *after* the client went away, keyed
    # by request id. Exposed via /debug/late so tests can assert late packets
    # really happen (and that the gateway ignores them).
    late_packets: dict[str, list[dict]] = {}

    def build_info(model_override: str | None, tokenizer_override: str | None) -> dict:
        vocab = load_vocab()
        return {
            "node_id": cfg.node_id,
            "model_id": model_override or cfg.model_id,
            "tokenizer_version": tokenizer_override or cfg.tokenizer_version,
            "vocab_size": len(vocab),
        }

    app = FastAPI(title=f"mock-node-{cfg.node_id}")

    @app.get("/info")
    async def info(
        x_mock_model_override: str | None = Header(default=None),
        x_mock_tokenizer_override: str | None = Header(default=None),
    ):
        return build_info(x_mock_model_override, x_mock_tokenizer_override)

    @app.get("/debug/late")
    async def debug_late():
        return late_packets

    @app.post("/generate")
    async def generate(
        body: GenerateRequest,
        x_mock_model_override: str | None = Header(default=None),
        x_mock_tokenizer_override: str | None = Header(default=None),
        x_mock_stall_seconds: float | None = Header(default=None),
        x_mock_fail_at: float | None = Header(default=None),
    ):
        # Model/tokenizer identity is part of the contract; the (possibly
        # overridden) identity is also reported inside the stream meta.
        node_info = build_info(x_mock_model_override, x_mock_tokenizer_override)

        fail_at = body.fail_at if body.fail_at is not None else (
            int(x_mock_fail_at) if x_mock_fail_at is not None else None
        )
        stall_before = body.stall_before
        stall_seconds = x_mock_stall_seconds

        # 1. Prefix verification: a continuation request must carry exactly the
        #    prefix this deterministic node would itself have produced.
        if body.start > 0:
            expected_prefix = expected_token_ids(body.prompt, 0, body.start)
            if body.accepted_ids != expected_prefix:
                logger.warning(
                    "%s prefix mismatch for %s: expected %s got %s",
                    cfg.node_id, body.request_id, expected_prefix, body.accepted_ids,
                )
                return JSONResponse(
                    status_code=409,
                    content={"error": "prefix_verification_failed", "expected_ids": expected_prefix},
                )

        # 2. Fail injection before any streaming starts (backup continuation
        #    failing immediately).
        if fail_at is not None and fail_at <= body.start:
            logger.warning("%s injected failure at index %s for %s", cfg.node_id, fail_at, body.request_id)
            return JSONResponse(
                status_code=500, content={"error": "injected_failure", "at": fail_at}
            )

        async def stream():
            yield sse_line("meta", json.dumps(node_info))
            emitted = 0
            try:
                for index in range(body.start, body.start + body.max_tokens):
                    if fail_at is not None and index == fail_at:
                        yield sse_line("error", f'{{"error":"injected_failure","at":{index}}}')
                        logger.warning(
                            "%s stream failure at index %s for %s", cfg.node_id, index, body.request_id
                        )
                        return
                    if stall_before is not None and index >= stall_before:
                        logger.warning(
                            "%s stalling for %ss before index %s (request %s) — client will time out",
                            cfg.node_id, stall_seconds or "inf", index, body.request_id,
                        )
                        # Detached "late generation": keeps producing after the
                        # gateway gives up. These packets are never accepted.
                        asyncio.create_task(
                            _late_generation(
                                late_packets, cfg.node_id, body.request_id, body.prompt,
                                index, body.start + body.max_tokens,
                            )
                        )
                        if stall_seconds:
                            await asyncio.sleep(stall_seconds)
                        else:
                            # Park until the client disconnects and cancels us.
                            while True:
                                await asyncio.sleep(3600)
                    tok_id = token_id(body.prompt, index)
                    payload = (
                        '{"index":%d,"id":%d,"text":%s,"node":%s}'
                        % (index, tok_id, json.dumps(token_text(body.prompt, index)),
                           json.dumps(cfg.node_id))
                    )
                    yield sse_line("token", payload)
                    emitted += 1
                    await asyncio.sleep(cfg.emit_interval)
                yield sse_line("done", '{"reason":"max_tokens"}')
            except asyncio.CancelledError:
                logger.info(
                    "%s stream cancelled by client after %d token(s) for %s",
                    cfg.node_id, emitted, body.request_id,
                )
                raise

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------
    # Draft decoding: speculative /draft proposal and primary /verify.
    # Both are plain JSON endpoints (the gateway requests one batch at a
    # time, so SSE framing adds nothing); they accept the same stall/fail
    # fault injection and the same prefix-verification contract.
    # ------------------------------------------------------------------

    def _check_prefix(prompt: str, start: int, accepted_ids: list[int], req_id: str):
        if start > 0:
            expected_prefix = expected_token_ids(prompt, 0, start)
            if accepted_ids != expected_prefix:
                logger.warning(
                    "%s prefix mismatch for %s: expected %s got %s",
                    cfg.node_id, req_id, expected_prefix, accepted_ids,
                )
                return JSONResponse(
                    status_code=409,
                    content={"error": "prefix_verification_failed",
                             "expected_ids": expected_prefix},
                )
        return None

    @app.post("/draft")
    async def draft(
        body: DraftRequest,
        x_mock_stall_seconds: float | None = Header(default=None),
        x_mock_fail_at: float | None = Header(default=None),
    ):
        fail_at = body.fail_at if body.fail_at is not None else (
            int(x_mock_fail_at) if x_mock_fail_at is not None else None
        )
        bad = _check_prefix(body.prompt, body.start, body.accepted_ids, body.request_id)
        if bad is not None:
            return bad
        if fail_at is not None and fail_at <= body.start:
            logger.warning("%s draft injected failure at index %s for %s",
                           cfg.node_id, fail_at, body.request_id)
            return JSONResponse(status_code=500,
                                content={"error": "injected_failure", "at": fail_at})
        if body.stall_before is not None and body.start >= body.stall_before:
            logger.warning(
                "%s draft stalling for %ss at start index %s (request %s) — caller will time out",
                cfg.node_id, x_mock_stall_seconds or "inf", body.start, body.request_id,
            )
            if x_mock_stall_seconds:
                await asyncio.sleep(x_mock_stall_seconds)
            else:
                while True:
                    await asyncio.sleep(3600)

        tokens = []
        for offset in range(body.max_tokens):
            index = body.start + offset
            if fail_at is not None and index == fail_at:
                return JSONResponse(status_code=500,
                                    content={"error": "injected_failure", "at": index})
            if body.draft_diverge_at is not None and index >= body.draft_diverge_at:
                tok_id = draft_token_id(body.prompt, index)
                text = draft_token_text(body.prompt, index)
            else:
                tok_id = token_id(body.prompt, index)
                text = token_text(body.prompt, index)
            tokens.append({"index": index, "id": tok_id, "text": text,
                           "node": cfg.node_id})
        return {
            "node_id": cfg.node_id,
            "start": body.start,
            "diverged": body.draft_diverge_at is not None,
            "tokens": tokens,
        }

    @app.post("/verify")
    async def verify(
        body: VerifyRequest,
        x_mock_stall_seconds: float | None = Header(default=None),
        x_mock_fail_at: float | None = Header(default=None),
    ):
        fail_at = body.fail_at if body.fail_at is not None else (
            int(x_mock_fail_at) if x_mock_fail_at is not None else None
        )
        bad = _check_prefix(body.prompt, body.start, body.accepted_ids, body.request_id)
        if bad is not None:
            return bad
        if fail_at is not None and fail_at <= body.start:
            logger.warning("%s verify injected failure at index %s for %s",
                           cfg.node_id, fail_at, body.request_id)
            return JSONResponse(status_code=500,
                                content={"error": "injected_failure", "at": fail_at})
        if body.stall_before is not None and body.start >= body.stall_before:
            logger.warning(
                "%s verify stalling for %ss at start index %s (request %s) — caller will time out",
                cfg.node_id, x_mock_stall_seconds or "inf", body.start, body.request_id,
            )
            if x_mock_stall_seconds:
                await asyncio.sleep(x_mock_stall_seconds)
            else:
                while True:
                    await asyncio.sleep(3600)

        tokens = []
        for offset, draft_id in enumerate(body.draft_ids):
            index = body.start + offset
            if fail_at is not None and index == fail_at:
                return JSONResponse(status_code=500,
                                    content={"error": "injected_failure", "at": index})
            primary_id = token_id(body.prompt, index)
            tokens.append({"index": index, "id": primary_id,
                           "text": token_text(body.prompt, index),
                           "node": cfg.node_id, "draft_id": draft_id,
                           "match": draft_id == primary_id})
        return {
            "node_id": cfg.node_id,
            "start": body.start,
            "tokens": tokens,
        }

    return app


async def _late_generation(
    store: dict[str, list[dict]],
    node_id: str,
    request_id: str,
    prompt: str,
    start_index: int,
    end_index: int,
) -> None:
    """Pretend the stalled node eventually finishes and sends late packets."""
    await asyncio.sleep(1.5)
    late = []
    for index in range(start_index, min(end_index, start_index + 4)):
        packet = {
            "ts": time.time(),
            "index": index,
            "id": token_id(prompt, index),
            "text": token_text(prompt, index),
        }
        late.append(packet)
        logger.warning(
            "%s LATE packet for %s index=%d text=%r (gateway must ignore)",
            node_id, request_id, index, packet["text"],
        )
        await asyncio.sleep(0.3)
    store.setdefault(request_id, []).extend(late)


# Uvicorn entry point: node.app:app
app = create_app()
