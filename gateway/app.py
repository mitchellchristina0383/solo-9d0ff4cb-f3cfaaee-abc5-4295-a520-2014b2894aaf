"""Streaming failover gateway.

Responsibilities:

* Serve a browser page and an SSE ``/api/chat`` stream.
* Forward generation to a primary mock node with a *per-read* timeout.
* On mid-stream timeout (or a mid-stream node error), continue on the backup
  node using the original prompt plus the tokens already accepted by the
  gateway (the user has already seen them).
* Append-only state machine: each absolute token index is accepted at most
  once; any late packet from the abandoned primary is ignored by index.
* One shared token budget for the whole reply: the backup only gets
  ``max_tokens - accepted_count``.
* Refuse failover (keep displayed text, mark incomplete) when the two nodes
  report different model/tokenizer identities, or when the backup fails.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from common.deterministic import expected_token_ids, token_text

logger = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")


# ---------------------------------------------------------------------------
# Scenario catalog: repeatable fault injection, applied by the gateway only.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeFault:
    """Fault instructions sent to one mock node for one request."""
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)
    info_headers: dict[str, str] = field(default_factory=dict)


SCENARIOS: dict[str, dict[str, NodeFault]] = {
    # No fault: the primary streams the whole reply.
    "healthy": {"primary": NodeFault(), "backup": NodeFault()},
    # Primary stalls mid-stream -> read timeout -> backup continues and the
    # final reply is identical to the healthy run.
    "failover_success": {
        "primary": NodeFault(
            headers={"X-Mock-Stall-Seconds": "30"},
            body={"stall_before": 4},
        ),
        "backup": NodeFault(),
    },
    # Primary times out, backup itself fails mid-continuation -> keep the
    # displayed prefix and mark the reply incomplete.
    "failover_backup_error": {
        "primary": NodeFault(
            headers={"X-Mock-Stall-Seconds": "30"},
            body={"stall_before": 4},
        ),
        "backup": NodeFault(body={"fail_at": 7}),
    },
    # Primary times out, backup reports a different model/tokenizer version
    # -> no continuation, keep prefix, mark incomplete.
    "version_mismatch": {
        "primary": NodeFault(
            headers={"X-Mock-Stall-Seconds": "30"},
            body={"stall_before": 4},
        ),
        "backup": NodeFault(
            headers={"X-Mock-Model-Override": "mock-llm-7b-hotfix",
                     "X-Mock-Tokenizer-Override": "tok-v2"},
            info_headers={"X-Mock-Model-Override": "mock-llm-7b-hotfix",
                          "X-Mock-Tokenizer-Override": "tok-v2"},
        ),
    },
}

# Draft-decoding (speculative decoding) scenarios. The backup node proposes
# batches of at most DRAFT_BATCH_SIZE tokens; the primary verifies each batch.
# Faults here target the /draft (backup) and /verify (primary) endpoints; when
# a draft call fails the gateway falls back to ordinary primary streaming.
DRAFT_SCENARIOS: dict[str, dict[str, NodeFault]] = {
    # Healthy draft model: every proposal matches -> whole batches accepted.
    "draft_all_match": {"primary": NodeFault(), "backup": NodeFault()},
    # Draft model diverges from absolute index 2: first batch accepts a
    # matching prefix then replaces the first divergent token; later batches
    # are rejected at their first token.
    "draft_replace": {
        "primary": NodeFault(),
        "backup": NodeFault(body={"draft_diverge_at": 2}),
    },
    # First draft batch (indices 0-3) succeeds; the second call stalls and
    # times out -> primary streams the rest from the confirmed position.
    "draft_timeout": {
        "primary": NodeFault(),
        "backup": NodeFault(
            headers={"X-Mock-Stall-Seconds": "30"},
            body={"stall_before": 4},
        ),
    },
    # Two full batches succeed; the third draft call fails outright ->
    # primary streams the rest from the confirmed position.
    "draft_error": {
        "primary": NodeFault(),
        "backup": NodeFault(body={"fail_at": 8}),
    },
    # Primary fails while verifying the second batch -> it cannot continue,
    # so the confirmed prefix is kept and the reply is marked incomplete.
    "draft_primary_fail": {
        "primary": NodeFault(body={"fail_at": 5}),
        "backup": NodeFault(),
    },
    # Backup reports a different model/tokenizer version -> drafts are not
    # trusted; the primary generates the whole reply itself.
    "draft_version_mismatch": {
        "primary": NodeFault(),
        "backup": NodeFault(
            info_headers={"X-Mock-Model-Override": "mock-llm-7b-hotfix",
                          "X-Mock-Tokenizer-Override": "tok-v2"},
        ),
    },
}


@dataclass(frozen=True)
class GatewayConfig:
    node_a_url: str = os.environ.get("NODE_A_URL", "http://node-a:8000")
    node_b_url: str = os.environ.get("NODE_B_URL", "http://node-b:8000")
    primary_name: str = "node-a"
    backup_name: str = "node-b"
    # Maximum silence (no SSE event) before the primary is declared timed out.
    read_timeout: float = float(os.environ.get("GATEWAY_READ_TIMEOUT", "1.0"))
    # Speculative batches contain at most this many tokens.
    draft_batch_size: int = int(os.environ.get("DRAFT_BATCH_SIZE", "4"))
    web_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    max_tokens: int = Field(ge=1, le=128)
    scenario: str = "healthy"
    draft: bool = False


def sse(event: str, data: dict) -> bytes:
    lines = "".join(f"data: {line}\n" for line in json.dumps(data, ensure_ascii=False).split("\n"))
    return f"event: {event}\n{lines}\n".encode()


async def iter_sse_events(response: httpx.Response) -> AsyncIterator[tuple[str, str]]:
    """Parse an httpx SSE stream into (event, data-string) pairs."""
    event: str | None = None
    data_lines: list[str] = []
    async for raw_line in response.aiter_lines():
        line = raw_line.rstrip("\r")
        if line == "":
            if event is not None or data_lines:
                yield event or "message", "\n".join(data_lines)
            event, data_lines = None, []
        elif line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
    if event is not None or data_lines:
        yield event or "message", "\n".join(data_lines)


ClientFactory = Callable[[str], Awaitable[httpx.AsyncClient]]


def default_client_factory(base_url: str) -> httpx.AsyncClient:
    # Generous client timeouts; the gateway enforces its own read timeout so
    # the behavior is identical with in-process test transports.
    return httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(30.0, read=30.0))


def create_app(
    config: GatewayConfig | None = None,
    client_factory: Callable[[str], httpx.AsyncClient] | None = None,
) -> FastAPI:
    cfg = config or GatewayConfig()
    make_client = client_factory or default_client_factory
    app = FastAPI(title="streaming-failover-gateway")

    @app.get("/api/scenarios")
    async def scenarios() -> dict:
        def describe(catalog: dict[str, dict[str, NodeFault]], draft: bool):
            return [
                {"id": sid, "draft": draft,
                 "primary": dict(headers=faults["primary"].headers,
                                 body=faults["primary"].body),
                 "backup": dict(headers=faults["backup"].headers,
                                body=faults["backup"].body)}
                for sid, faults in catalog.items()
            ]
        return {"scenarios": describe(SCENARIOS, False)
                + describe(DRAFT_SCENARIOS, True)}

    @app.get("/api/expected")
    async def expected(prompt: str, max_tokens: int) -> dict:
        """Ground-truth token stream of the deterministic mock model."""
        ids = expected_token_ids(prompt, 0, max_tokens)
        return {"ids": ids, "texts": [token_text(prompt, i) for i in range(max_tokens)]}

    @app.post("/api/chat")
    async def chat(body: ChatRequest) -> StreamingResponse:
        catalog = DRAFT_SCENARIOS if body.draft else SCENARIOS
        if body.scenario not in catalog:
            return StreamingResponse(
                iter([sse("incomplete", {"reason": "unknown_scenario",
                                         "scenario": body.scenario})]),
                media_type="text/event-stream",
            )
        faults = catalog[body.scenario]
        request_id = uuid.uuid4().hex
        orchestrator = _orchestrate_draft if body.draft else _orchestrate
        return StreamingResponse(
            orchestrator(body, request_id, faults, cfg, make_client),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.mount("/", StaticFiles(directory=cfg.web_dir, html=True), name="web")
    return app


# ---------------------------------------------------------------------------
# Orchestration state machine
# ---------------------------------------------------------------------------

async def _orchestrate(
    body: ChatRequest,
    request_id: str,
    faults: dict[str, NodeFault],
    cfg: GatewayConfig,
    make_client: Callable[[str], httpx.AsyncClient],
) -> AsyncIterator[bytes]:
    prompt, max_tokens = body.prompt, body.max_tokens
    accepted_ids: list[int] = []
    accepted_texts: list[str] = []
    provenance: list[str] = []
    primary_identity: tuple[str, str] | None = None
    next_index = 0

    yield sse("meta", {
        "request_id": request_id,
        "scenario": body.scenario,
        "max_tokens": max_tokens,
        "primary": cfg.primary_name,
        "backup": cfg.backup_name,
    })

    def accept_token(obj: dict, node: str) -> bool:
        """Append-only acceptance: one token per absolute index, in order."""
        nonlocal next_index
        index = int(obj["index"])
        if index != next_index:
            logger.warning(
                "ignoring out-of-order/late token index=%s (next expected=%s) from %s",
                index, next_index, node,
            )
            return False
        accepted_ids.append(int(obj["id"]))
        accepted_texts.append(str(obj["text"]))
        provenance.append(node)
        next_index += 1
        return True

    # ---- primary node ------------------------------------------------------
    primary_fault = faults["primary"]
    primary_payload = {
        "request_id": request_id,
        "prompt": prompt,
        "start": 0,
        "max_tokens": max_tokens,
        "accepted_ids": [],
        "prefix_text": "",
        **primary_fault.body,
    }

    timed_out = False
    primary_error: str | None = None
    try:
        async with make_client(cfg.node_a_url) as client:
            async with client.stream("POST", "/generate", json=primary_payload,
                                     headers=primary_fault.headers) as resp:
                if resp.status_code != 200:
                    detail = await resp.aread()
                    primary_error = f"http_{resp.status_code}:{detail[:200]!r}"
                else:
                    event_iter = iter_sse_events(resp)
                    while True:
                        try:
                            event, data = await asyncio.wait_for(
                                event_iter.__anext__(), timeout=cfg.read_timeout
                            )
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            timed_out = True
                            logger.warning(
                                "primary %s read timeout after %d accepted token(s) for %s",
                                cfg.primary_name, next_index, request_id,
                            )
                            break
                        if event == "meta":
                            info = json.loads(data)
                            primary_identity = (info["model_id"], info["tokenizer_version"])
                        elif event == "token":
                            obj = json.loads(data)
                            if accept_token(obj, cfg.primary_name):
                                yield sse("token", {**obj, "node": cfg.primary_name})
                                if next_index >= max_tokens:
                                    break
                        elif event == "error":
                            primary_error = data
                            logger.warning("primary %s stream error: %s", cfg.primary_name, data)
                            break
                        elif event == "done":
                            break
    except httpx.HTTPError as exc:
        primary_error = f"{type(exc).__name__}: {exc}"

    # Healthy completion on the primary: nothing else to do.
    if not timed_out and primary_error is None and next_index >= max_tokens:
        yield sse("done", {
            "reason": "max_tokens", "token_count": next_index,
            "nodes": _node_shares(provenance),
        })
        return

    # Primary finished without error but short of the budget: treat as failure.
    if not timed_out and primary_error is None:
        primary_error = "primary_ended_early"

    if next_index >= max_tokens:
        yield sse("done", {
            "reason": "max_tokens", "token_count": next_index,
            "nodes": _node_shares(provenance),
        })
        return

    yield sse("fallback", {
        "reason": "read_timeout" if timed_out else "primary_error",
        "from_node": cfg.primary_name,
        "to_node": cfg.backup_name,
        "accepted_count": next_index,
        "detail": primary_error,
    })

    # ---- backup node: identity check --------------------------------------
    backup_fault = faults["backup"]
    async with make_client(cfg.node_b_url) as client:
        try:
            info_resp = await client.get("/info", headers=backup_fault.info_headers)
            info_resp.raise_for_status()
            backup_info = info_resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            yield sse("incomplete", {
                "reason": "backup_unreachable",
                "detail": str(exc),
                "accepted_count": next_index,
                "prefix_text": " ".join(accepted_texts),
            })
            return
        backup_identity = (backup_info["model_id"], backup_info["tokenizer_version"])

        if primary_identity is not None and backup_identity != primary_identity:
            logger.warning(
                "refusing failover for %s: identity mismatch primary=%s backup=%s",
                request_id, primary_identity, backup_identity,
            )
            yield sse("incomplete", {
                "reason": "version_mismatch",
                "primary_identity": {"model_id": primary_identity[0],
                                     "tokenizer_version": primary_identity[1]},
                "backup_identity": {"model_id": backup_identity[0],
                                    "tokenizer_version": backup_identity[1]},
                "accepted_count": next_index,
                "prefix_text": " ".join(accepted_texts),
            })
            return

        # ---- backup continuation with the shared token budget -------------
        remaining = max_tokens - next_index
        backup_payload = {
            "request_id": request_id,
            "prompt": prompt,
            "start": next_index,
            "max_tokens": remaining,
            "accepted_ids": accepted_ids,
            "prefix_text": " ".join(accepted_texts),
            **backup_fault.body,
        }
        backup_failed: str | None = None
        try:
            async with client.stream("POST", "/generate", json=backup_payload,
                                     headers=backup_fault.headers) as resp:
                if resp.status_code != 200:
                    detail = await resp.aread()
                    backup_failed = f"http_{resp.status_code}:{detail[:200]!r}"
                else:
                    async for event, data in iter_sse_events(resp):
                        if event == "token":
                            obj = json.loads(data)
                            if accept_token(obj, cfg.backup_name):
                                yield sse("token", {**obj, "node": cfg.backup_name})
                                if next_index >= max_tokens:
                                    break
                        elif event == "error":
                            backup_failed = data
                            break
                        elif event == "done":
                            break
        except httpx.HTTPError as exc:
            backup_failed = f"{type(exc).__name__}: {exc}"

    if backup_failed is not None or next_index < max_tokens:
        logger.warning("backup continuation failed for %s: %s", request_id, backup_failed)
        yield sse("incomplete", {
            "reason": "backup_failed",
            "detail": backup_failed,
            "accepted_count": next_index,
            "prefix_text": " ".join(accepted_texts),
        })
        return

    yield sse("done", {
        "reason": "max_tokens",
        "token_count": next_index,
        "nodes": _node_shares(provenance),
        "identical_to_healthy": True,
    })


# ---------------------------------------------------------------------------
# Draft (speculative) decoding state machine
#
#   1. The backup (draft) node proposes up to DRAFT_BATCH_SIZE tokens that
#      continue after the confirmed prefix.
#   2. The primary node verifies the whole batch in one call and returns its
#      own tokens for the same absolute positions.
#   3. Matching prefix tokens are accepted (attributed to the draft node); at
#      the first mismatch the primary's token replaces the draft, the rest of
#      the batch is discarded, and the loop restarts at the new confirmed
#      position.
#   4. A draft-node timeout/error disables drafting; the primary continues by
#      itself from the confirmed position (ordinary streaming continuation).
#   5. If the primary itself cannot continue, the confirmed prefix is kept and
#      the reply is marked incomplete.
#
# Every accepted token comes from (or is replaced by) the primary's verified
# stream, so a finished draft reply is token-for-token identical to a healthy
# primary-only generation, under the same shared max_tokens budget.
# ---------------------------------------------------------------------------

async def _stream_primary_continuation(
    *,
    client: httpx.AsyncClient,
    node_url: str,
    node_name: str,
    fault: NodeFault,
    request_id: str,
    prompt: str,
    max_tokens: int,
    accepted_ids: list[int],
    accepted_texts: list[str],
    provenance: list[str],
    state: dict,
    cfg: GatewayConfig,
) -> AsyncIterator[bytes]:
    """Primary-only streaming from the current confirmed position.

    Shares the append-only ``accept_token`` state with the caller so the
    single token budget and index sequence stay consistent. Returns (via the
    mutated state dict) whether the stream itself failed.
    """
    next_index = state["get_next_index"]()
    payload = {
        "request_id": request_id,
        "prompt": prompt,
        "start": next_index,
        "max_tokens": max_tokens - next_index,
        "accepted_ids": accepted_ids,
        "prefix_text": " ".join(accepted_texts),
        **fault.body,
    }
    state["stream_failed"] = None
    try:
        async with client.stream("POST", f"{node_url}/generate", json=payload,
                                 headers=fault.headers) as resp:
            if resp.status_code != 200:
                detail = await resp.aread()
                state["stream_failed"] = f"http_{resp.status_code}:{detail[:200]!r}"
                return
            event_iter = iter_sse_events(resp)
            while True:
                try:
                    event, data = await asyncio.wait_for(
                        event_iter.__anext__(), timeout=cfg.read_timeout
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    state["stream_failed"] = "read_timeout"
                    logger.warning(
                        "primary %s continuation read timeout at index %d for %s",
                        node_name, state["get_next_index"](), request_id,
                    )
                    break
                if event == "token":
                    obj = json.loads(data)
                    if state["accept_token"](obj, node_name):
                        yield sse("token", {**obj, "node": node_name})
                        if state["get_next_index"]() >= max_tokens:
                            break
                elif event == "error":
                    state["stream_failed"] = data
                    break
                elif event in ("meta", "done"):
                    continue
    except httpx.HTTPError as exc:
        state["stream_failed"] = f"{type(exc).__name__}: {exc}"


async def _orchestrate_draft(
    body: ChatRequest,
    request_id: str,
    faults: dict[str, NodeFault],
    cfg: GatewayConfig,
    make_client: Callable[[str], httpx.AsyncClient],
) -> AsyncIterator[bytes]:
    prompt, max_tokens = body.prompt, body.max_tokens
    accepted_ids: list[int] = []
    accepted_texts: list[str] = []
    provenance: list[str] = []
    next_index = 0
    primary_identity: tuple[str, str] | None = None
    backup_identity: tuple[str, str] | None = None
    batch_no = 0
    draft_disabled = False

    yield sse("meta", {
        "request_id": request_id,
        "scenario": body.scenario,
        "max_tokens": max_tokens,
        "primary": cfg.primary_name,
        "backup": cfg.backup_name,
        "draft": True,
        "draft_batch_size": cfg.draft_batch_size,
    })

    def accept_token(obj: dict, node: str) -> bool:
        nonlocal next_index
        index = int(obj["index"])
        if index != next_index:
            logger.warning(
                "ignoring out-of-order/late token index=%s (next expected=%s) from %s",
                index, next_index, node,
            )
            return False
        accepted_ids.append(int(obj["id"]))
        accepted_texts.append(str(obj["text"]))
        provenance.append(node)
        next_index += 1
        return True

    state = {"get_next_index": lambda: next_index,
             "accept_token": accept_token, "stream_failed": None}

    # Identity check up front: drafts from an incompatible model/tokenizer are
    # not trusted — fall back to a primary-only run instead.
    async with make_client(cfg.node_a_url) as primary_client, \
            make_client(cfg.node_b_url) as backup_client:
        try:
            pa = await primary_client.get("/info")
            pa.raise_for_status()
            pinfo = pa.json()
            primary_identity = (pinfo["model_id"], pinfo["tokenizer_version"])
            bi = await backup_client.get("/info", headers=faults["backup"].info_headers)
            bi.raise_for_status()
            binfo = bi.json()
            backup_identity = (binfo["model_id"], binfo["tokenizer_version"])
        except (httpx.HTTPError, ValueError) as exc:
            yield sse("draft_fallback", {
                "reason": "identity_check_failed",
                "detail": str(exc),
                "accepted_count": next_index,
            })
            draft_disabled = True

        if not draft_disabled and backup_identity != primary_identity:
            logger.warning(
                "draft disabled for %s: identity mismatch primary=%s backup=%s",
                request_id, primary_identity, backup_identity,
            )
            yield sse("draft_fallback", {
                "reason": "version_mismatch",
                "primary_identity": {"model_id": primary_identity[0],
                                     "tokenizer_version": primary_identity[1]},
                "backup_identity": {"model_id": backup_identity[0],
                                    "tokenizer_version": backup_identity[1]},
                "accepted_count": 0,
            })
            draft_disabled = True

        while next_index < max_tokens and not draft_disabled:
            batch_no += 1
            remaining = max_tokens - next_index
            batch_size = min(cfg.draft_batch_size, remaining)
            primary_fault = faults["primary"]
            backup_fault = faults["backup"]

            # ---- 1. draft proposal from the backup node -------------------
            draft_payload = {
                "request_id": request_id,
                "prompt": prompt,
                "start": next_index,
                "max_tokens": batch_size,
                "accepted_ids": accepted_ids,
                "prefix_text": " ".join(accepted_texts),
                **backup_fault.body,
            }
            try:
                draft_resp = await asyncio.wait_for(
                    backup_client.post("/draft", json=draft_payload,
                                       headers=backup_fault.headers),
                    timeout=cfg.read_timeout,
                )
                draft_resp.raise_for_status()
                draft_data = draft_resp.json()
            except (asyncio.TimeoutError, httpx.HTTPError, ValueError) as exc:
                reason = "draft_timeout" if isinstance(exc, asyncio.TimeoutError) \
                    else "draft_error"
                detail = "read_timeout" if reason == "draft_timeout" else str(exc)
                logger.warning(
                    "draft node %s %s at index %d for %s (%s)",
                    cfg.backup_name, reason, next_index, request_id, detail,
                )
                yield sse("draft_fallback", {
                    "reason": reason, "detail": detail,
                    "accepted_count": next_index,
                })
                draft_disabled = True
                break

            draft_tokens = draft_data.get("tokens", [])
            if not draft_tokens:
                # Defensive: an empty proposal cannot be verified; let the
                # primary take over from the confirmed position.
                yield sse("draft_fallback", {
                    "reason": "draft_empty", "detail": "no draft tokens",
                    "accepted_count": next_index,
                })
                draft_disabled = True
                break

            # ---- 2. whole-batch verification on the primary node ----------
            verify_payload = {
                "request_id": request_id,
                "prompt": prompt,
                "start": next_index,
                "draft_ids": [int(t["id"]) for t in draft_tokens],
                "accepted_ids": accepted_ids,
                **primary_fault.body,
            }
            try:
                verify_resp = await asyncio.wait_for(
                    primary_client.post("/verify", json=verify_payload,
                                        headers=primary_fault.headers),
                    timeout=cfg.read_timeout,
                )
                verify_resp.raise_for_status()
                verify_data = verify_resp.json()
            except (asyncio.TimeoutError, httpx.HTTPError, ValueError) as exc:
                reason = "verify_timeout" if isinstance(exc, asyncio.TimeoutError) \
                    else "verify_error"
                logger.warning(
                    "primary %s %s at index %d for %s — cannot continue",
                    cfg.primary_name, reason, next_index, request_id,
                )
                yield sse("incomplete", {
                    "reason": reason,
                    "detail": "read_timeout" if reason == "verify_timeout" else str(exc),
                    "accepted_count": next_index,
                    "prefix_text": " ".join(accepted_texts),
                })
                return

            verified = verify_data.get("tokens", [])
            if not verified:
                yield sse("incomplete", {
                    "reason": "verify_empty",
                    "detail": "primary returned no verified tokens",
                    "accepted_count": next_index,
                    "prefix_text": " ".join(accepted_texts),
                })
                return

            # ---- 3. accept the matching prefix, replace at first mismatch --
            matched = 0
            rejected = 0
            discarded = 0
            replaced = False
            for pos, vtok in enumerate(verified):
                draft_tok = draft_tokens[pos]
                if not replaced and int(vtok["id"]) == int(draft_tok["id"]):
                    obj = {"index": int(vtok["index"]), "id": int(vtok["id"]),
                           "text": vtok["text"]}
                    if accept_token(obj, cfg.backup_name):
                        matched += 1
                        yield sse("token", {**obj, "node": cfg.backup_name,
                                            "draft_batch": batch_no,
                                            "accepted_as": "draft_match"})
                else:
                    # First mismatch: take the primary's token, discard the
                    # remaining draft tokens of this batch.
                    obj = {"index": int(vtok["index"]), "id": int(vtok["id"]),
                           "text": vtok["text"]}
                    accept_token(obj, cfg.primary_name)
                    rejected = len(draft_tokens) - matched
                    discarded = len(draft_tokens) - pos - 1
                    replaced = True
                    yield sse("token", {**obj, "node": cfg.primary_name,
                                        "draft_batch": batch_no,
                                        "accepted_as": "primary_replacement"})
                    break

            yield sse("draft_batch", {
                "batch": batch_no,
                "start_index": int(verified[0]["index"]),
                "draft_count": len(draft_tokens),
                "verified_count": len(verified),
                "matched": matched,
                "replaced": 1 if replaced else 0,
                "rejected": rejected,
                "discarded": discarded,
                "appended": matched + (1 if replaced else 0),
                "confirmed_count": next_index,
            })

        # ---- 4. drafting disabled (or unavailable): primary takes over -----
        if draft_disabled and next_index < max_tokens:
            async for chunk in _stream_primary_continuation(
                client=primary_client,
                node_url=cfg.node_a_url,
                node_name=cfg.primary_name,
                fault=faults["primary"],
                request_id=request_id,
                prompt=prompt,
                max_tokens=max_tokens,
                accepted_ids=accepted_ids,
                accepted_texts=accepted_texts,
                provenance=provenance,
                state=state,
                cfg=cfg,
            ):
                yield chunk
            next_index = state["get_next_index"]()
            failed = state["stream_failed"]
            if failed is not None or next_index < max_tokens:
                yield sse("incomplete", {
                    "reason": "primary_failed",
                    "detail": failed or "primary_ended_early",
                    "accepted_count": next_index,
                    "prefix_text": " ".join(accepted_texts),
                })
                return

    if next_index < max_tokens:
        yield sse("incomplete", {
            "reason": "primary_ended_early",
            "detail": "primary stopped before the token budget was reached",
            "accepted_count": next_index,
            "prefix_text": " ".join(accepted_texts),
        })
        return

    yield sse("done", {
        "reason": "max_tokens",
        "token_count": next_index,
        "nodes": _node_shares(provenance),
        "identical_to_healthy": True,
        "draft": True,
        "draft_batches": batch_no,
    })


def _node_shares(provenance: list[str]) -> dict[str, int]:
    shares: dict[str, int] = {}
    for node in provenance:
        shares[node] = shares.get(node, 0) + 1
    return shares


app = create_app()
