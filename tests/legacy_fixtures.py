"""Seed historical archives directly, without relaxing the new submission policy."""

from __future__ import annotations

from time import time
from typing import Any

from astrbot_plugin_submit_quotes.message_service import DownloadBudget
from astrbot_plugin_submit_quotes.models import (
    PendingForwardNode,
    PendingForwardSegment,
    PreparedImage,
    Quote,
)
from astrbot_plugin_submit_quotes.utils import random_id


async def seed_legacy_quote(service, event, content: list[dict[str, Any]]) -> Quote:
    budget = DownloadBudget()

    async def segments(items):
        result = []
        for item in items:
            kind, data = item["type"], item["data"]
            pending = PendingForwardSegment(type=kind)
            if kind == "text":
                pending.text = data["text"]
            elif kind in {"image", "record", "video", "file"}:
                prepared = await service.message_service._prepare_media(
                    event, data, kind, budget
                )
                if isinstance(prepared, PreparedImage):
                    pending.image = prepared
                else:
                    pending.media = prepared
            elif kind == "face":
                pending.face_id = int(data["id"])
            elif kind == "at":
                pending.qq = str(data["qq"])
                pending.name = data.get("name", "")
            elif kind == "forward":
                pending.type = "nodes"
                pending.nodes = [
                    PendingForwardNode(
                        sender_uin=str(node["sender"]["user_id"]),
                        sender_name=node["sender"]["nickname"],
                        timestamp=int(node.get("time", 0)),
                        segments=await segments(node["content"]),
                    )
                    for node in data["messages"]
                ]
            else:
                raise AssertionError(f"unsupported legacy fixture type: {kind}")
            result.append(pending)
        return result

    pending = await segments(content)
    message_id = service.get_reply_message_id(event)
    assert message_id is not None
    session_key = service.session_key(event)
    quote = Quote(
        id=random_id("q_"),
        qq="23456",
        name="历史作者",
        text="".join(part.text for part in pending if part.type == "text"),
        created_by=str(event.get_sender_id()),
        created_at=time(),
        group=session_key,
        source_key=service.source_key(event, message_id),
        source_message_id=message_id,
    )
    if len(pending) == 1 and pending[0].type == "nodes":
        created = await service.repository.create_quote_with_forward_nodes(
            session_key, quote, pending[0].nodes
        )
    else:
        created = await service.repository.create_quote_with_segments(
            session_key, quote, pending
        )
    return created.quote
