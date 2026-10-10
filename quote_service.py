"""Submission and original-message replay, adapted from the upstream QuoteService."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from time import time
from typing import Any

from .errors import SubmissionError
from .message_service import MessageService
from .models import ForwardNode, ForwardSegment, Quote, QuoteSegment
from .napcat_service import NapcatService
from .replay import make_replay_nodes_type
from .store import QuoteRepository
from .utils import make_session_key, random_id

RECEIVED_MESSAGE = "投稿收到了～"
EMPTY_MESSAGE = "还没有投稿，先回复一条消息发送“投稿”吧～"


class QuoteService:
    def __init__(
        self,
        repository: QuoteRepository,
        message_service: MessageService,
        napcat_service: NapcatService,
        components: Any,
    ):
        self.repository = repository
        self.message_service = message_service
        self.napcat_service = napcat_service
        self.components = components
        self.nodes_type = make_replay_nodes_type(components)

    @staticmethod
    def session_key(event: Any) -> str:
        getter = getattr(event, "get_platform_id", None)
        platform_id = str(getter()) if callable(getter) else event.get_platform_name()
        self_id = str(getattr(event.message_obj, "self_id", "") or "")
        prefix = hashlib.sha256(
            json.dumps([platform_id, self_id]).encode()
        ).hexdigest()[:16]
        return (
            prefix + "_" + make_session_key(event.get_group_id(), event.get_sender_id())
        )

    def source_key(self, event: Any, message_id: str) -> str:
        value = json.dumps([self.session_key(event), message_id], ensure_ascii=False)
        return hashlib.sha256(value.encode()).hexdigest()

    def get_reply_message_id(self, event: Any) -> str | None:
        replies = [
            component
            for component in event.get_messages()
            if isinstance(component, self.components.Reply)
        ]
        if len(replies) != 1:
            return None
        value = getattr(replies[0], "id", None)
        return str(value) if value is not None else None

    async def add_quote(self, event: Any) -> Quote:
        message_id = self.get_reply_message_id(event)
        if not message_id:
            raise SubmissionError("请先回复一条消息，再发送“投稿”")
        session_key = self.session_key(event)
        source_key = self.source_key(event, message_id)
        existing = self.repository.find_source(session_key, source_key)
        if existing is not None:
            self.build_quote_batches(existing)
            return existing
        try:
            payload = await asyncio.wait_for(
                self.napcat_service.fetch_onebot_message(event, message_id),
                timeout=self.message_service.limits.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise SubmissionError("原消息获取超时") from exc
        if not payload or "message" not in payload:
            raise SubmissionError("无法获取被回复的原消息，请换一条消息投稿")
        segments = await self.message_service.build_reply_segments(
            event, payload["message"]
        )
        sender = payload.get("sender") or {}
        if not isinstance(sender, dict):
            raise SubmissionError("原消息作者信息格式错误")
        qq = str(sender.get("user_id") or sender.get("qq") or "")
        quote = Quote(
            id=random_id("q_"),
            qq=qq,
            name=str(sender.get("card") or sender.get("nickname") or qq),
            text="".join(
                segment.text for segment in segments if segment.type == "text"
            ),
            created_by=str(event.get_sender_id()),
            created_at=time(),
            group=session_key,
            source_key=source_key,
            source_message_id=message_id,
        )
        if len(segments) == 1 and segments[0].type == "nodes":
            result = await self.repository.create_quote_with_forward_nodes(
                session_key, quote, segments[0].nodes
            )
        else:
            result = await self.repository.create_quote_with_segments(
                session_key, quote, segments
            )
        return result.quote

    def build_quote_batches(self, quote: Quote) -> list[list[Any]]:
        if quote.kind == "forward":
            chain = [self._build_nodes(quote.group, quote.forward_nodes)]
        else:
            chain = self._build_segments(quote.group, quote.segments)
        if not chain:
            raise SubmissionError("投稿内容为空，无法完整回放")
        batches: list[list[Any]] = []
        ordinary: list[Any] = []
        isolated = (
            self.components.Record,
            self.components.Video,
            self.components.File,
            self.components.Nodes,
        )
        for component in chain:
            if isinstance(component, isolated):
                if ordinary:
                    batches.append(ordinary)
                    ordinary = []
                batches.append([component])
            else:
                ordinary.append(component)
        if ordinary:
            batches.append(ordinary)
        return batches

    async def send_quote(self, event: Any, quote: Quote) -> None:
        comp = self.components
        group_id = str(event.get_group_id() or "")
        target_id = group_id or str(event.get_sender_id())
        if not target_id.isdigit():
            raise SubmissionError("无法确定语录发送目标")
        target = {"group_id" if group_id else "user_id": int(target_id)}
        self_id = getattr(event.message_obj, "self_id", None)
        if self_id:
            target["self_id"] = self_id
        actions: list[tuple[str, dict]] = []
        # 先完成整条的资源检查和序列化，避免后段转换失败时已发出前段。
        for batch in self.build_quote_batches(quote):
            if isinstance(batch[0], comp.Nodes):
                payload = await batch[0].to_dict()
                action = (
                    "send_group_forward_msg" if group_id else "send_private_forward_msg"
                )
            else:
                message = []
                for component in batch:
                    if isinstance(component, (comp.Image, comp.Record)):
                        value = await component.convert_to_base64()
                        kind = (
                            "image" if isinstance(component, comp.Image) else "record"
                        )
                        message.append(
                            {"type": kind, "data": {"file": "base64://" + value}}
                        )
                    else:
                        data = await component.to_dict()
                        if isinstance(component, comp.File):
                            file = data["data"].get("file", "")
                            if file and "://" not in file and Path(file).is_absolute():
                                data["data"]["file"] = Path(file).as_uri()
                        message.append(data)
                payload = {"message": message}
                action = "send_group_msg" if group_id else "send_private_msg"
            actions.append((action, {**payload, **target}))
        # OneBot 原样发送文字和 @，绕开框架自动删空白和补空格的行为。
        for action, payload in actions:
            response = await event.bot.api.call_action(action, **payload)
            if isinstance(response, dict) and (
                response.get("status") == "failed"
                or response.get("retcode", 0) not in (0, "0", None)
            ):
                raise SubmissionError("平台拒绝发送语录，请检查 NapCat 日志")

    def _build_nodes(self, session_key: str, nodes: list[ForwardNode]) -> Any:
        if not nodes:
            raise SubmissionError("合并转发内容缺失，无法完整回放")
        return self.nodes_type(
            nodes=[
                self.components.Node(
                    uin=node.sender_uin,
                    name=node.sender_name,
                    time=node.timestamp,
                    content=self._build_segments(session_key, node.segments),
                )
                for node in nodes
            ]
        )

    def _build_segments(
        self, session_key: str, segments: list[QuoteSegment | ForwardSegment]
    ) -> list[Any]:
        comp = self.components
        chain: list[Any] = []
        for segment in segments:
            if segment.type == "text":
                chain.append(comp.Plain(segment.text))
            elif segment.type == "face":
                chain.append(comp.Face(id=segment.face_id))
            elif segment.type == "at":
                chain.append(comp.At(qq=segment.qq, name=segment.name))
            elif segment.type == "nodes":
                chain.append(self._build_nodes(session_key, segment.nodes))
            elif segment.type == "image":
                asset = self.repository.find_asset(session_key, segment.asset_id)
                if asset is None:
                    raise SubmissionError("投稿图片索引缺失，无法完整回放")
                chain.append(
                    comp.Image.fromFileSystem(str(self.repository.asset_path(asset)))
                )
            elif segment.type in {"record", "video", "file"}:
                asset = self.repository.find_media_asset(session_key, segment.asset_id)
                if asset is None:
                    raise SubmissionError("投稿媒体索引缺失，无法完整回放")
                path = self.repository.asset_path(asset)
                if segment.type == "file":
                    chain.append(
                        comp.File(file=str(path), name=asset.display_name or path.name)
                    )
                else:
                    constructor = (
                        comp.Record if segment.type == "record" else comp.Video
                    )
                    chain.append(constructor.fromFileSystem(str(path)))
            else:
                raise SubmissionError(f"投稿含有无法回放的消息类型 {segment.type}")
        if not chain:
            raise SubmissionError("消息内容缺失，无法完整回放")
        return chain
