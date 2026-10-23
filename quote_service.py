"""Submission and original-message replay, adapted from the upstream QuoteService."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
from contextlib import AsyncExitStack
from pathlib import Path
from time import time
from typing import Any

from PIL import Image as PILImage
from PIL import ImageOps

from .errors import SubmissionError
from .message_service import SUBMISSION_TYPE_ERROR, MessageService
from .models import (
    ForwardNode,
    ForwardSegment,
    PendingForwardNode,
    PendingForwardSegment,
    PreparedImage,
    PreparedMedia,
    Quote,
    QuoteSegment,
)
from .moderation import ContentModerator
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
        moderator: ContentModerator | None = None,
    ):
        self.repository = repository
        self.message_service = message_service
        self.napcat_service = napcat_service
        self.components = components
        self.nodes_type = make_replay_nodes_type(components)
        self.moderator = moderator

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
            async with self.repository.get_store(session_key).lock:
                existing = self.repository.find_source(session_key, source_key)
                if existing is not None:
                    self._validate_stored_submission_types(existing)
                    if self.moderator is not None:
                        pending = self._stored_review_segments(existing)
            if existing is not None and self.moderator is not None:
                await self.moderator.review(event, pending)
            if existing is not None:
                async with self.repository.get_store(session_key).lock:
                    existing = self.repository.find_source(session_key, source_key)
                    if existing is not None:
                        self._validate_stored_submission_types(existing)
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
        if self.moderator is not None:
            # 只审查被回复的原消息；通过之前不提交记录、媒体或索引。
            await self.moderator.review(event, segments)
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

    @staticmethod
    def _validate_stored_submission_types(quote: Quote) -> None:
        # 重复投稿也受当前收录策略约束；检查类型时不读取或转换任何媒体。
        def walk(segments: list[QuoteSegment | ForwardSegment], depth: int) -> None:
            for segment in segments:
                if segment.type in {"text", "image"}:
                    continue
                if segment.type != "nodes":
                    raise SubmissionError(
                        f"{SUBMISSION_TYPE_ERROR}；旧投稿包含 {segment.type or 'unknown'}"
                    )
                if depth >= 3:
                    raise SubmissionError("合并转发超过三层，无法完整保存")
                for node in segment.nodes:
                    walk(node.segments, depth + 1)

        walk(quote.segments, 0)
        for node in quote.forward_nodes:
            walk(node.segments, 1)

    def _stored_review_segments(self, quote: Quote) -> list[PendingForwardSegment]:
        review_images = (
            getattr(getattr(self.moderator, "config", None), "scope", "text_and_images")
            != "text_only"
        )

        def convert(
            segments: list[QuoteSegment | ForwardSegment],
        ) -> list[PendingForwardSegment]:
            pending: list[PendingForwardSegment] = []
            for segment in segments:
                item = PendingForwardSegment(
                    type=segment.type,
                    text=segment.text,
                    qq=segment.qq,
                    name=segment.name,
                    face_id=segment.face_id,
                )
                if segment.type == "image" and review_images:
                    asset = self.repository.find_asset(quote.group, segment.asset_id)
                    if asset is None:
                        raise SubmissionError("投稿图片索引缺失，无法审查")
                    path = self.repository.asset_path(asset)
                    item.image = PreparedImage(
                        content=path.read_bytes(),
                        extension=path.suffix,
                        sha256=asset.sha256,
                        width=asset.width,
                        height=asset.height,
                    )
                elif segment.type == "nodes":
                    item.nodes = nodes(segment.nodes)
                elif segment.type in {"record", "video", "file"}:
                    asset = self.repository.find_media_asset(
                        quote.group, segment.asset_id
                    )
                    if asset is None:
                        raise SubmissionError("投稿媒体索引缺失，无法审查")
                    item.media = PreparedMedia(
                        content=b"",
                        extension=Path(asset.file_name).suffix,
                        media_type=segment.type,
                        display_name=asset.display_name,
                    )
                pending.append(item)
            return pending

        def nodes(saved: list[ForwardNode]) -> list[PendingForwardNode]:
            return [
                PendingForwardNode(
                    sender_uin=node.sender_uin,
                    sender_name=node.sender_name,
                    timestamp=node.timestamp,
                    segments=convert(node.segments),
                )
                for node in saved
            ]

        if quote.kind == "forward":
            return [PendingForwardSegment(type="nodes", nodes=nodes(quote.forward_nodes))]
        return convert(quote.segments)

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
        await self._send_quotes(event, [quote], banquet=False)

    async def send_catalog(
        self,
        event: Any,
        entries: list[tuple[Quote, str]],
        *,
        header: str,
        footer: str,
    ) -> None:
        if len(entries) > 10:
            raise SubmissionError("每页最多预览 10 条投稿")
        group_id = str(event.get_group_id() or "")
        target_id = group_id or str(event.get_sender_id())
        if not target_id.isdigit():
            raise SubmissionError("无法确定投稿预览发送目标")
        self_id = getattr(event.message_obj, "self_id", None)
        sender_id = str(self_id or event.get_sender_id())
        timestamp = int(time())

        def text_segment(text: str) -> dict:
            return {"type": "text", "data": {"text": text}}

        def preview_node(content: list[dict]) -> dict:
            return {
                "type": "node",
                "data": {
                    "user_id": sender_id,
                    "nickname": "投稿预览",
                    "time": timestamp,
                    "content": content,
                },
            }

        messages = [preview_node([text_segment(header)])]
        async with AsyncExitStack() as locks:
            # 缩略图生成完毕即可释放资源锁；发送的是内存图片，不再依赖原文件。
            sessions = sorted({quote.group for quote, _ in entries})
            for session in sessions:
                await locks.enter_async_context(self.repository.get_store(session).lock)
            current = {
                (session, quote.id): quote
                for session in sessions
                for quote in self.repository.list_quotes(session)
            }
            for quote, preview in entries:
                saved = current.get((quote.group, quote.id))
                content = [text_segment(f"ID：{quote.id}\n{preview}")]
                if saved is None:
                    content.append(text_segment("\n[投稿已删除]"))
                else:
                    image_ids = self._catalog_image_ids(saved)
                    for index, asset_id in enumerate(image_ids[:3], start=1):
                        try:
                            asset = self.repository.find_asset(saved.group, asset_id)
                            if asset is None:
                                raise SubmissionError("投稿图片索引缺失")
                            path = self.repository.asset_path(asset)
                            thumbnail = await asyncio.to_thread(
                                self._catalog_thumbnail, path
                            )
                        except Exception:
                            # 损坏的投稿也必须能通过 ID 识别和删除。
                            content.append(
                                text_segment(f"\n[第 {index} 张图片无法预览]")
                            )
                        else:
                            content.append(
                                {"type": "image", "data": {"file": thumbnail}}
                            )
                    if len(image_ids) > 3:
                        content.append(
                            text_segment(f"\n[还有 {len(image_ids) - 3} 张图片]")
                        )
                messages.append(preview_node(content))
        messages.append(preview_node([text_segment(footer)]))
        target = {"group_id" if group_id else "user_id": int(target_id)}
        if self_id:
            target["self_id"] = self_id
        action = "send_group_forward_msg" if group_id else "send_private_forward_msg"
        response = await event.bot.api.call_action(action, messages=messages, **target)
        if isinstance(response, dict) and (
            response.get("status") == "failed"
            or response.get("retcode", 0) not in (0, "0", None)
        ):
            raise SubmissionError("平台拒绝发送投稿预览，请检查 NapCat 日志")

    @staticmethod
    def _catalog_image_ids(quote: Quote) -> list[str]:
        image_ids: list[str] = []

        def walk(segments: list[QuoteSegment | ForwardSegment]) -> None:
            for segment in segments:
                if segment.type == "image":
                    image_ids.append(segment.asset_id)
                if segment.type == "nodes":
                    for node in segment.nodes:
                        walk(node.segments)

        if quote.kind == "forward":
            for node in quote.forward_nodes:
                walk(node.segments)
        else:
            walk(quote.segments)
        return image_ids or list(quote.image_ids)

    @staticmethod
    def _catalog_thumbnail(path: Path) -> str:
        with PILImage.open(path) as original:
            original.seek(0)
            # 动图只取首帧，且只生成内存副本，不覆盖存档图片。
            frame = ImageOps.exif_transpose(original)
            frame.thumbnail((480, 480), PILImage.Resampling.LANCZOS)
            if "A" in frame.getbands() or "transparency" in frame.info:
                rgba = frame.convert("RGBA")
                thumbnail = PILImage.new("RGB", rgba.size, "white")
                thumbnail.paste(rgba, mask=rgba.getchannel("A"))
            else:
                thumbnail = frame.convert("RGB")
            output = io.BytesIO()
            thumbnail.save(output, format="JPEG", quality=85)
        return "base64://" + base64.b64encode(output.getvalue()).decode("ascii")

    async def send_banquet(self, event: Any, quotes: list[Quote]) -> None:
        if not 6 <= len(quotes) <= 9:
            raise SubmissionError("摆酒席需要 6～9 条不同投稿")
        if len({(quote.group, quote.id) for quote in quotes}) != len(quotes):
            raise SubmissionError("摆酒席不能重复使用同一条投稿")
        await self._send_quotes(event, quotes, banquet=True)

    async def _send_quotes(
        self, event: Any, quotes: list[Quote], *, banquet: bool
    ) -> None:
        async with AsyncExitStack() as locks:
            # 固定顺序取锁，且持有到回放完成，避免删除正在发送的媒体。
            for session in sorted({quote.group for quote in quotes}):
                await locks.enter_async_context(self.repository.get_store(session).lock)
            current = {
                (session, quote.id): quote
                for session in {quote.group for quote in quotes}
                for quote in self.repository.list_quotes(session)
            }
            actions: list[tuple[str, dict]] = []
            for index, quote in enumerate(quotes, start=1):
                saved = current.get((quote.group, quote.id))
                if saved is None:
                    raise SubmissionError("投稿已被删除，请重新抽取")
                label = ("\n" if index > 1 else "") + f"第{index}道菜："
                prepared = await self._prepare_quote_actions(
                    event, saved, prefix=label if banquet else ""
                )
                for action, payload in prepared:
                    # 连续普通消息可以一起发送，使纯文字酒席呈现为一条多行消息。
                    if (
                        banquet
                        and actions
                        and action in {"send_group_msg", "send_private_msg"}
                        and actions[-1][0] == action
                        and all(
                            part["type"] not in {"record", "video", "file"}
                            for part in actions[-1][1]["message"] + payload["message"]
                        )
                    ):
                        actions[-1][1]["message"].extend(payload["message"])
                    else:
                        actions.append((action, payload))
            # 所有菜品的资源检查和序列化均已完成，再开始发送。
            for action, payload in actions:
                response = await event.bot.api.call_action(action, **payload)
                if isinstance(response, dict) and (
                    response.get("status") == "failed"
                    or response.get("retcode", 0) not in (0, "0", None)
                ):
                    raise SubmissionError("平台拒绝发送语录，请检查 NapCat 日志")

    async def _prepare_quote_actions(
        self, event: Any, quote: Quote, *, prefix: str = ""
    ) -> list[tuple[str, dict]]:
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
        batches = self.build_quote_batches(quote)
        if prefix:
            if isinstance(batches[0][0], (comp.Record, comp.Video, comp.File, comp.Nodes)):
                batches.insert(0, [comp.Plain(prefix)])
            else:
                batches[0].insert(0, comp.Plain(prefix))
        for batch in batches:
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
        return actions

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
