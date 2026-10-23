"""Strict adaptation of the upstream ImageService's reply/forward media parsing."""

from __future__ import annotations

import asyncio
import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import httpx
from PIL import Image

from .errors import SubmissionError
from .models import (
    PendingForwardNode,
    PendingForwardSegment,
    PreparedImage,
    PreparedMedia,
)
from .napcat_service import NapcatService
from .utils import parse_cq_message, prepare_image, prepare_media

MEDIA_NAMES = {"image": "图片", "record": "语音", "video": "视频", "file": "文件"}
SUBMISSION_TYPE_ERROR = "仅支持文字和图片投稿（合并转发也只能包含文字和图片）"
SUBMISSION_TYPES = {"text", "plain", "image", "forward", "node", "nodes"}


@dataclass(frozen=True, slots=True)
class MediaLimits:
    timeout_seconds: int = 20
    max_file_bytes: int = 50 * 1024 * 1024
    max_submission_bytes: int = 100 * 1024 * 1024

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> MediaLimits:
        def positive_int(key: str, default: int) -> int:
            value = config.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"配置 {key} 必须为正整数")
            return value

        return cls(
            timeout_seconds=positive_int("download_timeout_seconds", 20),
            max_file_bytes=positive_int("max_file_size_mb", 50) * 1024 * 1024,
            max_submission_bytes=positive_int("max_submission_size_mb", 100)
            * 1024
            * 1024,
        )


@dataclass(slots=True)
class DownloadBudget:
    used_bytes: int = 0


class MessageService:
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        napcat_service: NapcatService,
        limits: MediaLimits,
    ):
        self.http_client = http_client
        self.napcat_service = napcat_service
        self.limits = limits

    async def build_reply_segments(
        self, event: Any, message: Any
    ) -> list[PendingForwardSegment]:
        # 先解析整棵转发树并检查类型，禁止内容不能触发图片或其他媒体下载。
        message = await self._check_submission_types(event, message, depth=0)
        return await self._build_segments(event, message, DownloadBudget(), depth=0)

    @staticmethod
    def _message_items(message: Any) -> list[dict]:
        if isinstance(message, str):
            message = parse_cq_message(message)
        elif isinstance(message, dict):
            message = [message]
        if not isinstance(message, list) or not message:
            raise SubmissionError("未获取到可保存的消息内容")
        for item in message:
            if not isinstance(item, dict) or not isinstance(item.get("data", {}), dict):
                raise SubmissionError("消息段格式错误，无法完整保存")
            seg_type = str(item.get("type") or "").lower()
            if seg_type not in SUBMISSION_TYPES:
                raise SubmissionError(
                    f"{SUBMISSION_TYPE_ERROR}；不支持消息类型 {seg_type or 'unknown'}"
                )
        return message

    async def _check_submission_types(
        self, event: Any, message: Any, depth: int
    ) -> list[dict]:
        checked: list[dict] = []
        for item in self._message_items(message):
            kind = str(item.get("type") or "").lower()
            if kind not in {"forward", "node", "nodes"}:
                checked.append(item)
                continue
            if depth >= 3:
                raise SubmissionError("合并转发超过三层，无法完整保存")
            nodes = []
            for node in await self._forward_messages(event, item):
                if not isinstance(node, dict):
                    raise SubmissionError("合并转发节点格式错误")
                if node.get("type") == "node":
                    node = node.get("data", {})
                if not isinstance(node, dict):
                    raise SubmissionError("合并转发节点格式错误")
                chain = next(
                    (
                        node[key]
                        for key in ("content", "message", "raw_message")
                        if key in node
                    ),
                    None,
                )
                content = await self._check_submission_types(event, chain, depth + 1)
                nodes.append({**node, "content": content})
            # 缓存已获取的转发内容，后续媒体处理不再次请求平台转发接口。
            checked.append({"type": "nodes", "data": {"nodes": nodes}})
        return checked

    async def _build_segments(
        self, event: Any, message: Any, budget: DownloadBudget, depth: int
    ) -> list[PendingForwardSegment]:
        segments: list[PendingForwardSegment] = []
        for item in self._message_items(message):
            seg_type = str(item.get("type") or "").lower()
            data = item.get("data", {})
            if seg_type in {"text", "plain"}:
                text = data.get("text")
                if not isinstance(text, str):
                    raise SubmissionError("文字消息段内容无效")
                segments.append(PendingForwardSegment(type="text", text=text))
            elif seg_type == "image":
                prepared = await self._prepare_media(event, data, seg_type, budget)
                segments.append(PendingForwardSegment(type="image", image=prepared))
            elif seg_type in {"forward", "node", "nodes"}:
                if depth >= 3:
                    raise SubmissionError("合并转发超过三层，无法完整保存")
                nodes = await self._build_forward_nodes(event, item, budget, depth + 1)
                segments.append(PendingForwardSegment(type="nodes", nodes=nodes))
            else:
                raise SubmissionError(
                    f"暂不支持消息类型 {seg_type or 'unknown'}，无法完整保存"
                )
        if not any(segment.type != "text" or segment.text for segment in segments):
            raise SubmissionError("原消息内容为空")
        return segments

    async def _forward_messages(self, event: Any, item: dict) -> list[dict]:
        data = item.get("data", {})
        seg_type = str(item.get("type") or "").lower()
        if seg_type == "node":
            messages = [data]
        elif seg_type == "nodes":
            messages = data.get("nodes") or data.get("messages")
        else:
            forward_id, payload = self.napcat_service.extract_forward_reference(item)
            messages = await asyncio.wait_for(
                self.napcat_service.fetch_forward_messages(
                    event, forward_id=forward_id, forward_payload=payload
                ),
                timeout=self.limits.timeout_seconds,
            )
        if not isinstance(messages, list) or not messages:
            raise SubmissionError("无法获取完整的合并转发内容")
        return messages

    async def _build_forward_nodes(
        self, event: Any, item: dict, budget: DownloadBudget, depth: int
    ) -> list[PendingForwardNode]:
        nodes: list[PendingForwardNode] = []
        for raw_node in await self._forward_messages(event, item):
            if not isinstance(raw_node, dict):
                raise SubmissionError("合并转发节点格式错误")
            if raw_node.get("type") == "node":
                raw_node = raw_node.get("data", {})
            if not isinstance(raw_node, dict):
                raise SubmissionError("合并转发节点格式错误")
            sender = raw_node.get("sender") or {}
            if not isinstance(sender, dict):
                raise SubmissionError("合并转发作者信息格式错误")
            uin = str(
                sender.get("user_id")
                or sender.get("uin")
                or raw_node.get("user_id")
                or raw_node.get("uin")
                or ""
            )
            name = str(
                sender.get("card")
                or sender.get("nickname")
                or raw_node.get("nickname")
                or raw_node.get("name")
                or uin
            )
            if not uin or not name:
                raise SubmissionError("合并转发缺少作者信息")
            chain = next(
                (
                    raw_node[key]
                    for key in ("content", "message", "raw_message")
                    if key in raw_node
                ),
                None,
            )
            segments = await self._build_segments(event, chain, budget, depth)
            try:
                timestamp = int(raw_node.get("time") or 0)
            except (TypeError, ValueError) as exc:
                raise SubmissionError("合并转发时间格式错误") from exc
            nodes.append(
                PendingForwardNode(
                    sender_uin=uin,
                    sender_name=name,
                    segments=segments,
                    timestamp=timestamp,
                )
            )
        return nodes

    def _check_size(self, size: int, budget: DownloadBudget) -> None:
        if size > self.limits.max_file_bytes:
            raise SubmissionError("单个媒体超过大小上限")
        if size + budget.used_bytes > self.limits.max_submission_bytes:
            raise SubmissionError("本条投稿的媒体总大小超过上限")

    async def _prepare_media(
        self, event: Any, data: dict, media_type: str, budget: DownloadBudget
    ) -> PreparedImage | PreparedMedia:
        name = MEDIA_NAMES[media_type]
        reported_size = data.get("file_size", data.get("size"))
        if reported_size is not None:
            try:
                size = int(reported_size)
            except (ValueError, TypeError) as exc:
                raise SubmissionError(f"{name}大小信息无效") from exc
            if size < 0:
                raise SubmissionError(f"{name}大小信息无效")
            self._check_size(size, budget)
        try:
            content, content_type, source = await asyncio.wait_for(
                self._load_media(event, data, media_type, budget),
                timeout=self.limits.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise SubmissionError(f"{name}获取超时") from exc
        self._check_size(len(content), budget)
        if not content and media_type != "file":
            raise SubmissionError(f"{name}内容为空")
        if media_type == "image":
            try:
                with Image.open(io.BytesIO(content)) as image:
                    actual_format = image.format
                    image.verify()
                prepared = prepare_image(
                    content, source=source, content_type=content_type
                )
                if actual_format:
                    prepared.extension = {"JPEG": ".jpg"}.get(
                        actual_format, "." + actual_format.lower()
                    )
            except Exception as exc:
                raise SubmissionError("图片内容无效，无法完整保存") from exc
        else:
            display_name = str(
                data.get("name") or data.get("fileName") or data.get("file_name") or ""
            )
            if not display_name:
                display_name = Path(urlparse(source).path).name or media_type
            prepared = prepare_media(
                content,
                media_type=media_type,
                source=source,
                content_type=content_type,
                display_name=display_name,
            )
        budget.used_bytes += len(content)
        return prepared

    @staticmethod
    def _sources(data: dict) -> list[str]:
        sources = [
            str(data[key])
            for key in (
                "url",
                "image_url",
                "source_url",
                "src",
                "origin_url",
                "path",
                "file_path",
                "file",
            )
            if isinstance(data.get(key), str) and data[key]
        ]
        if isinstance(data.get("base64"), str) and data["base64"]:
            sources.append("base64://" + data["base64"].removeprefix("base64://"))
        return list(dict.fromkeys(sources))

    async def _load_media(
        self, event: Any, data: dict, media_type: str, budget: DownloadBudget
    ) -> tuple[bytes, str, str]:
        seen: set[str] = set()

        async def try_sources(payload: dict) -> tuple[bytes, str, str] | None:
            for source in self._sources(payload):
                if source in seen:
                    continue
                seen.add(source)
                try:
                    content, content_type = await self._read_source(source, budget)
                    return content, content_type, source
                except SubmissionError:
                    raise
                except (OSError, ValueError, httpx.HTTPError):
                    continue
            return None

        found = await try_sources(data)
        if found is not None:
            return found
        file_id = str(data.get("file_id") or data.get("file") or "")
        actions: list[tuple[str, dict]] = []
        if file_id:
            if media_type == "image":
                actions.append(("get_image", {"file": file_id}))
            elif media_type == "record":
                actions.append(("get_record", {"file": file_id, "out_format": "mp3"}))
            if media_type == "file" and data.get("file_id"):
                if event.get_group_id():
                    params = {"group_id": int(event.get_group_id()), "file_id": file_id}
                    if data.get("busid") is not None:
                        params["busid"] = data["busid"]
                    actions.append(("get_group_file_url", params))
                else:
                    actions.append(("get_private_file_url", {"file_id": file_id}))
            actions.append(
                (
                    "get_file",
                    {"file_id": file_id} if data.get("file_id") else {"file": file_id},
                )
            )
        for action, params in actions:
            try:
                response = await event.bot.api.call_action(action, **params)
            except Exception:
                continue
            payload = self.napcat_service._unwrap_action_payload(response)
            found = await try_sources(payload)
            if found is not None:
                return found
        raise SubmissionError(f"无法获取或下载完整的{MEDIA_NAMES[media_type]}")

    async def _read_source(
        self, source: str, budget: DownloadBudget
    ) -> tuple[bytes, str]:
        if source.startswith(("http://", "https://")):
            async with self.http_client.stream(
                "GET",
                source,
                follow_redirects=True,
                timeout=self.limits.timeout_seconds,
            ) as response:
                response.raise_for_status()
                length = response.headers.get("content-length")
                if length and length.isdigit():
                    self._check_size(int(length), budget)
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    self._check_size(len(content) + len(chunk), budget)
                    content.extend(chunk)
                return bytes(content), response.headers.get("content-type", "")
        if source.startswith("base64://") or source.startswith("data:"):
            content_type = ""
            if source.startswith("data:"):
                header, separator, encoded = source.partition(",")
                if not separator or not header.endswith(";base64"):
                    raise ValueError("不支持的 data URI")
                content_type = header[5:].split(";", 1)[0]
            else:
                encoded = source[len("base64://") :]
            maximum = min(
                self.limits.max_file_bytes,
                self.limits.max_submission_bytes - budget.used_bytes,
            )
            if len(encoded) > ((maximum + 2) // 3) * 4:
                raise SubmissionError("媒体内容超过大小上限")
            content = base64.b64decode(encoded, validate=True)
            self._check_size(len(content), budget)
            return content, content_type
        path = (
            Path(unquote(urlparse(source).path))
            if source.startswith("file://")
            else Path(source)
        )
        maximum = min(
            self.limits.max_file_bytes,
            self.limits.max_submission_bytes - budget.used_bytes,
        )

        def read_local() -> bytes:
            if not path.is_file():
                raise FileNotFoundError(path.name)
            self._check_size(path.stat().st_size, budget)
            with path.open("rb") as stream:
                content = stream.read(maximum + 1)
            self._check_size(len(content), budget)
            return content

        return await asyncio.to_thread(read_local), ""
