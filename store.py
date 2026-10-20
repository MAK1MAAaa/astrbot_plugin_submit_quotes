"""Adapted from exynos967/astrbot_quotes_plugin (feb4bb3), author 薄暝.

Keep the upstream session/media layout, but commit quotes and asset indexes together.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import time

from .constants import (
    GROUPS_DIRNAME,
    IMAGES_DIRNAME,
    MEDIA_DIRNAME,
    QUOTES_FILENAME,
    SCHEMA_VERSION,
)
from .errors import SubmissionError
from .models import (
    ForwardNode,
    ForwardSegment,
    ImageAsset,
    MediaAsset,
    PendingForwardNode,
    PendingForwardSegment,
    PendingQuoteSegment,
    PreparedImage,
    PreparedMedia,
    Quote,
    QuoteSegment,
    collect_forward_asset_ids,
)
from .utils import (
    atomic_write_json,
    random_id,
    read_json,
    rel_image_path,
    rel_media_path,
)

PendingSegment = PendingForwardSegment | PendingQuoteSegment
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CreateQuoteResult:
    quote: Quote
    duplicate: bool = False


class SessionStore:
    def __init__(self, plugin_root: Path, session_key: str):
        self.plugin_root = plugin_root
        self.session_key = session_key
        self.root = plugin_root / GROUPS_DIRNAME / session_key
        self.images_dir = self.root / IMAGES_DIRNAME
        self.media_dir = self.root / MEDIA_DIRNAME
        self.quotes_file = self.root / QUOTES_FILENAME
        self.lock = asyncio.Lock()
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)

    def load_payload(self) -> dict:
        payload = read_json(
            self.quotes_file,
            {
                "schema_version": SCHEMA_VERSION,
                "session_key": self.session_key,
                "quotes": [],
                "images": [],
                "media": [],
            },
        )
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise SubmissionError("语录数据版本不兼容，请检查插件数据目录")
        for key in ("quotes", "images", "media"):
            if not isinstance(payload.get(key), list):
                raise SubmissionError("语录数据或资源索引损坏，请检查插件数据目录")
        return payload

    def load_quotes(self) -> list[Quote]:
        return [Quote.from_dict(item) for item in self.load_payload()["quotes"]]

    def load_assets(self) -> list[ImageAsset]:
        return [ImageAsset.from_dict(item) for item in self.load_payload()["images"]]

    def load_media_assets(self) -> list[MediaAsset]:
        return [MediaAsset.from_dict(item) for item in self.load_payload()["media"]]

    def save_snapshot(
        self, quotes: list[Quote], images: list[ImageAsset], media: list[MediaAsset]
    ) -> None:
        # 一个原子替换同时发布语录和两个索引，避免多文件提交留下半条投稿。
        atomic_write_json(
            self.quotes_file,
            {
                "schema_version": SCHEMA_VERSION,
                "session_key": self.session_key,
                "quotes": [item.to_dict() for item in quotes],
                "images": [item.to_dict() for item in images],
                "media": [item.to_dict() for item in media],
            },
        )

    def image_abs_path(self, file_name: str) -> Path:
        return self.images_dir / file_name

    def media_abs_path(self, file_name: str) -> Path:
        return self.media_dir / file_name


class QuoteRepository:
    def __init__(self, plugin_root: Path):
        self.root = Path(plugin_root)
        self.groups_dir = self.root / GROUPS_DIRNAME
        self.groups_dir.mkdir(parents=True, exist_ok=True)
        self._stores: dict[str, SessionStore] = {}

    def get_store(self, session_key: str) -> SessionStore:
        if (
            not session_key
            or Path(session_key).name != session_key
            or session_key in {".", ".."}
        ):
            raise ValueError("无效会话标识")
        if session_key not in self._stores:
            self._stores[session_key] = SessionStore(self.root, session_key)
        return self._stores[session_key]

    def session_keys(self) -> list[str]:
        return sorted(path.name for path in self.groups_dir.iterdir() if path.is_dir())

    def list_quotes(self, session_key: str) -> list[Quote]:
        return self.get_store(session_key).load_quotes()

    def list_all_quotes(self) -> list[Quote]:
        quotes = [
            quote for key in self.session_keys() for quote in self.list_quotes(key)
        ]
        return sorted(quotes, key=lambda quote: (-quote.created_at, quote.id))

    def find_source(self, session_key: str, source_key: str) -> Quote | None:
        return next(
            (
                quote
                for quote in self.list_quotes(session_key)
                if quote.source_key == source_key
            ),
            None,
        )

    def find_asset(self, session_key: str, asset_id: str) -> ImageAsset | None:
        return next(
            (
                asset
                for asset in self.get_store(session_key).load_assets()
                if asset.asset_id == asset_id
            ),
            None,
        )

    def find_media_asset(self, session_key: str, asset_id: str) -> MediaAsset | None:
        return next(
            (
                asset
                for asset in self.get_store(session_key).load_media_assets()
                if asset.asset_id == asset_id
            ),
            None,
        )

    def asset_path(self, asset: ImageAsset | MediaAsset) -> Path:
        path = (self.root / asset.rel_path).resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file():
            raise SubmissionError("投稿媒体文件缺失，无法完整回放")
        return path

    async def random_quote(self) -> Quote | None:
        # 与上游一致：按条目抽样，不能先抽群再抽语录，否则小群条目会被加权。
        candidates = self.list_all_quotes()
        return secrets.choice(candidates) if candidates else None

    async def random_quotes(
        self, min_count: int = 6, max_count: int = 9
    ) -> list[Quote]:
        if min_count < 1 or max_count < min_count:
            raise ValueError("抽取投稿数量范围无效")
        candidates = self.list_all_quotes()
        total = len(candidates)
        if total < min_count:
            raise SubmissionError(
                f"投稿不足，至少需要 {min_count} 条投稿才能摆酒席（当前 {total} 条）"
            )
        upper = min(max_count, total)
        count = min_count + secrets.randbelow(upper - min_count + 1)
        return secrets.SystemRandom().sample(candidates, count)

    def _quote_locations(self, quote_id: str) -> list[tuple[str, Quote]]:
        return [
            (key, quote)
            for key in self.session_keys()
            for quote in self.list_quotes(key)
            if quote.id == quote_id
        ]

    @staticmethod
    def _asset_references(quote: Quote) -> tuple[Counter[str], Counter[str]]:
        images: Counter[str] = Counter()
        media: Counter[str] = Counter()

        def walk(segments: list[QuoteSegment] | list[ForwardSegment]) -> None:
            for segment in segments:
                if segment.asset_id:
                    if segment.type == "image":
                        images[segment.asset_id] += 1
                    elif segment.type in {"record", "video", "file"}:
                        media[segment.asset_id] += 1
                for node in segment.nodes:
                    walk(node.segments)

        walk(quote.segments)
        for node in quote.forward_nodes:
            walk(node.segments)
        return images, media

    def _cleanup_asset_path(
        self, store: SessionStore, asset: ImageAsset | MediaAsset
    ) -> Path | None:
        relative = Path(asset.rel_path)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        path = self.root / relative
        directory = (
            store.images_dir if isinstance(asset, ImageAsset) else store.media_dir
        )
        resolved = path.resolve()
        expected = directory.resolve()
        if resolved == expected or not resolved.is_relative_to(expected):
            return None
        if not expected.is_relative_to(self.root.resolve()):
            return None
        # 索引只能删除本会话的普通文件；不跟随被替换的目录或文件链接。
        current = self.root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                return None
        return path

    async def delete_quote(self, quote_id: str) -> bool:
        if not quote_id:
            return False
        matches = self._quote_locations(quote_id)
        if not matches:
            return False
        if len(matches) != 1:
            raise SubmissionError("投稿 ID 重复，无法安全删除，请检查投稿数据")
        store = self.get_store(matches[0][0])
        async with store.lock:
            # 等待回放或其他写入后重新确认全局 ID，避免使用锁外的过期快照。
            matches = self._quote_locations(quote_id)
            if not matches:
                return False
            if len(matches) != 1:
                raise SubmissionError("投稿 ID 重复，无法安全删除，请检查投稿数据")
            if matches[0][0] != store.session_key:
                raise SubmissionError("投稿所属会话已变化，请重新选择投稿 ID")
            payload = store.load_payload()
            quotes = [Quote.from_dict(item) for item in payload["quotes"]]
            target = next(quote for quote in quotes if quote.id == quote_id)
            remaining = [quote for quote in quotes if quote.id != quote_id]
            target_images, target_media = self._asset_references(target)
            image_refs: Counter[str] = Counter()
            media_refs: Counter[str] = Counter()
            for quote in remaining:
                images, media = self._asset_references(quote)
                image_refs.update(images)
                media_refs.update(media)

            images = [ImageAsset.from_dict(item) for item in payload["images"]]
            media = [MediaAsset.from_dict(item) for item in payload["media"]]
            removed: list[ImageAsset | MediaAsset] = []
            kept_images: list[ImageAsset] = []
            kept_media: list[MediaAsset] = []
            for image in images:
                if image.asset_id in target_images and not image_refs[image.asset_id]:
                    removed.append(image)
                else:
                    image.ref_count = image_refs[image.asset_id]
                    kept_images.append(image)
            for asset in media:
                if asset.asset_id in target_media and not media_refs[asset.asset_id]:
                    removed.append(asset)
                else:
                    asset.ref_count = media_refs[asset.asset_id]
                    kept_media.append(asset)

            store.save_snapshot(remaining, kept_images, kept_media)
            # 索引提交成功后才允许清理文件。不同资源 ID 也可能指向同一路径。
            try:
                retained_paths = {
                    (self.root / asset.rel_path).resolve()
                    for asset in [*kept_images, *kept_media]
                }
            except Exception:
                logger.exception("投稿已删除，但无法确认保留资源路径，跳过文件清理")
                return True
            for asset in removed:
                try:
                    path = self._cleanup_asset_path(store, asset)
                    if path is None:
                        logger.warning("投稿已删除，跳过不安全的媒体路径：%s", asset.rel_path)
                    elif path.resolve() not in retained_paths:
                        path.unlink(missing_ok=True)
                except Exception:
                    logger.exception("投稿已删除，但媒体清理失败：%s", asset.rel_path)
            return True

    async def create_quote_with_segments(
        self, session_key: str, quote: Quote, segments: list[PendingSegment]
    ) -> CreateQuoteResult:
        return await self._create(session_key, quote, segments=segments)

    async def create_quote_with_forward_nodes(
        self, session_key: str, quote: Quote, nodes: list[PendingForwardNode]
    ) -> CreateQuoteResult:
        return await self._create(session_key, quote, nodes=nodes)

    async def _create(
        self,
        session_key: str,
        quote: Quote,
        *,
        segments: list[PendingSegment] | None = None,
        nodes: list[PendingForwardNode] | None = None,
    ) -> CreateQuoteResult:
        store = self.get_store(session_key)
        async with store.lock:
            payload = store.load_payload()
            quotes = [Quote.from_dict(item) for item in payload["quotes"]]
            existing = next(
                (item for item in quotes if item.source_key == quote.source_key), None
            )
            if existing is not None:
                return CreateQuoteResult(existing, duplicate=True)
            images = [ImageAsset.from_dict(item) for item in payload["images"]]
            media = [MediaAsset.from_dict(item) for item in payload["media"]]
            created_files: list[Path] = []
            with tempfile.TemporaryDirectory(
                prefix=".staging-", dir=store.root
            ) as staging:
                stage = Path(staging)
                try:
                    if nodes is not None:
                        quote.kind = "forward"
                        quote.forward_nodes = self._persist_forward_nodes(
                            store, nodes, images, media, created_files, stage
                        )
                        quote.segments = []
                    else:
                        quote.kind = "standard"
                        saved = self._persist_segments(
                            store, segments or [], images, media, created_files, stage
                        )
                        quote.segments = [
                            QuoteSegment.from_dict(segment.to_dict())
                            for segment in saved
                        ]
                        quote.forward_nodes = []
                    asset_nodes = quote.forward_nodes or [
                        ForwardNode(segments=quote.segments)
                    ]
                    quote.image_ids, quote.media_ids = collect_forward_asset_ids(
                        asset_nodes
                    )
                    quotes.append(quote)
                    store.save_snapshot(quotes, images, media)
                except BaseException:
                    # 索引尚未发布；只删除本次创建的文件，已有投稿不会受到影响。
                    for path in created_files:
                        path.unlink(missing_ok=True)
                    raise
        return CreateQuoteResult(quote)

    def _persist_segments(
        self,
        store: SessionStore,
        segments: list[PendingSegment],
        images: list[ImageAsset],
        media: list[MediaAsset],
        created_files: list[Path],
        stage: Path,
    ) -> list[ForwardSegment]:
        if not segments:
            raise SubmissionError("消息内容为空")
        saved: list[ForwardSegment] = []
        for segment in segments:
            if segment.type == "text":
                saved.append(ForwardSegment(type="text", text=segment.text))
            elif segment.type == "image" and segment.image is not None:
                asset = self._persist_image_asset(
                    store, segment.image, created_files, stage
                )
                images.append(asset)
                saved.append(ForwardSegment(type="image", asset_id=asset.asset_id))
            elif (
                segment.type in {"record", "video", "file"}
                and segment.media is not None
            ):
                asset = self._persist_media_asset(
                    store, segment.media, created_files, stage
                )
                media.append(asset)
                saved.append(ForwardSegment(type=segment.type, asset_id=asset.asset_id))
            elif segment.type == "face":
                saved.append(ForwardSegment(type="face", face_id=segment.face_id))
            elif segment.type == "at":
                saved.append(
                    ForwardSegment(type="at", qq=segment.qq, name=segment.name)
                )
            elif segment.type == "nodes" and segment.nodes:
                nested = self._persist_forward_nodes(
                    store, segment.nodes, images, media, created_files, stage
                )
                saved.append(ForwardSegment(type="nodes", nodes=nested))
            else:
                raise SubmissionError(f"消息段 {segment.type} 无法完整保存")
        return saved

    def _persist_forward_nodes(
        self,
        store: SessionStore,
        nodes: list[PendingForwardNode],
        images: list[ImageAsset],
        media: list[MediaAsset],
        created_files: list[Path],
        stage: Path,
    ) -> list[ForwardNode]:
        if not nodes:
            raise SubmissionError("聊天记录为空")
        return [
            ForwardNode(
                sender_uin=node.sender_uin,
                sender_name=node.sender_name,
                timestamp=node.timestamp,
                segments=self._persist_segments(
                    store, node.segments, images, media, created_files, stage
                ),
            )
            for node in nodes
        ]

    @staticmethod
    def _write_asset(
        content: bytes, path: Path, created_files: list[Path], stage: Path
    ) -> None:
        temporary = stage / path.name
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        created_files.append(path)
        temporary.replace(path)

    def _persist_image_asset(
        self,
        store: SessionStore,
        image: PreparedImage,
        created_files: list[Path],
        stage: Path,
    ) -> ImageAsset:
        file_name = f"{random_id()}{image.extension}"
        self._write_asset(
            image.content, store.image_abs_path(file_name), created_files, stage
        )
        return ImageAsset(
            asset_id=random_id("img_"),
            file_name=file_name,
            rel_path=rel_image_path(store.session_key, file_name),
            sha256=image.sha256,
            dhash=image.dhash,
            width=image.width,
            height=image.height,
            ref_count=1,
            created_at=time(),
        )

    def _persist_media_asset(
        self,
        store: SessionStore,
        media: PreparedMedia,
        created_files: list[Path],
        stage: Path,
    ) -> MediaAsset:
        file_name = f"{random_id()}{media.extension}"
        self._write_asset(
            media.content, store.media_abs_path(file_name), created_files, stage
        )
        return MediaAsset(
            asset_id=random_id("media_"),
            media_type=media.media_type,
            file_name=file_name,
            rel_path=rel_media_path(store.session_key, file_name),
            display_name=media.display_name,
            ref_count=1,
            created_at=time(),
        )
