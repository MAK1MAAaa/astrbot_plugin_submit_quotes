"""Offline, explicit video-submission cleanup for this plugin's schema4 data."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import IMAGES_DIRNAME, MEDIA_DIRNAME, QUOTES_FILENAME, SCHEMA_VERSION
from .errors import SubmissionError
from .models import ImageAsset, MediaAsset, Quote
from .store import QuoteRepository

_SEGMENT_TYPES = {"text", "image", "at", "face", "record", "video", "file", "nodes"}
_MEDIA_TYPES = {"record", "video", "file"}


class _PrivateCleanupLog(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.had_issue = False

    def filter(self, record: logging.LogRecord) -> bool:
        self.had_issue = self.had_issue or record.levelno >= logging.WARNING
        record.msg = "投稿记录已删除，但部分资源清理失败；请检查文件权限。"
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def _invalid() -> SubmissionError:
    # 运维输出不包含投稿正文、群号或用户身份。
    return SubmissionError(
        "投稿数据未知或损坏；未完成清理，请先检查并恢复有效的 schema4 数据"
    )


def _optional_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise _invalid()
    return value


def _string_ids(data: dict[str, Any], key: str) -> list[str]:
    values = _optional_list(data, key)
    if any(not isinstance(value, str) or not value for value in values):
        raise _invalid()
    return values


def _validate_asset_path(
    repository: QuoteRepository, session: Path, asset: dict, directory: str
) -> None:
    value = asset.get("rel_path")
    if not isinstance(value, str) or not value or "\x00" in value:
        raise _invalid()
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise _invalid()
    path = repository.root / relative
    expected = (session / directory).resolve()
    if not expected.is_relative_to(repository.root.resolve()):
        raise _invalid()
    if path.resolve() == expected or not path.resolve().is_relative_to(expected):
        raise _invalid()
    current = repository.root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise _invalid()


def _inspect_tree(data: dict[str, Any]) -> tuple[bool, set[str], set[str], set[str]]:
    has_video = False
    images: set[str] = set()
    media: set[str] = set()
    explicit_nonvideo: set[str] = set()

    def walk_segments(segments: list[Any]) -> None:
        nonlocal has_video
        for segment in segments:
            if (
                not isinstance(segment, dict)
                or segment.get("type") not in _SEGMENT_TYPES
            ):
                raise _invalid()
            kind = segment["type"]
            asset_id = segment.get("asset_id") or ""
            if not isinstance(asset_id, str):
                raise _invalid()
            if kind == "video":
                has_video = True
            if asset_id:
                if kind == "image":
                    images.add(asset_id)
                elif kind in _MEDIA_TYPES:
                    media.add(asset_id)
                    if kind != "video":
                        explicit_nonvideo.add(asset_id)
            nodes = _optional_list(segment, "nodes")
            if nodes and kind != "nodes":
                raise _invalid()
            walk_nodes(nodes)

    def walk_nodes(nodes: list[Any]) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                raise _invalid()
            walk_segments(_optional_list(node, "segments"))

    walk_segments(_optional_list(data, "segments"))
    walk_nodes(_optional_list(data, "forward_nodes"))
    return has_video, images, media, explicit_nonvideo


def scan_video_quotes(repository: QuoteRepository) -> list[Quote]:
    """Validate all snapshots before selecting entire video-containing quotes.

    Missing message trees in older schema4 records are supported through media_ids.
    File/voice segments remain authoritative when stale media metadata disagrees.
    """
    candidates: list[Quote] = []
    ids: Counter[str] = Counter()
    try:
        if repository.groups_dir.is_symlink():
            raise _invalid()
        for key in repository.session_keys():
            session = repository.groups_dir / key
            snapshot = session / QUOTES_FILENAME
            if session.is_symlink() or snapshot.is_symlink():
                raise _invalid()
            if not snapshot.exists():
                # 保存失败会留下空会话目录；有资源却缺少索引时不能猜测归属。
                if any(
                    path.is_file() or path.is_symlink() for path in session.rglob("*")
                ):
                    raise _invalid()
                continue
            payload = json.loads(snapshot.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != SCHEMA_VERSION
            ):
                raise _invalid()
            if payload.get("session_key", key) != key:
                raise _invalid()
            if any(
                not isinstance(payload.get(name), list)
                for name in ("quotes", "images", "media")
            ):
                raise _invalid()
            image_ids: set[str] = set()
            media_types: dict[str, str] = {}
            for entry in payload["images"]:
                if not isinstance(entry, dict):
                    raise _invalid()
                image = ImageAsset.from_dict(entry)
                if not image.asset_id or image.asset_id in image_ids:
                    raise _invalid()
                image_ids.add(image.asset_id)
                _validate_asset_path(repository, session, entry, IMAGES_DIRNAME)
            for entry in payload["media"]:
                if not isinstance(entry, dict):
                    raise _invalid()
                asset = MediaAsset.from_dict(entry)
                if (
                    not asset.asset_id
                    or asset.asset_id in media_types
                    or asset.asset_id in image_ids
                    or asset.media_type not in _MEDIA_TYPES
                ):
                    raise _invalid()
                media_types[asset.asset_id] = asset.media_type
                _validate_asset_path(repository, session, entry, MEDIA_DIRNAME)
            for entry in payload["quotes"]:
                if not isinstance(entry, dict):
                    raise _invalid()
                quote_id = entry.get("id")
                if not isinstance(quote_id, str) or not quote_id:
                    raise _invalid()
                if entry.get("kind", "") not in {"", "standard", "forward"}:
                    raise _invalid()
                quote = Quote.from_dict(entry)
                if not math.isfinite(quote.created_at):
                    raise _invalid()
                ids[quote.id] += 1
                has_video, tree_images, tree_media, explicit_nonvideo = _inspect_tree(
                    entry
                )
                stored_images = set(_string_ids(entry, "image_ids"))
                stored_media = set(_string_ids(entry, "media_ids"))
                if not (tree_images | stored_images).issubset(image_ids):
                    raise _invalid()
                if not (tree_media | stored_media).issubset(media_types):
                    raise _invalid()
                indexed_video = any(
                    media_types[asset_id] == "video"
                    and asset_id not in explicit_nonvideo
                    for asset_id in stored_media
                )
                if has_video or indexed_video:
                    candidates.append(quote)
        if any(count != 1 for count in ids.values()):
            raise SubmissionError("投稿 ID 重复；为避免误删，未执行任何清理")
        return sorted(candidates, key=lambda quote: (-quote.created_at, quote.id))
    except SubmissionError:
        raise
    except Exception as exc:
        raise _invalid() from exc


async def remove_video_quotes(repository: QuoteRepository) -> list[Quote]:
    """Remove exact records after complete preflight; propagate any write failure."""
    candidates = scan_video_quotes(repository)
    deleted: list[Quote] = []
    for quote in candidates:
        if not await repository.delete_quote(quote.id):
            raise SubmissionError("清理期间投稿已变化；操作未全部完成，请重新扫描")
        deleted.append(quote)
    return deleted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="扫描或删除本插件全部含视频的投稿")
    parser.add_argument("--root", type=Path, required=True, help="本插件的数据目录")
    parser.add_argument("--apply", action="store_true", help="执行删除；默认仅预览")
    arguments = parser.parse_args(argv)
    try:
        root = arguments.root.expanduser().resolve()
        if not root.is_dir() or not (root / "groups").is_dir():
            raise SubmissionError("插件数据目录不存在或缺少 groups；未执行清理")
        repository = QuoteRepository(root)
        cleanup_logger = logging.getLogger(QuoteRepository.__module__)
        privacy_filter = _PrivateCleanupLog()
        cleanup_logger.addFilter(privacy_filter)
        try:
            selected = (
                asyncio.run(remove_video_quotes(repository))
                if arguments.apply
                else scan_video_quotes(repository)
            )
        finally:
            cleanup_logger.removeFilter(privacy_filter)
        if privacy_filter.had_issue:
            print(
                "投稿记录已删除，但资源文件未全部清理；请检查文件权限和备份。",
                file=sys.stderr,
            )
            return 1
        print(
            json.dumps(
                {
                    "mode": "apply" if arguments.apply else "dry-run",
                    "count": len(selected),
                    "quote_ids": [quote.id for quote in selected],
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception:
        print(
            "清理未全部完成；数据校验或写入失败。请检查备份和数据后重新扫描。",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
