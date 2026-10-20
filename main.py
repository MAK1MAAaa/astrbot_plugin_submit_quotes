from __future__ import annotations

import asyncio

import httpx
from astrbot.api import AstrBotConfig, logger
from astrbot.api import message_components as Comp
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from .constants import PLUGIN_NAME
from .errors import SubmissionError
from .message_service import MediaLimits, MessageService
from .models import ForwardSegment, Quote, QuoteSegment
from .moderation import (
    REJECTED_MESSAGE,
    ContentModerator,
    ModerationConfig,
    ModerationRejected,
)
from .napcat_service import NapcatService
from .quote_service import EMPTY_MESSAGE, RECEIVED_MESSAGE, QuoteService
from .store import QuoteRepository
from .utils import ensure_plugin_data_dir


def parse_delete_whitelist(config: AstrBotConfig) -> frozenset[str]:
    users = config.get("delete_whitelist", [])
    if not isinstance(users, list):
        raise ValueError("delete_whitelist 必须为用户 ID 列表")
    result: set[str] = set()
    for user in users:
        if isinstance(user, bool) or not isinstance(user, (str, int)):
            raise ValueError("delete_whitelist 只能包含有效的 QQ 用户 ID")
        value = str(user).strip()
        if not value.isascii() or not value.isdigit() or not value.strip("0"):
            raise ValueError("delete_whitelist 只能包含有效的 QQ 用户 ID")
        result.add(value)
    return frozenset(result)


def quote_preview(quote: Quote) -> str:
    parts: list[str] = []

    def walk(segments: list[QuoteSegment | ForwardSegment]) -> None:
        for segment in segments:
            if segment.type == "text":
                parts.append(segment.text[:80])
            elif segment.type == "nodes":
                parts.append("[聊天记录]")
                for node in segment.nodes:
                    walk(node.segments)
            elif segment.type == "at":
                parts.append("@全体成员" if segment.qq == "all" else f"@{segment.qq}")
            else:
                parts.append(
                    {
                        "image": "[图片]",
                        "record": "[语音]",
                        "video": "[视频]",
                        "file": "[文件]",
                        "face": "[QQ表情]",
                    }.get(segment.type, "[未知类型]")
                )
            if sum(map(len, parts)) >= 80:
                return

    if quote.kind == "forward":
        for node in quote.forward_nodes:
            walk(node.segments)
            if sum(map(len, parts)) >= 80:
                break
    else:
        walk(quote.segments)
    preview = " ".join("".join(parts).split()) or "[空内容]"
    return preview[:79] + "…" if len(preview) > 80 else preview


class SubmitQuotesPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        parse_delete_whitelist(config)
        moderation_config = ModerationConfig.from_config(config)
        self.config = config
        limits = MediaLimits.from_config(config)
        self.repository = QuoteRepository(
            ensure_plugin_data_dir(plugin_name=PLUGIN_NAME)
        )
        self.http_client = httpx.AsyncClient(
            timeout=limits.timeout_seconds, follow_redirects=True
        )
        napcat = NapcatService()
        self.quote_service = QuoteService(
            self.repository,
            MessageService(self.http_client, napcat, limits),
            napcat,
            Comp,
            moderator=ContentModerator(context, moderation_config),
        )

    @staticmethod
    def _trigger(event: AstrMessageEvent) -> str | None:
        parts: list[str] = []
        has_reply = False
        has_at = False
        for segment in event.get_messages():
            if isinstance(segment, Comp.Plain):
                parts.append(segment.text)
            elif isinstance(segment, Comp.Reply):
                has_reply = True
            elif isinstance(segment, Comp.At):
                has_at = True
            else:
                return None
        text = "".join(parts).strip()
        if has_at and not (text == "投稿" and has_reply):
            return None
        if text in {"投稿", "语录", "摆酒席", "删除投稿"}:
            return text
        if text and text.split(maxsplit=1)[0] == "删除投稿":
            return text
        return None

    @filter.event_message_type(filter.EventMessageType.ALL, priority=10)
    async def on_message(self, event: AstrMessageEvent):
        if event.get_platform_name() != "aiocqhttp":
            return
        self_id = str(getattr(event.message_obj, "self_id", "") or "")
        if self_id and str(event.get_sender_id()) == self_id:
            return
        trigger = self._trigger(event)
        if trigger is None:
            return
        # 命中后始终终止本次事件，包括保存失败，避免 LLM 再回复一次。
        event.stop_event()
        if trigger.startswith("删除投稿"):
            await self._delete_quote(event, trigger)
            return
        if trigger == "摆酒席":
            try:
                quotes = await self.repository.random_quotes()
                await self.quote_service.send_banquet(event, quotes)
            except SubmissionError as exc:
                await event.send(event.plain_result(f"摆酒席失败：{exc}。"))
            except Exception:
                logger.exception("摆酒席回放失败")
                await event.send(event.plain_result("摆酒席失败，请检查插件日志。"))
            return
        if trigger == "投稿":
            try:
                await self.quote_service.add_quote(event)
            except ModerationRejected:
                await event.send(event.plain_result(REJECTED_MESSAGE))
            except SubmissionError as exc:
                await event.send(event.plain_result(f"投稿失败：{exc}。"))
            except asyncio.TimeoutError:
                await event.send(event.plain_result("投稿失败：合并转发获取超时。"))
            except Exception:
                logger.exception("投稿完整保存失败")
                await event.send(
                    event.plain_result("投稿失败：保存失败，请检查磁盘空间和插件日志。")
                )
            else:
                await event.send(event.plain_result(RECEIVED_MESSAGE))
            return
        try:
            quote = await self.repository.random_quote()
            if quote is None:
                await event.send(event.plain_result(EMPTY_MESSAGE))
                return
            await self.quote_service.send_quote(event, quote)
        except SubmissionError as exc:
            await event.send(event.plain_result(f"语录发送失败：{exc}。"))
        except Exception:
            logger.exception("语录回放失败")
            await event.send(event.plain_result("语录发送失败，请检查插件日志。"))

    async def _delete_quote(self, event: AstrMessageEvent, command: str) -> None:
        try:
            # 每次执行都检查当前配置，撤销白名单后旧列表不再授予删除权限。
            if str(event.get_sender_id()) not in parse_delete_whitelist(
                getattr(self, "config", {})
            ):
                await event.send(
                    event.plain_result("你不在删除投稿白名单中，无法查看或删除投稿。")
                )
                return
            args = command.split()[1:]
            if not args or args == ["列表"] or (
                len(args) == 2 and args[0] == "列表"
            ):
                page_text = args[1] if len(args) == 2 else "1"
                if (
                    not page_text.isascii()
                    or not page_text.isdigit()
                    or len(page_text) > 9
                    or int(page_text) < 1
                ):
                    await event.send(event.plain_result(self._delete_usage()))
                    return
                quotes = self.repository.list_all_quotes()
                if not quotes:
                    await event.send(event.plain_result("还没有投稿可以删除。"))
                    return
                page = int(page_text)
                pages = (len(quotes) + 9) // 10
                if page > pages:
                    await event.send(event.plain_result(f"页码超出范围，共 {pages} 页。"))
                    return
                header = f"投稿列表（第 {page}/{pages} 页，共 {len(quotes)} 条）："
                instructions = ["删除指定记录：删除投稿 <完整ID>"]
                if page < pages:
                    instructions.append(f"查看下一页：删除投稿 列表 {page + 1}")
                await self.quote_service.send_catalog(
                    event,
                    [
                        (quote, quote_preview(quote))
                        for quote in quotes[(page - 1) * 10 : page * 10]
                    ],
                    header=header,
                    footer="\n".join(instructions),
                )
                return
            if len(args) != 1:
                await event.send(event.plain_result(self._delete_usage()))
                return
            quote_id = args[0]
            deleted = await self.repository.delete_quote(quote_id)
            result = (
                f"已删除投稿：{quote_id}"
                if deleted
                else "未找到该投稿，请发送“删除投稿”查看当前列表，并使用完整 ID。"
            )
            await event.send(event.plain_result(result))
        except SubmissionError as exc:
            await event.send(event.plain_result(f"删除投稿失败：{exc}。"))
        except Exception:
            logger.exception("投稿列表读取或删除失败")
            await event.send(event.plain_result("删除投稿失败，请检查插件日志。"))

    @staticmethod
    def _delete_usage() -> str:
        return "用法：删除投稿；删除投稿 列表 <页码>；删除投稿 <完整ID>。"

    async def terminate(self):
        await self.http_client.aclose()
