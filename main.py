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
from .napcat_service import NapcatService
from .quote_service import EMPTY_MESSAGE, RECEIVED_MESSAGE, QuoteService
from .store import QuoteRepository
from .utils import ensure_plugin_data_dir


class SubmitQuotesPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
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
        return text if text in {"投稿", "语录"} else None

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
        if trigger == "投稿":
            try:
                await self.quote_service.add_quote(event)
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

    async def terminate(self):
        await self.http_client.aclose()
