from __future__ import annotations

import asyncio
import base64
import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image as PILImage

from astrbot_plugin_submit_quotes.message_service import MediaLimits, MessageService
from astrbot_plugin_submit_quotes.napcat_service import NapcatService
from astrbot_plugin_submit_quotes.quote_service import QuoteService
from astrbot_plugin_submit_quotes.store import QuoteRepository
from astrbot_plugin_submit_quotes.tests.fakes import (
    COMPONENTS,
    At,
    Face,
    FakeAPI,
    FakeEvent,
    Plain,
    Reply,
    import_plugin_entry,
)
from astrbot_plugin_submit_quotes.tests.legacy_fixtures import seed_legacy_quote

ENTRY = import_plugin_entry()


def segment(kind: str, **data) -> dict:
    return {"type": kind, "data": data}


def encoded(content: bytes) -> str:
    return "base64://" + base64.b64encode(content).decode()


def png_bytes() -> bytes:
    stream = io.BytesIO()
    PILImage.new("RGB", (2, 3), "blue").save(stream, format="PNG")
    return stream.getvalue()


def forward(content: list) -> dict:
    return segment(
        "forward",
        messages=[
            {
                "sender": {"user_id": "34567", "nickname": "作者"},
                "time": 1234,
                "content": content,
            }
        ],
    )


class ManagementAndBanquetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = QuoteRepository(self.root)
        self.client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(404))
        )
        self.napcat = NapcatService()
        self.service = QuoteService(
            self.repository,
            MessageService(self.client, self.napcat, MediaLimits()),
            self.napcat,
            COMPONENTS,
        )
        self.plugin = ENTRY.SubmitQuotesPlugin.__new__(ENTRY.SubmitQuotesPlugin)
        self.plugin.repository = self.repository
        self.plugin.quote_service = self.service
        self.plugin.http_client = self.client
        self.plugin.config = {"delete_whitelist": ["12345"]}
        self.message_counter = 0

    async def asyncTearDown(self):
        await self.client.aclose()
        self.temporary.cleanup()

    async def add(self, content, *, group="100", created_at=None):
        self.message_counter += 1

        def get_message(action, params):
            if action == "get_msg":
                return {
                    "data": {
                        "message": content,
                        "sender": {"user_id": "23456", "nickname": "原作者"},
                    }
                }
            raise RuntimeError("不应请求额外媒体接口")

        event = FakeEvent(
            [Reply(self.message_counter), Plain("投稿")],
            group=group,
            api=FakeAPI(get_message),
        )
        if created_at is not None:
            with patch(
                "astrbot_plugin_submit_quotes.quote_service.time",
                return_value=created_at,
            ):
                return await self.service.add_quote(event)
        return await self.service.add_quote(event)

    async def seed_text(self, count=6):
        return [
            await self.add([segment("text", text=f"投稿内容{index}")])
            for index in range(count)
        ]

    async def add_legacy(self, content):
        self.message_counter += 1
        event = FakeEvent([Reply(self.message_counter), Plain("投稿")])
        return await seed_legacy_quote(self.service, event, content)

    async def command(self, text, **options):
        event = FakeEvent([Plain(text)], **options)
        await self.plugin.on_message(event)
        return event

    @staticmethod
    def sent_text(event):
        return "".join(
            part["data"]["text"]
            for action, params in event.bot.api.calls
            if action in {"send_group_msg", "send_private_msg"}
            for part in params["message"]
            if part["type"] == "text"
        )

    @staticmethod
    def catalog_text(event):
        return "\n".join(
            part["data"]["text"]
            for action, params in event.bot.api.calls
            if action in {"send_group_forward_msg", "send_private_forward_msg"}
            for node in params["messages"]
            for part in node["data"]["content"]
            if part["type"] == "text"
        )

    async def test_whitelist_config_accepts_user_id_lists_and_rejects_invalid_values(self):
        self.assertEqual(ENTRY.parse_delete_whitelist({}), frozenset())
        self.assertEqual(
            ENTRY.parse_delete_whitelist(
                {"delete_whitelist": [12345, "23456", " 34567 ", "12345"]}
            ),
            frozenset({"12345", "23456", "34567"}),
        )
        for value in (
            "12345",
            None,
            {"12345": True},
            [True],
            [False],
            [1.5],
            [None],
            [""],
            ["123 456"],
            ["１２３４５"],
            ["not-a-user"],
            [0],
            ["000"],
            [-12345],
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ENTRY.SubmitQuotesPlugin(None, {"delete_whitelist": value})

    async def test_delete_is_disabled_by_default_without_reading_the_pool(self):
        self.plugin.config = {}
        with patch.object(
            self.repository,
            "list_all_quotes",
            side_effect=AssertionError("未授权请求不能读取投稿"),
        ) as listing:
            event = await self.command("删除投稿")
        self.assertTrue(event.stopped)
        self.assertEqual(len(event.sent), 1)
        self.assertIn("白名单", event.sent[0])
        listing.assert_not_called()

    async def test_unlisted_sender_cannot_list_or_delete(self):
        quote = await self.add([segment("text", text="保留")])
        for command in ("删除投稿", f"删除投稿 {quote.id}"):
            with self.subTest(command=command), patch.object(
                self.repository,
                "list_all_quotes",
                side_effect=AssertionError("越权请求不能读取投稿"),
            ) as listing, patch.object(
                self.repository,
                "delete_quote",
                side_effect=AssertionError("越权请求不能删除投稿"),
            ) as deletion:
                event = await self.command(command, sender="55555")
            self.assertTrue(event.stopped)
            self.assertIn("白名单", event.sent[0])
            listing.assert_not_called()
            deletion.assert_not_called()
        self.assertEqual(len(self.repository.list_all_quotes()), 1)

    async def test_listing_is_global_and_private_deletion_removes_other_group_quote(self):
        quotes = [
            await self.add([segment("text", text="甲群")], group="100"),
            await self.add([segment("text", text="乙群")], group="200"),
            await self.add([segment("text", text="私聊")], group=""),
        ]
        listing = await self.command("删除投稿", group="300")
        self.assertTrue(listing.stopped)
        self.assertEqual(len(listing.bot.api.sent), 1)
        for quote in quotes:
            self.assertIn(quote.id, self.catalog_text(listing))
            self.assertIn(quote.text, self.catalog_text(listing))
        event = await self.command(f"删除投稿 {quotes[0].id}", group="")
        self.assertTrue(event.stopped)
        self.assertIn("已删除", event.sent[0])
        self.assertNotIn(
            quotes[0].id, {quote.id for quote in self.repository.list_all_quotes()}
        )

    async def test_numeric_whitelist_entries_authorize_the_same_qq(self):
        self.plugin.config = {"delete_whitelist": [12345]}
        quote = await self.add([segment("text", text="数字白名单")])
        event = await self.command("删除投稿")
        self.assertIn(quote.id, self.catalog_text(event))

    async def test_revoking_whitelist_takes_effect_before_a_listed_id_is_deleted(self):
        quote = await self.add([segment("text", text="撤销后保留")])
        listing = await self.command("删除投稿")
        self.assertIn(quote.id, self.catalog_text(listing))
        self.plugin.config["delete_whitelist"] = []
        with patch.object(self.repository, "delete_quote") as deletion:
            event = await self.command(f"删除投稿 {quote.id}")
        self.assertTrue(event.stopped)
        self.assertIn("白名单", event.sent[0])
        deletion.assert_not_called()
        self.assertEqual(self.repository.list_all_quotes()[0].id, quote.id)

    async def test_listing_pages_ten_rows_in_newest_first_order(self):
        quotes = [
            await self.add(
                [segment("text", text=f"预览{index}")], created_at=float(index)
            )
            for index in range(12)
        ]
        first = await self.command("删除投稿")
        second = await self.command("删除投稿 列表 2")
        for event, expected in (
            (first, list(reversed(quotes[2:]))),
            (second, list(reversed(quotes[:2]))),
        ):
            self.assertTrue(event.stopped)
            text = self.catalog_text(event)
            self.assertEqual(re.findall(r"q_[A-Za-z0-9_-]+", text), [
                quote.id for quote in expected
            ])

    async def test_preview_of_media_only_quote_is_useful_and_text_preview_is_bounded(self):
        media = await self.add([segment("image", file=encoded(png_bytes()))])
        long_text = await self.add([segment("text", text="长内容" * 500)])
        event = await self.command("删除投稿")
        text = self.catalog_text(event)
        self.assertIn(media.id, text)
        self.assertIn("图片", text)
        self.assertIn(long_text.id, text)
        self.assertLess(len(text), 1000)
        images = [
            part
            for node in event.bot.api.sent[0]
            for part in node["data"]["content"]
            if part["type"] == "image"
        ]
        self.assertEqual(len(images), 1)

    async def test_empty_pool_listing_is_still_a_handled_command(self):
        event = await self.command("删除投稿")
        self.assertTrue(event.stopped)
        self.assertEqual(len(event.sent), 1)
        self.assertIn("投稿", event.sent[0])
        self.assertEqual(event.bot.api.calls, [])

    async def test_bad_parameters_never_delete_anything(self):
        quote = await self.add([segment("text", text="保留")])
        for text in (
            "删除投稿 列表 0",
            "删除投稿 列表 -1",
            "删除投稿 列表 二",
            "删除投稿 列表 1 多余",
            f"删除投稿 {quote.id} {quote.id}",
        ):
            with self.subTest(text=text), patch.object(
                self.repository,
                "delete_quote",
                side_effect=AssertionError("非法参数不可删除"),
            ) as deletion:
                event = await self.command(text)
            self.assertTrue(event.stopped)
            self.assertIn("删除投稿", event.sent[0])
            self.assertNotIn("已删除", event.sent[0])
            deletion.assert_not_called()
        self.assertEqual(len(self.repository.list_all_quotes()), 1)

    async def test_unknown_and_partial_ids_do_not_match_existing_record(self):
        quote = await self.add([segment("text", text="保留")])
        for value in (quote.id[:-1], "q_unknown_id"):
            with self.subTest(value=value):
                event = await self.command(f"删除投稿 {value}")
            self.assertTrue(event.stopped)
            self.assertNotIn("已删除", event.sent[0])
        self.assertEqual(self.repository.list_all_quotes()[0].id, quote.id)

    async def test_bare_listed_id_is_not_a_delete_command(self):
        quote = await self.add([segment("text", text="保留")])
        event = await self.command(quote.id)
        self.assertFalse(event.stopped)
        self.assertEqual(event.sent, [])
        self.assertEqual(self.repository.list_all_quotes()[0].id, quote.id)

    async def test_failed_snapshot_never_reports_delete_success(self):
        quote = await self.add([segment("text", text="保留")])
        snapshot = self.repository.get_store(quote.group).quotes_file
        before = snapshot.read_bytes()
        with patch.object(
            self.repository.get_store(quote.group),
            "save_snapshot",
            side_effect=OSError("disk full"),
        ), self.assertLogs(ENTRY.logger, level="ERROR"):
            event = await self.command(f"删除投稿 {quote.id}")
        self.assertTrue(event.stopped)
        self.assertIn("失败", event.sent[0])
        self.assertNotIn("已删除", event.sent[0])
        self.assertEqual(snapshot.read_bytes(), before)

    async def test_banquet_outputs_six_to_nine_distinct_quotes_in_one_text_message(self):
        await self.seed_text(12)
        event = await self.command(" \n摆酒席\t ")
        self.assertTrue(event.stopped)
        self.assertEqual(event.sent, [])
        self.assertEqual(len(event.bot.api.sent), 1)
        lines = self.sent_text(event).splitlines()
        self.assertGreaterEqual(len(lines), 6)
        self.assertLessEqual(len(lines), 9)
        options = []
        for index, line in enumerate(lines, 1):
            self.assertTrue(line.startswith(f"第{index}道菜："))
            options.append(line.split("：", 1)[1])
        self.assertEqual(len(options), len(set(options)))
        self.assertTrue(all(value.startswith("投稿内容") for value in options))

    async def test_exactly_six_quotes_from_other_groups_and_private_can_make_a_banquet(self):
        for group in ("100", "200", ""):
            for index in range(2):
                await self.add(
                    [segment("text", text=f"{group or '私聊'}投稿{index}")],
                    group=group,
                )
        event = await self.command("摆酒席", group="", sender="45678")
        self.assertEqual(len(self.sent_text(event).splitlines()), 6)
        self.assertEqual(event.bot.api.calls[0][0], "send_private_msg")
        self.assertEqual(event.bot.api.calls[0][1]["user_id"], 45678)

    async def test_too_few_quotes_produce_a_hint_and_no_partial_dishes(self):
        for count in (0, 5):
            if count:
                await self.seed_text(count)
            with self.subTest(count=count):
                event = await self.command("摆酒席")
            self.assertTrue(event.stopped)
            self.assertEqual(len(event.sent), 1)
            self.assertIn("6", event.sent[0])
            self.assertEqual(event.bot.api.calls, [])

    async def test_banquet_preserves_image_face_at_and_original_text_order(self):
        first = await self.add_legacy(
            [
                segment("text", text="  前\n"),
                segment("image", file=encoded(png_bytes())),
                segment("face", id="14"),
                segment("at", qq="23456"),
                segment("text", text="紧接@  "),
            ]
        )
        quotes = [first, *await self.seed_text(5)]
        with patch.object(self.repository, "random_quotes", return_value=quotes):
            event = await self.command("摆酒席")
        self.assertEqual(len(event.bot.api.sent), 1)
        content = event.bot.api.sent[0]
        types = [part["type"] for part in content]
        self.assertEqual([kind for kind in types if kind != "text"], [
            "image", "face", "at"
        ])
        image_index = types.index("image")
        before_image = "".join(part["data"]["text"] for part in content[:image_index])
        self.assertEqual(before_image, "第1道菜：  前\n")
        at_index = types.index("at")
        self.assertEqual(content[at_index]["data"]["qq"], "23456")
        self.assertTrue(content[at_index + 1]["data"]["text"].startswith("紧接@  "))

    async def test_banquet_sends_voice_video_file_and_forward_alone_in_order(self):
        first = await self.add_legacy(
            [
                segment("text", text="前"),
                segment("record", file=encoded(b"audio")),
                segment("text", text="中"),
                segment("video", file=encoded(b"video")),
                segment("file", file=encoded(b"file"), name="材料.txt"),
                segment("text", text="后"),
            ]
        )
        second = await self.add([forward([segment("text", text="记录原文")])])
        quotes = [first, second, *await self.seed_text(4)]
        with patch.object(self.repository, "random_quotes", return_value=quotes):
            event = await self.command("摆酒席")
        isolated_kinds = []
        for action, params in event.bot.api.calls:
            if action == "send_group_forward_msg":
                isolated_kinds.append("forward")
                self.assertEqual(
                    params["messages"][0]["data"]["content"][0]["data"]["text"],
                    "记录原文",
                )
                continue
            content = params["message"]
            kinds = [part["type"] for part in content]
            for kind in ("record", "video", "file"):
                if kind in kinds:
                    self.assertEqual(kinds, [kind])
                    isolated_kinds.append(kind)
        self.assertEqual(isolated_kinds, ["record", "video", "file", "forward"])
        text = self.sent_text(event)
        self.assertLess(text.index("第1道菜：前"), text.index("中"))
        self.assertLess(text.index("中"), text.index("后"))
        self.assertLess(text.index("后"), text.index("第2道菜："))
        for number in range(1, 7):
            self.assertEqual(text.count(f"第{number}道菜："), 1)

    async def test_missing_last_dish_media_prevents_sending_even_first_dish(self):
        quotes = await self.seed_text(5)
        last = await self.add([segment("image", file=encoded(png_bytes()))])
        asset = self.repository.find_asset(last.group, last.image_ids[0])
        self.repository.asset_path(asset).unlink()
        with patch.object(
            self.repository, "random_quotes", return_value=[*quotes, last]
        ):
            event = await self.command("摆酒席")
        self.assertTrue(event.stopped)
        self.assertEqual(event.bot.api.calls, [])
        self.assertEqual(len(event.sent), 1)
        self.assertIn("缺失", event.sent[0])

    async def test_new_command_exact_trigger_and_bot_platform_guards(self):
        for message in (
            [Plain("/摆酒席")],
            [Plain("请摆酒席")],
            [Plain("摆酒席 更多")],
            [At("99999"), Plain("摆酒席")],
            [Plain("摆酒席"), Face(1)],
            [Plain("/删除投稿")],
            [Plain("请删除投稿")],
            [At("99999"), Plain("删除投稿")],
        ):
            with self.subTest(message=message):
                event = FakeEvent(message)
                await self.plugin.on_message(event)
            self.assertFalse(event.stopped)
            self.assertEqual(event.sent, [])
            self.assertEqual(event.bot.api.calls, [])
        for command in ("摆酒席", "删除投稿"):
            for options in ({"sender": "99999"}, {"platform": "other"}):
                with self.subTest(command=command, options=options):
                    event = await self.command(command, **options)
                self.assertFalse(event.stopped)
                self.assertEqual(event.sent, [])
                self.assertEqual(event.bot.api.calls, [])

    async def test_deleted_selection_cannot_be_replayed_after_removal(self):
        quote = await self.add([segment("text", text="不再发送")])
        await self.repository.delete_quote(quote.id)
        event = FakeEvent([Plain("语录")])
        with patch.object(self.repository, "random_quote", return_value=quote):
            await self.plugin.on_message(event)
        self.assertEqual(event.bot.api.calls, [])
        self.assertEqual(len(event.sent), 1)
        self.assertIn("失败", event.sent[0])

    async def test_deleted_last_selection_prevents_replaying_any_banquet_dishes(self):
        quotes = await self.seed_text(6)
        await self.repository.delete_quote(quotes[-1].id)
        with patch.object(self.repository, "random_quotes", return_value=quotes):
            event = await self.command("摆酒席")
        self.assertTrue(event.stopped)
        self.assertEqual(event.bot.api.calls, [])
        self.assertEqual(len(event.sent), 1)
        self.assertIn("失败", event.sent[0])

    async def test_deletion_waits_for_current_replay_to_finish(self):
        quote = await self.add([segment("text", text="发送中")])
        event = FakeEvent([Plain("语录")])
        sending = asyncio.Event()
        finish = asyncio.Event()

        async def send(action, **params):
            sending.set()
            await finish.wait()
            return {"status": "ok", "retcode": 0}

        with patch.object(event.bot.api, "call_action", side_effect=send):
            replay = asyncio.create_task(self.service.send_quote(event, quote))
            await asyncio.wait_for(sending.wait(), timeout=1)
            deletion = asyncio.create_task(self.repository.delete_quote(quote.id))
            await asyncio.sleep(0)
            self.assertFalse(deletion.done())
            finish.set()
            await asyncio.wait_for(replay, timeout=1)
            self.assertTrue(await asyncio.wait_for(deletion, timeout=1))

