from __future__ import annotations

import asyncio
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image as PILImage

from astrbot_plugin_submit_quotes.message_service import MediaLimits, MessageService
from astrbot_plugin_submit_quotes.napcat_service import NapcatService
from astrbot_plugin_submit_quotes.quote_service import (
    EMPTY_MESSAGE,
    RECEIVED_MESSAGE,
    QuoteService,
)
from astrbot_plugin_submit_quotes.store import QuoteRepository
from astrbot_plugin_submit_quotes.tests.fakes import (
    COMPONENTS,
    At,
    Face,
    FakeAPI,
    FakeEvent,
    File,
    Image,
    Nodes,
    Plain,
    Record,
    Reply,
    Video,
    import_plugin_entry,
)

ENTRY = import_plugin_entry()


def segment(kind: str, **data) -> dict:
    return {"type": kind, "data": data}


def encoded(content: bytes) -> str:
    return "base64://" + base64.b64encode(content).decode()


def png_bytes() -> bytes:
    stream = io.BytesIO()
    PILImage.new("RGB", (2, 3), "blue").save(stream, format="PNG")
    return stream.getvalue()


def forward(content: list, *, nickname="作者", time=1234) -> dict:
    return segment(
        "forward",
        messages=[
            {
                "sender": {"user_id": "34567", "nickname": nickname},
                "time": time,
                "content": content,
            }
        ],
    )


class SubmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = QuoteRepository(self.root)
        self.requests = []

        async def download(request):
            self.requests.append(request.url)
            return httpx.Response(404)

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(download))
        self.napcat = NapcatService()
        self.message_service = MessageService(self.client, self.napcat, MediaLimits())
        self.service = QuoteService(
            self.repository, self.message_service, self.napcat, COMPONENTS
        )
        self.plugin = ENTRY.SubmitQuotesPlugin.__new__(ENTRY.SubmitQuotesPlugin)
        self.plugin.repository, self.plugin.quote_service = (
            self.repository,
            self.service,
        )
        self.plugin.http_client = self.client

    async def asyncTearDown(self):
        await self.client.aclose()
        self.temporary.cleanup()

    def submission(self, content, *, message_id="-123", **event_options):
        def action(name, params):
            if name == "get_msg":
                return {
                    "status": "ok",
                    "retcode": 0,
                    "message": "",
                    "data": {
                        "message": content,
                        "sender": {"user_id": "23456", "nickname": "原作者"},
                    },
                }
            raise RuntimeError("模拟媒体接口不可用")

        return FakeEvent(
            [Reply(message_id), Plain("投稿")], api=FakeAPI(action), **event_options
        )

    def all_quotes(self):
        return [
            quote
            for key in self.repository.session_keys()
            for quote in self.repository.list_quotes(key)
        ]

    def assert_no_resources(self):
        self.assertEqual(self.all_quotes(), [])
        self.assertEqual([path for path in self.root.rglob("*") if path.is_file()], [])

    async def assert_submission_failed(self, event, reason):
        await self.plugin.on_message(event)
        self.assertTrue(event.stopped)
        self.assertEqual(len(event.sent), 1)
        self.assertIn(reason, event.sent[0])
        self.assertNotIn(RECEIVED_MESSAGE, event.sent)
        self.assert_no_resources()

    async def test_text_preserves_whitespace_and_original_content(self):
        event = self.submission([segment("text", text="  原文\n[图片]  ")])
        await self.plugin.on_message(event)
        self.assertEqual(event.sent, [RECEIVED_MESSAGE])
        self.assertTrue(event.stopped)
        quote = self.all_quotes()[0]
        self.assertEqual(quote.segments[0].text, "  原文\n[图片]  ")
        self.assertEqual(
            self.service.build_quote_batches(quote)[0][0].text, "  原文\n[图片]  "
        )
        self.assertEqual(event.bot.api.calls[0], ("get_msg", {"message_id": -123}))

    async def test_all_media_and_components_round_trip_in_order(self):
        image = png_bytes()
        content = [
            segment("text", text="[图片]"),
            segment("image", file=encoded(image)),
            segment("face", id="0"),
            segment("at", qq="all"),
            segment("text", text="前"),
            segment("record", file=encoded(b"audio"), name="a.mp3"),
            segment("text", text="中"),
            segment("video", file=encoded(b"video"), name="v.mp4"),
            segment("file", file=encoded(b"document"), name="原文件.txt"),
            segment("text", text="后"),
        ]
        quote = await self.service.add_quote(self.submission(content))
        restored = QuoteRepository(self.root)
        self.service.repository = restored
        quote = restored.list_quotes(quote.group)[0]
        batches = self.service.build_quote_batches(quote)
        chain = [component for batch in batches for component in batch]
        self.assertEqual(
            [type(component) for component in chain],
            [Plain, Image, Face, At, Plain, Record, Plain, Video, File, Plain],
        )
        self.assertEqual([len(batch) for batch in batches], [5, 1, 1, 1, 1, 1])
        self.assertEqual(
            [part.text for part in chain if isinstance(part, Plain)],
            ["[图片]", "前", "中", "后"],
        )
        self.assertEqual(chain[2].id, 0)
        self.assertEqual(chain[3].qq, "all")
        self.assertEqual(chain[8].name, "原文件.txt")
        self.assertEqual(
            restored.asset_path(
                restored.find_asset(quote.group, quote.image_ids[0])
            ).read_bytes(),
            image,
        )
        self.assertEqual(
            [
                restored.asset_path(
                    restored.find_media_asset(quote.group, asset)
                ).read_bytes()
                for asset in quote.media_ids
            ],
            [b"audio", b"video", b"document"],
        )

    async def test_nested_forward_serializes_three_layers_without_losing_adjacent_content(
        self,
    ):
        third = forward(
            [
                segment("text", text="三层"),
                segment("image", file=encoded(png_bytes())),
                segment("face", id=14),
                segment("at", qq="23456"),
            ]
        )
        second = forward(
            [
                segment("record", file=encoded(b"audio")),
                third,
                segment("video", file=encoded(b"video")),
                segment("file", file=encoded(b"file"), name="f.txt"),
            ]
        )
        first = forward(
            [segment("text", text="前"), second, segment("text", text="后")]
        )
        quote = await self.service.add_quote(self.submission([first]))
        self.assertEqual(quote.kind, "forward")
        component = self.service.build_quote_batches(quote)[0][0]
        self.assertIsInstance(component, Nodes)
        payload = await component.to_dict()
        self.assertEqual(
            [node["data"]["time"] for node in payload["messages"]], [1234, 1234, 1234]
        )
        self.assertEqual(
            [node["data"]["nickname"] for node in payload["messages"]], ["作者"] * 3
        )
        kinds, text = [], []

        def walk(nodes, depth=1):
            self.assertLessEqual(depth, 3)
            for node in nodes:
                self.assertEqual(node["type"], "node")
                parts = node["data"]["content"]
                if any(part.get("type") == "node" for part in parts):
                    self.assertTrue(all(part.get("type") == "node" for part in parts))
                    walk(parts, depth + 1)
                else:
                    for part in parts:
                        kinds.append(part["type"])
                        if part["type"] == "text":
                            text.append(part["data"]["text"])

        walk(payload["messages"])
        self.assertEqual(
            kinds,
            ["text", "record", "text", "image", "face", "at", "video", "file", "text"],
        )
        self.assertEqual(text, ["前", "三层", "后"])

    async def test_fourth_forward_layer_fails_entire_submission(self):
        content = segment("text", text="深层")
        for _ in range(4):
            content = forward([content])
        await self.assert_submission_failed(self.submission([content]), "超过三层")

    async def test_forward_reference_is_fetched_through_onebot_api(self):
        event = self.submission([segment("forward", id="forward-id")])
        original = event.bot.api.handler

        def action(name, params):
            if name == "get_forward_msg":
                return {
                    "data": {
                        "messages": forward([segment("text", text="记录")])["data"][
                            "messages"
                        ]
                    }
                }
            return original(name, params)

        event.bot.api.handler = action
        quote = await self.service.add_quote(event)
        self.assertEqual(quote.forward_nodes[0].segments[0].text, "记录")
        self.assertIn(("get_forward_msg", {"id": "forward-id"}), event.bot.api.calls)

    async def test_unavailable_and_malformed_forwards_fail(self):
        cases = [
            [segment("forward", id="missing")],
            [segment("forward", messages=["invalid"])],
            [
                forward([segment("text", text="valid")], nickname="")
                | {"data": {"messages": [{"content": []}]}}
            ],
        ]
        for content in cases:
            with self.subTest(content=content):
                await self.assert_submission_failed(
                    self.submission(content), "投稿失败"
                )

    async def test_unknown_type_after_valid_media_does_not_partially_save(self):
        await self.assert_submission_failed(
            self.submission(
                [
                    segment("image", file=encoded(png_bytes())),
                    segment("json", data="unsupported"),
                ]
            ),
            "暂不支持消息类型 json",
        )

    async def test_media_failure_does_not_leave_files_or_success_confirmation(self):
        cases = [
            [segment("image", file=encoded(b"bad image"))],
            [
                segment("image", file=encoded(png_bytes())),
                segment("record", url="https://media.test/missing"),
            ],
            [segment("file", file_id="missing")],
        ]
        for content in cases:
            with self.subTest(content=content):
                await self.assert_submission_failed(
                    self.submission(content), "投稿失败"
                )

    async def test_single_file_limit_checks_content_and_reported_size(self):
        self.message_service.limits = MediaLimits(
            max_file_bytes=3, max_submission_bytes=100
        )
        for data in ({"file": encoded(b"1234")}, {"file_id": "remote", "size": 4}):
            with self.subTest(data=data):
                await self.assert_submission_failed(
                    self.submission([segment("file", **data)]), "上限"
                )

    async def test_total_limit_includes_nested_media(self):
        self.message_service.limits = MediaLimits(
            max_file_bytes=6, max_submission_bytes=6
        )
        await self.assert_submission_failed(
            self.submission(
                [
                    segment("record", file=encoded(b"1234")),
                    forward([segment("file", file=encoded(b"567"))]),
                ]
            ),
            "上限",
        )

    async def test_http_success_and_stream_size_limits(self):
        async def response(request):
            return httpx.Response(
                200,
                content=b"downloaded",
                headers={"content-type": "application/octet-stream"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            self.message_service.http_client = client
            quote = await self.service.add_quote(
                self.submission(
                    [segment("file", url="https://media.test/doc", name="d.bin")]
                )
            )
            asset = self.repository.find_media_asset(quote.group, quote.media_ids[0])
            self.assertEqual(
                self.repository.asset_path(asset).read_bytes(), b"downloaded"
            )
            self.message_service.limits = MediaLimits(max_file_bytes=3)
            event = self.submission(
                [segment("file", url="https://media.test/doc")], message_id="456"
            )
            await self.plugin.on_message(event)
            self.assertIn("上限", event.sent[0])
            self.assertEqual(len(self.all_quotes()), 1)

    async def test_download_timeout_fails_submission(self):
        async def response(request):
            await asyncio.sleep(1)
            return httpx.Response(200, content=b"late")

        self.message_service.limits = MediaLimits(timeout_seconds=0.01)
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            self.message_service.http_client = client
            await self.assert_submission_failed(
                self.submission([segment("file", url="https://media.test/slow")]),
                "获取超时",
            )

    async def test_http_chunked_body_limit_without_content_length(self):
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"123"
                yield b"456"

        async def response(request):
            return httpx.Response(200, stream=Stream())

        self.message_service.limits = MediaLimits(max_file_bytes=5)
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            self.message_service.http_client = client
            await self.assert_submission_failed(
                self.submission(
                    [
                        segment("file", url="https://media.test/chunked"),
                    ]
                ),
                "上限",
            )

    async def test_original_message_timeout_fails_submission(self):
        async def action(name, params):
            await asyncio.sleep(1)
            return {}

        self.message_service.limits = MediaLimits(timeout_seconds=0.01)
        event = FakeEvent([Reply(1), Plain("投稿")], api=FakeAPI(action))
        await self.assert_submission_failed(event, "原消息获取超时")

    async def test_api_media_fallbacks_for_group_and_private(self):
        for kind, action, group in (
            ("image", "get_image", "100"),
            ("record", "get_record", "100"),
            ("file", "get_group_file_url", "100"),
            ("file", "get_private_file_url", ""),
        ):
            with self.subTest(kind=kind, group=group):
                content = png_bytes() if kind == "image" else b"fallback"
                event = self.submission(
                    [segment(kind, file="identifier", file_id="identifier")],
                    message_id=action,
                    group=group,
                )
                original = event.bot.api.handler

                def handler(name, params):
                    if name == action:
                        return {"status": "ok", "data": {"file": encoded(content)}}
                    return original(name, params)

                event.bot.api.handler = handler
                await self.plugin.on_message(event)
                self.assertEqual(event.sent, [RECEIVED_MESSAGE])
                self.assertTrue(any(call[0] == action for call in event.bot.api.calls))

    async def test_local_media_is_copied_and_survives_source_removal(self):
        with tempfile.TemporaryDirectory() as sources:
            source = Path(sources) / "文件 空格.txt"
            source.write_bytes(b"local")
            quote = await self.service.add_quote(
                self.submission(
                    [segment("file", file=source.as_uri(), name=source.name)]
                )
            )
        asset = self.repository.find_media_asset(quote.group, quote.media_ids[0])
        self.assertEqual(self.repository.asset_path(asset).read_bytes(), b"local")

    async def test_no_reply_or_multiple_replies_reports_usage(self):
        for message in ([Plain("投稿")], [Reply(1), Reply(2), Plain("投稿")]):
            with self.subTest(message=message):
                await self.assert_submission_failed(
                    FakeEvent(message), "请先回复一条消息"
                )

    async def test_message_lookup_failure_or_empty_original_fails(self):
        for response in (
            {"status": "failed", "retcode": 100, "data": {"message": "fake"}},
            {"data": {}},
            {"data": {"message": []}},
        ):
            with self.subTest(response=response):
                event = FakeEvent(
                    [Reply(1), Plain("投稿")],
                    api=FakeAPI(lambda name, params: response),
                )
                await self.assert_submission_failed(event, "投稿失败")

    async def test_duplicate_submission_does_not_fetch_or_add_weight(self):
        first = self.submission([segment("text", text="唯一")])
        await self.plugin.on_message(first)
        repeat = self.submission([], sender="56789")
        repeat.bot.api.handler = None
        await self.plugin.on_message(repeat)
        self.assertEqual(repeat.sent, [RECEIVED_MESSAGE])
        self.assertEqual(repeat.bot.api.calls, [])
        self.assertEqual(len(self.all_quotes()), 1)

    async def test_concurrent_duplicate_and_distinct_submissions(self):
        events = [
            self.submission(
                [segment("image", file=encoded(png_bytes()))], message_id="same"
            )
            for _ in range(5)
        ]
        events += [
            self.submission([segment("text", text=str(i))], message_id=str(i))
            for i in range(5)
        ]
        await asyncio.gather(*(self.plugin.on_message(event) for event in events))
        self.assertTrue(all(event.sent == [RECEIVED_MESSAGE] for event in events))
        self.assertEqual(len(self.all_quotes()), 6)
        self.assertEqual(len(list(self.root.rglob("*.png"))), 1)

    async def test_restart_restores_deduplication_and_random_pool(self):
        event = self.submission([segment("text", text="重启后")])
        await self.service.add_quote(event)
        self.service.repository = QuoteRepository(self.root)
        quote = await self.service.add_quote(self.submission([]))
        self.assertEqual(quote.text, "重启后")
        self.assertEqual((await self.service.repository.random_quote()).id, quote.id)

    async def test_global_pool_uses_every_group_and_private_quote_once(self):
        for group, count in (("100", 1), ("200", 3), ("", 2)):
            for index in range(count):
                await self.service.add_quote(
                    self.submission(
                        [segment("text", text=f"{group}:{index}")],
                        group=group,
                        message_id=str(index),
                    )
                )
        with patch(
            "astrbot_plugin_submit_quotes.store.secrets.choice",
            side_effect=lambda pool: pool[-1],
        ) as choice:
            quote = await self.repository.random_quote()
            pool = choice.call_args.args[0]
        self.assertEqual(len(pool), 6)
        self.assertEqual(len({item.source_key for item in pool}), 6)
        query = FakeEvent([Plain("语录")], group="300")
        with patch.object(self.repository, "random_quote", return_value=quote):
            await self.plugin.on_message(query)
        self.assertEqual(query.bot.api.sent[0][0]["data"]["text"], quote.text)
        self.assertTrue(query.stopped)

    async def test_snapshot_failure_rolls_back_new_assets_and_keeps_previous_data(self):
        original = await self.service.add_quote(
            self.submission([segment("text", text="已存在")])
        )
        snapshot = self.repository.get_store(original.group).quotes_file
        old_data = snapshot.read_bytes()
        original_replace = Path.replace

        def replace(source, target):
            if Path(target).name == "quotes.json":
                raise OSError("disk full")
            return original_replace(source, target)

        with patch.object(Path, "replace", new=replace):
            event = self.submission(
                [
                    segment("image", file=encoded(png_bytes())),
                    segment("record", file=encoded(b"record")),
                ],
                message_id="new",
            )
            with self.assertLogs(ENTRY.logger, level="ERROR"):
                await self.plugin.on_message(event)
        self.assertIn("保存失败", event.sent[0])
        self.assertEqual(snapshot.read_bytes(), old_data)
        self.assertEqual(len(self.all_quotes()), 1)
        self.assertEqual(
            [path for path in self.root.rglob("*") if path.is_file()], [snapshot]
        )

    async def test_late_asset_write_failure_removes_earlier_assets(self):
        original_write = self.repository._write_asset
        count = 0

        def write(content, path, created, stage):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("write failed")
            original_write(content, path, created, stage)

        with (
            patch.object(self.repository, "_write_asset", side_effect=write),
            self.assertLogs(ENTRY.logger, level="ERROR"),
        ):
            await self.assert_submission_failed(
                self.submission(
                    [
                        segment("image", file=encoded(png_bytes())),
                        segment("record", file=encoded(b"audio")),
                    ]
                ),
                "保存失败",
            )

    async def test_corrupt_snapshot_is_not_overwritten(self):
        event = self.submission([segment("text", text="原文")])
        store = self.repository.get_store(self.service.session_key(event))
        store.quotes_file.write_text("{bad", encoding="utf-8")
        with self.assertLogs(ENTRY.logger, level="ERROR"):
            await self.plugin.on_message(event)
        self.assertIn("投稿失败", event.sent[0])
        self.assertEqual(store.quotes_file.read_text(), "{bad")

    async def test_missing_asset_prevents_any_partial_replay(self):
        quote = await self.service.add_quote(
            self.submission(
                [
                    segment("text", text="前"),
                    segment("image", file=encoded(png_bytes())),
                ]
            )
        )
        asset = self.repository.find_asset(quote.group, quote.image_ids[0])
        self.repository.asset_path(asset).unlink()
        query = FakeEvent([Plain("语录")])
        await self.plugin.on_message(query)
        self.assertEqual(len(query.sent), 1)
        self.assertIn("文件缺失", query.sent[0])

    async def test_empty_pool_and_exact_trigger_rules(self):
        query = FakeEvent([Plain(" \n语录\t ")])
        await self.plugin.on_message(query)
        self.assertEqual(query.sent, [EMPTY_MESSAGE])
        self.assertTrue(query.stopped)
        for message in (
            [Plain("/语录")],
            [Plain("语录 更多")],
            [Plain("听听语录")],
            [Plain("语录"), Face(1)],
            [At("99999"), Plain("语录")],
        ):
            with self.subTest(message=message):
                event = FakeEvent(message)
                await self.plugin.on_message(event)
                self.assertEqual(event.sent, [])
                self.assertFalse(event.stopped)

    async def test_reply_automatic_at_is_allowed_for_submission(self):
        event = self.submission([segment("text", text="内容")])
        event.message = [Reply(-123), At("23456"), Plain(" 投稿 ")]
        await self.plugin.on_message(event)
        self.assertEqual(event.sent, [RECEIVED_MESSAGE])

    async def test_bot_itself_and_other_platform_are_ignored(self):
        for options in ({"sender": "99999"}, {"platform": "other"}):
            with self.subTest(options=options):
                event = self.submission([segment("text", text="内容")], **options)
                await self.plugin.on_message(event)
                self.assertFalse(event.stopped)
                self.assertEqual(event.sent, [])
                self.assertEqual(event.bot.api.calls, [])

    async def test_query_sends_media_as_sequential_batches(self):
        await self.service.add_quote(
            self.submission(
                [
                    segment("text", text="前"),
                    segment("record", file=encoded(b"audio")),
                    segment("text", text="后"),
                    forward([segment("text", text="记录")]),
                ]
            )
        )
        query = FakeEvent([Plain("语录")])
        await self.plugin.on_message(query)
        self.assertEqual(
            [batch[0]["type"] for batch in query.bot.api.sent],
            ["text", "record", "text", "node"],
        )
        self.assertEqual(query.bot.api.calls[-1][0], "send_group_forward_msg")
        self.assertEqual(len(self.all_quotes()), 1)

    async def test_onebot_replay_preserves_whitespace_and_at_without_inserting_text(
        self,
    ):
        quote = await self.service.add_quote(
            self.submission(
                [
                    segment("text", text=" \n"),
                    segment("image", file=encoded(png_bytes())),
                    segment("at", qq="23456"),
                    segment("text", text="紧接@"),
                ]
            )
        )
        query = FakeEvent([Plain("语录")], group="", sender="34567")
        await self.service.send_quote(query, quote)
        self.assertEqual(
            [part["type"] for part in query.bot.api.sent[0]],
            ["text", "image", "at", "text"],
        )
        self.assertEqual(query.bot.api.sent[0][0]["data"]["text"], " \n")
        self.assertEqual(query.bot.api.calls[0][0], "send_private_msg")
        self.assertEqual(query.bot.api.calls[0][1]["user_id"], 34567)
        self.assertEqual(query.bot.api.calls[0][1]["self_id"], "99999")

    async def test_late_media_conversion_failure_sends_no_partial_quote(self):
        quote = await self.service.add_quote(
            self.submission(
                [
                    segment("text", text="前"),
                    segment("record", file=encoded(b"audio")),
                ]
            )
        )
        query = FakeEvent([Plain("语录")])
        with patch.object(
            Record, "convert_to_base64", side_effect=ValueError("conversion failed")
        ):
            with self.assertRaises(ValueError):
                await self.service.send_quote(query, quote)
        self.assertEqual(query.bot.api.calls, [])

    async def test_platform_send_rejection_is_reported(self):
        await self.service.add_quote(self.submission([segment("text", text="内容")]))
        query = FakeEvent([Plain("语录")])
        with patch.object(
            query.bot.api,
            "call_action",
            return_value={"status": "failed", "retcode": 100},
        ):
            await self.plugin.on_message(query)
        self.assertIn("平台拒绝", query.sent[0])
        self.assertEqual(len(self.all_quotes()), 1)

    async def test_plugin_configuration_and_shutdown(self):
        with (
            patch.object(ENTRY, "ensure_plugin_data_dir", return_value=self.root),
            patch.object(
                ENTRY.httpx,
                "AsyncClient",
                return_value=self.client,
            ),
        ):
            plugin = ENTRY.SubmitQuotesPlugin(None, {})
        self.assertEqual(plugin.quote_service.message_service.limits, MediaLimits())
        await plugin.terminate()
        self.assertTrue(self.client.is_closed)

    async def test_configuration_requires_positive_integers(self):
        for key in (
            "download_timeout_seconds",
            "max_file_size_mb",
            "max_submission_size_mb",
        ):
            for invalid in (0, -1, True, "20"):
                with (
                    self.subTest(key=key, invalid=invalid),
                    self.assertRaises(ValueError),
                ):
                    MediaLimits.from_config({key: invalid})
        schema = json.loads(
            (Path(ENTRY.__file__).parent / "_conf_schema.json").read_text()
        )
        self.assertEqual(
            {key: entry["default"] for key, entry in schema.items()},
            {
                "delete_whitelist": [],
                "download_timeout_seconds": 20,
                "max_file_size_mb": 50,
                "max_submission_size_mb": 100,
            },
        )


if __name__ == "__main__":
    unittest.main()
