from __future__ import annotations

import base64
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image as PILImage

from astrbot_plugin_submit_quotes.models import (
    PendingForwardNode,
    PendingForwardSegment,
    PreparedMedia,
    Quote,
)
from astrbot_plugin_submit_quotes.quote_service import RECEIVED_MESSAGE
from astrbot_plugin_submit_quotes.tests.fakes import (
    FakeAPI,
    FakeEvent,
    Plain,
    Reply,
    import_plugin_entry,
)

ENTRY = import_plugin_entry()


def segment(kind: str, **data) -> dict:
    return {"type": kind, "data": data}


def image_segment() -> dict:
    stream = io.BytesIO()
    PILImage.new("RGB", (2, 3), "blue").save(stream, format="PNG")
    return segment(
        "image", file="base64://" + base64.b64encode(stream.getvalue()).decode()
    )


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


class SubmissionTypePolicyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

        async def allow(**kwargs):
            return types.SimpleNamespace(
                completion_text=json.dumps(
                    {
                        "decision": "allow",
                        "images_reviewed": len(kwargs.get("image_urls", [])),
                    }
                )
            )

        self.context = types.SimpleNamespace(
            get_current_chat_provider_id=AsyncMock(return_value="review-model"),
            llm_generate=AsyncMock(side_effect=allow),
        )
        self.plugin = self.make_plugin({})
        self.counter = 0

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.temporary.cleanup()

    def make_plugin(self, config):
        with patch.object(ENTRY, "ensure_plugin_data_dir", return_value=self.root):
            return ENTRY.SubmitQuotesPlugin(self.context, config)

    def submission(self, content, **options):
        self.counter += 1

        def fetch(action, params):
            if action == "get_msg":
                return {
                    "data": {
                        "message": content,
                        "sender": {"user_id": "23456", "nickname": "原作者"},
                    }
                }
            raise AssertionError(f"不应解析媒体直链：{action}")

        event = FakeEvent(
            [Reply(self.counter), Plain("投稿")], api=FakeAPI(fetch), **options
        )
        event.unified_msg_origin = "qq-test:GroupMessage:100"
        return event

    def files(self):
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }

    def assert_type_rejected(self, event):
        self.assertTrue(event.stopped)
        self.assertEqual(len(event.sent), 1)
        self.assertIn("仅支持文字和图片", event.sent[0])
        self.assertNotIn(RECEIVED_MESSAGE, event.sent)
        self.context.llm_generate.assert_not_awaited()
        self.context.get_current_chat_provider_id.assert_not_awaited()

    async def test_text_and_image_submissions_keep_content_and_order(self):
        for content in (
            [segment("plain", text="  纯文字\n ")],
            [image_segment()],
            [segment("text", text="前"), image_segment(), segment("text", text="后")],
        ):
            with self.subTest(content=content):
                event = self.submission(content)
                await self.plugin.on_message(event)
                self.assertTrue(event.stopped)
                self.assertEqual(event.sent, [RECEIVED_MESSAGE])
        saved = self.plugin.repository.list_all_quotes()
        mixed = next(quote for quote in saved if quote.text == "前后")
        self.assertEqual(
            [part.type for part in mixed.segments], ["text", "image", "text"]
        )
        self.assertEqual(mixed.segments[0].text, "前")
        self.assertEqual(mixed.segments[-1].text, "后")

    async def test_three_levels_of_nested_text_images_are_accepted(self):
        content = [segment("text", text="深层"), image_segment()]
        for _ in range(3):
            content = [forward(content)]
        event = self.submission(content)
        await self.plugin.on_message(event)
        self.assertEqual(event.sent, [RECEIVED_MESSAGE])
        quote = self.plugin.repository.list_all_quotes()[0]
        nodes = quote.forward_nodes
        for _ in range(2):
            nodes = nodes[0].segments[0].nodes
        self.assertEqual([part.type for part in nodes[0].segments], ["text", "image"])

    async def test_node_and_nodes_containers_are_accepted(self):
        for kind in ("node", "nodes"):
            node = {
                "user_id": "34567",
                "nickname": "作者",
                "content": [image_segment()],
            }
            content = segment(kind, **(node if kind == "node" else {"nodes": [node]}))
            event = self.submission([content])
            await self.plugin.on_message(event)
            self.assertEqual(event.sent, [RECEIVED_MESSAGE])

    async def test_forward_reference_is_resolved_once_before_loading_images(self):
        event = self.submission([segment("forward", id="history")])
        original = event.bot.api.handler

        def fetch(action, params):
            if action == "get_forward_msg":
                return {
                    "data": {"messages": forward([image_segment()])["data"]["messages"]}
                }
            return original(action, params)

        event.bot.api.handler = fetch
        await self.plugin.on_message(event)
        self.assertEqual(event.sent, [RECEIVED_MESSAGE])
        self.assertEqual(
            sum(action == "get_forward_msg" for action, _ in event.bot.api.calls), 1
        )

    async def test_forbidden_leaf_types_fail_pure_mixed_and_nested_without_media_or_review(
        self,
    ):
        for kind in ("video", "record", "file", "face", "at", "reply", "json"):
            forbidden = segment(
                kind, file="https://media.invalid/forbidden", id="1", qq="12345"
            )
            for content in (
                [forbidden],
                [image_segment(), segment("text", text="不能裁剪"), forbidden],
                [
                    image_segment(),
                    forward([segment("text", text="嵌套"), image_segment(), forbidden]),
                ],
            ):
                with (
                    self.subTest(kind=kind, content=content),
                    patch.object(
                        self.plugin.quote_service.message_service,
                        "_prepare_media",
                        side_effect=AssertionError("类型不合规时不得获取任何媒体"),
                    ) as prepare,
                ):
                    event = self.submission(content)
                    await self.plugin.on_message(event)
                self.assert_type_rejected(event)
                prepare.assert_not_called()
                self.assertEqual(
                    [action for action, _ in event.bot.api.calls], ["get_msg"]
                )
                self.assertEqual(self.plugin.repository.list_all_quotes(), [])
                self.assertEqual(self.files(), {})

    async def test_forbidden_later_forward_node_does_not_load_an_earlier_image(self):
        nodes = forward([image_segment()])
        nodes["data"]["messages"].append(
            {
                "user_id": "34567",
                "nickname": "作者",
                "content": [segment("video", file="old-video")],
            }
        )
        event = self.submission([nodes])
        with patch.object(
            self.plugin.quote_service.message_service,
            "_prepare_media",
            side_effect=AssertionError("禁止在预检完毕前下载图片"),
        ) as prepare:
            await self.plugin.on_message(event)
        self.assert_type_rejected(event)
        prepare.assert_not_called()
        self.assertEqual(self.files(), {})

    async def test_disabling_or_relaxing_review_cannot_allow_forbidden_submission_types(
        self,
    ):
        for config in (
            {"moderation_enabled": False},
            {
                "moderation_scope": "text_only",
                "moderation_unreviewable_policy": "allow",
            },
        ):
            await self.plugin.terminate()
            self.plugin = self.make_plugin(config)
            for kind in ("video", "record", "file", "face", "at"):
                with self.subTest(config=config, kind=kind):
                    event = self.submission(
                        [segment("text", text="文字"), segment(kind)]
                    )
                    await self.plugin.on_message(event)
                self.assert_type_rejected(event)
                self.assertEqual(self.plugin.repository.list_all_quotes(), [])
                self.assertEqual(self.files(), {})

    async def seed_legacy(self, kind, *, nested):
        event = self.submission([])
        service = self.plugin.quote_service
        session = service.session_key(event)
        quote = Quote(
            id=f"q_legacy_{self.counter}",
            qq="23456",
            name="原作者",
            text="",
            created_by="12345",
            created_at=float(self.counter),
            group=session,
            source_key=service.source_key(event, str(self.counter)),
            source_message_id=str(self.counter),
        )
        leaf = PendingForwardSegment(type=kind, qq="23456", face_id=14)
        if kind in {"video", "record", "file"}:
            leaf.media = PreparedMedia(
                content=b"legacy", extension=".bin", media_type=kind
            )
        if nested:
            result = await self.plugin.repository.create_quote_with_forward_nodes(
                session,
                quote,
                [
                    PendingForwardNode(
                        sender_uin="23456", sender_name="原作者", segments=[leaf]
                    )
                ],
            )
        else:
            result = await self.plugin.repository.create_quote_with_segments(
                session, quote, [leaf]
            )
        return event, result.quote

    async def test_duplicate_legacy_forbidden_types_fail_without_fetching_reading_or_review(
        self,
    ):
        for kind in ("video", "record", "file", "face", "at"):
            for nested in (False, True):
                event, _ = await self.seed_legacy(kind, nested=nested)
                before = self.files()
                with (
                    self.subTest(kind=kind, nested=nested),
                    patch.object(
                        self.plugin.quote_service,
                        "_stored_review_segments",
                        side_effect=AssertionError("禁止读取旧媒体"),
                    ) as stored_review,
                    patch.object(
                        self.plugin.repository,
                        "asset_path",
                        side_effect=AssertionError("禁止读取旧媒体"),
                    ) as asset_path,
                ):
                    await self.plugin.on_message(event)
                self.assert_type_rejected(event)
                stored_review.assert_not_called()
                asset_path.assert_not_called()
                self.assertEqual(event.bot.api.calls, [])
                self.assertEqual(self.files(), before)

    async def test_duplicate_legacy_cannot_bypass_type_policy_with_review_disabled(
        self,
    ):
        await self.plugin.terminate()
        self.plugin = self.make_plugin({"moderation_enabled": False})
        event, _ = await self.seed_legacy("video", nested=True)
        before = self.files()
        await self.plugin.on_message(event)
        self.assert_type_rejected(event)
        self.assertEqual(event.bot.api.calls, [])
        self.assertEqual(self.files(), before)

    async def test_existing_non_image_media_can_still_be_replayed(self):
        for kind in ("record", "file", "face", "at"):
            event, quote = await self.seed_legacy(kind, nested=False)
            await self.plugin.quote_service.send_quote(event, quote)
            self.assertEqual(event.bot.api.sent[0][0]["type"], kind)

    async def test_bot_events_and_non_exact_commands_remain_ignored(self):
        for options in ({"sender": "99999"}, {"platform": "other"}):
            event = self.submission([segment("video")], **options)
            await self.plugin.on_message(event)
            self.assertFalse(event.stopped)
            self.assertEqual(event.bot.api.calls, [])
            self.assertEqual(event.sent, [])
        for text in ("/投稿", "请投稿", "投稿 其他"):
            event = FakeEvent([Reply("1"), Plain(text)])
            await self.plugin.on_message(event)
            self.assertFalse(event.stopped)
            self.assertEqual(event.bot.api.calls, [])
        self.context.llm_generate.assert_not_awaited()
