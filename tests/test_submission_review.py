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

from astrbot_plugin_submit_quotes.moderation import (
    ContentModerator,
    ModerationRejected,
    ModerationUnavailable,
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


def encoded(content: bytes) -> str:
    return "base64://" + base64.b64encode(content).decode()


def png_bytes() -> bytes:
    stream = io.BytesIO()
    PILImage.new("RGB", (3, 4), "green").save(stream, format="PNG")
    return stream.getvalue()


def forward(content: list) -> dict:
    return segment(
        "forward",
        messages=[
            {
                "sender": {"user_id": "34567", "nickname": "记录作者"},
                "time": 1234,
                "content": content,
            }
        ],
    )


class SubmissionReviewIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.review_images: list[bytes] = []
        self.review_image_paths: list[Path] = []

        async def allow_review(**kwargs):
            paths = [Path(path) for path in kwargs.get("image_urls", [])]
            self.review_image_paths.extend(paths)
            self.review_images.extend(path.read_bytes() for path in paths)
            return types.SimpleNamespace(
                completion_text=json.dumps(
                    {"decision": "allow", "images_reviewed": len(paths)}
                )
            )

        self.context = types.SimpleNamespace(
            get_current_chat_provider_id=AsyncMock(return_value="review-model"),
            llm_generate=AsyncMock(side_effect=allow_review),
        )
        self.plugin = self.make_plugin({})

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.temporary.cleanup()

    def make_plugin(self, config):
        with patch.object(ENTRY, "ensure_plugin_data_dir", return_value=self.root):
            return ENTRY.SubmitQuotesPlugin(self.context, config)

    def submission(self, content, *, message_id="123", sender="12345"):
        def get_message(action, params):
            if action == "get_msg":
                return {
                    "data": {
                        "message": content,
                        "sender": {"user_id": "23456", "nickname": "原作者"},
                    }
                }
            raise AssertionError(f"不应请求额外接口：{action}")

        event = FakeEvent(
            [Reply(message_id), Plain("投稿")],
            sender=sender,
            api=FakeAPI(get_message),
        )
        event.unified_msg_origin = "qq-test:GroupMessage:100"
        return event

    def all_quotes(self):
        return self.plugin.repository.list_all_quotes()

    def files(self):
        return {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }

    def assert_failed_without_resources(self, event, *, rejected=False):
        self.assertTrue(event.stopped)
        self.assertEqual(len(event.sent), 1)
        if rejected:
            self.assertEqual(event.sent, ["不中不中"])
        else:
            self.assertTrue(event.sent[0].startswith("投稿失败："), event.sent)
        self.assertNotIn(RECEIVED_MESSAGE, event.sent)
        self.assertEqual(self.all_quotes(), [])
        self.assertEqual(self.files(), {})

    async def seed_legacy(self, content):
        service = self.plugin.quote_service
        moderator = service.moderator
        service.moderator = None
        event = self.submission(content)
        try:
            quote = await service.add_quote(event)
        finally:
            service.moderator = moderator
        return quote

    async def test_production_constructor_reviews_text_by_default(self):
        self.assertIsInstance(self.plugin.quote_service.moderator, ContentModerator)
        event = self.submission([segment("text", text="  原文，保留空白\n ")])
        await self.plugin.on_message(event)
        self.assertTrue(event.stopped)
        self.assertEqual(event.sent, [RECEIVED_MESSAGE])
        self.context.get_current_chat_provider_id.assert_awaited_once_with(
            umo=event.unified_msg_origin
        )
        self.context.llm_generate.assert_awaited_once()
        self.assertEqual(
            self.all_quotes()[0].segments[0].text, "  原文，保留空白\n "
        )

    async def test_rejected_text_replies_exactly_and_never_creates_record(self):
        self.context.llm_generate.side_effect = None
        self.context.llm_generate.return_value = types.SimpleNamespace(
            completion_text='{"decision":"reject","images_reviewed":0}'
        )
        event = self.submission([segment("text", text="模拟被模型拒绝的内容")])
        await self.plugin.on_message(event)
        self.assert_failed_without_resources(event, rejected=True)

    async def test_approved_image_is_reviewed_before_original_is_persisted(self):
        image = png_bytes()
        content = [
            segment("text", text="  前文 "),
            segment("image", file=encoded(image)),
            segment("text", text=" 后文\n"),
        ]
        original_review = self.context.llm_generate.side_effect

        async def review_before_commit(**kwargs):
            self.assertEqual(self.files(), {})
            self.assertEqual(self.all_quotes(), [])
            return await original_review(**kwargs)

        self.context.llm_generate.side_effect = review_before_commit
        event = self.submission(content)
        await self.plugin.on_message(event)
        self.assertEqual(event.sent, [RECEIVED_MESSAGE])
        quote = self.all_quotes()[0]
        self.assertEqual([item.type for item in quote.segments], ["text", "image", "text"])
        self.assertEqual(quote.segments[0].text, "  前文 ")
        self.assertEqual(quote.segments[2].text, " 后文\n")
        self.assertTrue(self.review_images)
        asset = self.plugin.repository.find_asset(quote.group, quote.image_ids[0])
        self.assertEqual(self.plugin.repository.asset_path(asset).read_bytes(), image)
        self.assertTrue(all(not path.exists() for path in self.review_image_paths))

    async def test_rejected_image_leaves_no_media_or_index(self):
        async def reject_review(**kwargs):
            paths = kwargs.get("image_urls", [])
            self.assertTrue(paths)
            self.assertTrue(all(Path(path).is_file() for path in paths))
            return types.SimpleNamespace(
                completion_text=json.dumps(
                    {"decision": "reject", "images_reviewed": len(paths)}
                )
            )

        self.context.llm_generate.side_effect = reject_review
        event = self.submission([segment("image", file=encoded(png_bytes()))])
        await self.plugin.on_message(event)
        self.assert_failed_without_resources(event, rejected=True)

    async def test_model_failure_after_image_download_does_not_commit(self):
        self.context.llm_generate.side_effect = RuntimeError("模拟模型离线")
        event = self.submission([segment("image", file=encoded(png_bytes()))])
        with self.assertLogs(
            "astrbot_plugin_submit_quotes.moderation", level="ERROR"
        ) as logs:
            await self.plugin.on_message(event)
        self.assertIn("模拟模型离线", "\n".join(logs.output))
        self.assert_failed_without_resources(event)

    async def test_malformed_review_response_never_confirms_submission(self):
        self.context.llm_generate.side_effect = None
        self.context.llm_generate.return_value = types.SimpleNamespace(
            completion_text="这是一条没有结构化判定的回复"
        )
        event = self.submission([segment("text", text="内容")])
        await self.plugin.on_message(event)
        self.assert_failed_without_resources(event)

    async def test_moderation_timeout_failure_rolls_back_whole_submission(self):
        self.plugin.quote_service.moderator = types.SimpleNamespace(
            review=AsyncMock(side_effect=ModerationUnavailable("审查超时"))
        )
        event = self.submission(
            [
                segment("text", text="文本"),
                segment("image", file=encoded(png_bytes())),
                segment("file", file=encoded(b"document"), name="a.txt"),
            ]
        )
        await self.plugin.on_message(event)
        self.assert_failed_without_resources(event)

    async def test_disabled_review_preserves_previous_media_submission_support(self):
        await self.plugin.terminate()
        self.plugin = self.make_plugin({"moderation_enabled": False})
        event = self.submission(
            [
                segment("text", text="文字"),
                segment("image", file=encoded(png_bytes())),
                segment("record", file=encoded(b"audio"), name="a.mp3"),
            ]
        )
        await self.plugin.on_message(event)
        self.assertEqual(event.sent, [RECEIVED_MESSAGE])
        self.context.get_current_chat_provider_id.assert_not_awaited()
        self.context.llm_generate.assert_not_awaited()
        quote = self.all_quotes()[0]
        self.assertEqual([part.type for part in quote.segments], ["text", "image", "record"])
        asset = self.plugin.repository.find_media_asset(quote.group, quote.media_ids[0])
        self.assertEqual(self.plugin.repository.asset_path(asset).read_bytes(), b"audio")

    async def test_no_reference_uses_usage_hint_without_calling_review(self):
        event = FakeEvent([Plain("投稿")])
        event.unified_msg_origin = "qq-test:GroupMessage:100"
        await self.plugin.on_message(event)
        self.assert_failed_without_resources(event)
        self.context.llm_generate.assert_not_awaited()

    async def test_old_image_is_rechecked_from_local_archive_without_api_fetch(self):
        image = png_bytes()
        quote = await self.seed_legacy([segment("image", file=encoded(image))])
        before = self.files()
        moderator = types.SimpleNamespace(
            review=AsyncMock(side_effect=ModerationRejected("拒绝"))
        )
        self.plugin.quote_service.moderator = moderator
        event = self.submission([], sender="67890")
        await self.plugin.on_message(event)
        self.assertEqual(event.sent, ["不中不中"])
        self.assertTrue(event.stopped)
        self.assertEqual(event.bot.api.calls, [])
        self.assertEqual([item.id for item in self.all_quotes()], [quote.id])
        self.assertEqual(self.files(), before)
        moderator.review.assert_awaited_once()
        reviewed = moderator.review.await_args.args[1]
        self.assertEqual(reviewed[0].image.content, image)

    async def test_old_nested_forward_images_are_rechecked_recursively(self):
        image = png_bytes()
        quote = await self.seed_legacy(
            [forward([forward([forward([segment("image", file=encoded(image))])])])]
        )
        moderator = types.SimpleNamespace(review=AsyncMock())
        self.plugin.quote_service.moderator = moderator
        event = self.submission([])
        await self.plugin.on_message(event)
        self.assertEqual(event.sent, [RECEIVED_MESSAGE])
        self.assertEqual(event.bot.api.calls, [])
        self.assertEqual([item.id for item in self.all_quotes()], [quote.id])
        reviewed = moderator.review.await_args.args[1]
        for _ in range(3):
            self.assertEqual(reviewed[0].type, "nodes")
            reviewed = reviewed[0].nodes[0].segments
        self.assertEqual(reviewed[0].type, "image")
        self.assertEqual(reviewed[0].image.content, image)

    async def test_repeated_source_still_calls_real_model_and_never_adds_weight(self):
        content = [segment("text", text="可通过的内容")]
        first = self.submission(content)
        await self.plugin.on_message(first)
        original_id = self.all_quotes()[0].id
        repeated = self.submission([])
        await self.plugin.on_message(repeated)
        self.assertEqual(repeated.sent, [RECEIVED_MESSAGE])
        self.assertEqual(repeated.bot.api.calls, [])
        self.assertEqual(self.context.llm_generate.await_count, 2)
        self.assertEqual([item.id for item in self.all_quotes()], [original_id])


if __name__ == "__main__":
    unittest.main()
