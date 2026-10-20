from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image

from astrbot_plugin_submit_quotes.models import PendingQuoteSegment, PreparedImage
from astrbot_plugin_submit_quotes.moderation import (
    REJECTED_MESSAGE,
    ContentModerator,
    ModerationConfig,
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
ERROR_MESSAGE = (
    "The request contains sensitive content. Please modify your input and try again."
)


def error_payload(code="content_check_rejected", *, error_type="content_check_error"):
    return {"error": {"message": ERROR_MESSAGE, "type": error_type, "code": code}}


def production_error():
    # AstrBot's provider wraps a raw SSE error string in this diagnostic.
    return Exception(
        "API 返回的 completion 类型错误：<class 'str'>: data: "
        + json.dumps(error_payload(), ensure_ascii=False)
        + "\n\n"
    )


def sdk_error(**attributes):
    exc = RuntimeError("provider request failed")
    for key, value in attributes.items():
        setattr(exc, key, value)
    return exc


def png_content():
    output = io.BytesIO()
    with Image.new("RGB", (3, 4), "green") as image:
        image.save(output, format="PNG")
    return output.getvalue()


class ProviderRefusalModerationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = SimpleNamespace(
            get_current_chat_provider_id=AsyncMock(return_value="review-model"),
            llm_generate=AsyncMock(),
        )
        self.event = SimpleNamespace(unified_msg_origin="qq-test:GroupMessage:100")
        self.moderator = ContentModerator(self.context, ModerationConfig())
        logging_patch = patch("astrbot_plugin_submit_quotes.moderation.logger")
        self.logger = logging_patch.start()
        self.addCleanup(logging_patch.stop)

    async def review_error(self, exc, *, segments=None):
        self.context.llm_generate.side_effect = exc
        return await self.moderator.review(
            self.event,
            segments or [PendingQuoteSegment(type="text", text="待审文字")],
        )

    async def assert_rejected(self, exc):
        with self.assertRaises(ModerationRejected) as raised:
            await self.review_error(exc)
        self.assertEqual(str(raised.exception), REJECTED_MESSAGE)
        self.logger.exception.assert_not_called()

    async def test_production_wrapped_sse_error_is_rejected_without_error_log(self):
        await self.assert_rejected(production_error())

    async def test_sdk_error_body_envelope_is_rejected(self):
        await self.assert_rejected(sdk_error(body=error_payload()))

    async def test_sdk_error_root_body_is_rejected(self):
        await self.assert_rejected(sdk_error(body=error_payload()["error"]))

    async def test_sdk_error_code_attribute_is_rejected(self):
        await self.assert_rejected(sdk_error(code="content_check_rejected"))

    async def test_explicit_content_policy_codes_are_rejected(self):
        for code in (
            "content_check_rejected",
            "content_filter",
            "content_policy_violation",
        ):
            with self.subTest(code=code):
                await self.assert_rejected(sdk_error(body=error_payload(code)))

    async def test_wrapped_exception_cause_retains_content_refusal(self):
        wrapper = RuntimeError("generation wrapper failed")
        wrapper.__cause__ = sdk_error(body=error_payload())
        await self.assert_rejected(wrapper)

    async def test_multiple_wrapping_levels_retain_content_refusal(self):
        middle = RuntimeError("inner wrapper")
        middle.__cause__ = production_error()
        outer = RuntimeError("outer wrapper")
        outer.__cause__ = middle
        await self.assert_rejected(outer)

    async def test_http_error_response_json_envelope_is_rejected(self):
        request = httpx.Request("POST", "https://example.invalid/chat/completions")
        response = httpx.Response(400, json=error_payload(), request=request)
        exc = httpx.HTTPStatusError("bad request", request=request, response=response)
        await self.assert_rejected(exc)

    async def test_plain_json_provider_error_is_rejected(self):
        await self.assert_rejected(RuntimeError(json.dumps(error_payload())))

    async def test_plain_sse_provider_error_is_rejected(self):
        await self.assert_rejected(
            RuntimeError("data: " + json.dumps(error_payload()) + "\n\n")
        )

    async def test_transport_auth_rate_limit_and_server_errors_stay_unavailable(self):
        cases = [
            RuntimeError("connection reset"),
            httpx.ConnectError("connection failed"),
        ]
        request = httpx.Request("POST", "https://example.invalid/chat/completions")
        for status, code in (
            (401, "invalid_api_key"),
            (429, "rate_limit_exceeded"),
            (500, "internal_error"),
        ):
            response = httpx.Response(status, json=error_payload(code), request=request)
            cases.append(
                httpx.HTTPStatusError(
                    "request failed", request=request, response=response
                )
            )
        for exc in cases:
            with self.subTest(error=type(exc).__name__, message=str(exc)):
                with self.assertRaises(ModerationUnavailable):
                    await self.review_error(exc)
                self.logger.exception.assert_called()
                self.logger.reset_mock()

    async def test_content_error_type_without_known_code_stays_unavailable(self):
        for code in (None, "", "unknown_sensitive_error"):
            with self.subTest(code=code), self.assertRaises(ModerationUnavailable):
                await self.review_error(sdk_error(body=error_payload(code)))

    async def test_unstructured_sensitive_words_and_code_names_are_not_refusals(self):
        for message in (
            "provider unavailable: sensitive content processing service offline",
            "unknown provider failure mentions content_check_rejected",
            "content_filter service connection failed",
            "API error: The request contains sensitive content",
        ):
            with (
                self.subTest(message=message),
                self.assertRaises(ModerationUnavailable),
            ):
                await self.review_error(RuntimeError(message))

    async def test_quoted_prompt_error_json_is_not_provider_refusal(self):
        payload = json.dumps(error_payload())
        for message in (
            "request failed; prompt=" + payload,
            "request failed; prompt=" + json.dumps(payload),
            json.dumps({"prompt": payload, "message": "internal service failure"}),
            "original message quoted data: " + payload,
        ):
            with (
                self.subTest(message=message),
                self.assertRaises(ModerationUnavailable),
            ):
                await self.review_error(RuntimeError(message))

    async def test_json_non_error_fields_do_not_supply_a_policy_code(self):
        body = {"prompt": error_payload(), "error": {"code": "internal_error"}}
        with self.assertRaises(ModerationUnavailable):
            await self.review_error(sdk_error(body=body))

    async def test_err_response_structured_json_and_sse_are_rejected(self):
        raw_json = json.dumps(error_payload())
        for raw in (raw_json, "data: " + raw_json + "\n\n"):
            with self.subTest(raw=raw):
                self.context.llm_generate.return_value = SimpleNamespace(
                    role="err", completion_text=raw
                )
                with self.assertRaises(ModerationRejected) as raised:
                    await self.moderator.review(
                        self.event, [PendingQuoteSegment(type="text", text="待审文字")]
                    )
                self.assertEqual(str(raised.exception), REJECTED_MESSAGE)
                self.logger.exception.assert_not_called()

    async def test_err_response_unknown_code_is_unavailable(self):
        self.context.llm_generate.return_value = SimpleNamespace(
            role="err", completion_text=json.dumps(error_payload("internal_error"))
        )
        with self.assertRaises(ModerationUnavailable):
            await self.moderator.review(
                self.event, [PendingQuoteSegment(type="text", text="待审文字")]
            )

    async def test_assistant_natural_language_refusal_stays_unavailable(self):
        for text in (ERROR_MESSAGE, "content_check_rejected", "暂时不能审查", ""):
            with self.subTest(text=text):
                self.context.llm_generate.return_value = SimpleNamespace(
                    role="assistant", completion_text=text
                )
                with self.assertRaises(ModerationUnavailable):
                    await self.moderator.review(
                        self.event, [PendingQuoteSegment(type="text", text="待审文字")]
                    )

    async def test_provider_refusal_cleans_every_temporary_image(self):
        paths = []

        async def reject(**kwargs):
            paths.extend(Path(path) for path in kwargs["image_urls"])
            self.assertTrue(paths)
            self.assertTrue(all(path.is_file() for path in paths))
            raise production_error()

        self.context.llm_generate.side_effect = reject
        image = PendingQuoteSegment(
            type="image", image=PreparedImage(png_content(), ".png", "test")
        )
        with self.assertRaises(ModerationRejected):
            await self.moderator.review(self.event, [image])
        self.assertTrue(
            all(not path.exists() and not path.parent.exists() for path in paths)
        )
        self.logger.exception.assert_not_called()

    async def test_prompt_quoting_error_json_can_still_receive_allow(self):
        self.context.llm_generate.return_value = SimpleNamespace(
            role="assistant", completion_text='{"decision":"allow","images_reviewed":0}'
        )
        await self.moderator.review(
            self.event,
            [PendingQuoteSegment(type="text", text=json.dumps(error_payload()))],
        )


class ProviderRefusalSubmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = []

        async def refuse(**kwargs):
            self.paths.extend(Path(path) for path in kwargs["image_urls"])
            self.assertTrue(all(path.is_file() for path in self.paths))
            raise production_error()

        self.context = SimpleNamespace(
            get_current_chat_provider_id=AsyncMock(return_value="review-model"),
            llm_generate=AsyncMock(side_effect=refuse),
        )
        with patch.object(ENTRY, "ensure_plugin_data_dir", return_value=self.root):
            self.plugin = ENTRY.SubmitQuotesPlugin(self.context, {})

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.temporary.cleanup()

    def submission(self, *, image=False):
        content = [{"type": "text", "data": {"text": "待审文字"}}]
        if image:
            content.append(
                {
                    "type": "image",
                    "data": {
                        "file": "base64://" + base64.b64encode(png_content()).decode()
                    },
                }
            )

        def get_message(action, _params):
            self.assertEqual(action, "get_msg")
            return {"data": {"message": content, "sender": {"user_id": "23456"}}}

        event = FakeEvent([Reply("123"), Plain("投稿")], api=FakeAPI(get_message))
        event.unified_msg_origin = "qq-test:GroupMessage:100"
        return event

    def assert_no_submission(self, event, expected):
        self.assertTrue(event.stopped)
        self.assertEqual(event.sent, [expected])
        self.assertNotIn(RECEIVED_MESSAGE, event.sent)
        self.assertEqual(self.plugin.repository.list_all_quotes(), [])
        self.assertEqual([path for path in self.root.rglob("*") if path.is_file()], [])

    async def test_production_event_sends_exact_refusal_and_never_persists(self):
        event = self.submission()
        with patch("astrbot_plugin_submit_quotes.moderation.logger") as logger:
            await self.plugin.on_message(event)
        self.assert_no_submission(event, REJECTED_MESSAGE)
        self.assertIsInstance(self.plugin.quote_service.moderator, ContentModerator)
        self.context.llm_generate.assert_awaited_once()
        logger.exception.assert_not_called()

    async def test_production_image_refusal_cleans_review_frames_and_store(self):
        event = self.submission(image=True)
        await self.plugin.on_message(event)
        self.assert_no_submission(event, REJECTED_MESSAGE)
        self.assertTrue(self.paths)
        self.assertTrue(
            all(not path.exists() and not path.parent.exists() for path in self.paths)
        )

    async def test_production_text_only_scope_refusal_still_uses_exact_reply(self):
        self.plugin.quote_service.moderator.config = ModerationConfig(scope="text_only")
        event = self.submission(image=True)
        await self.plugin.on_message(event)
        self.assert_no_submission(event, REJECTED_MESSAGE)
        self.assertEqual(self.paths, [])

    async def test_ordinary_provider_failure_never_confirms_or_creates_quote(self):
        self.context.llm_generate.side_effect = RuntimeError("network unavailable")
        event = self.submission(image=True)
        with self.assertLogs(
            "astrbot_plugin_submit_quotes.moderation", level="ERROR"
        ) as logs:
            await self.plugin.on_message(event)
        self.assertEqual(len(event.sent), 1)
        self.assertTrue(event.sent[0].startswith("投稿失败："))
        self.assert_no_submission(event, event.sent[0])
        self.assertIn("network unavailable", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
