from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image as PILImage

from astrbot_plugin_submit_quotes.errors import SubmissionError
from astrbot_plugin_submit_quotes.models import (
    ForwardNode,
    ForwardSegment,
    ImageAsset,
    Quote,
    QuoteSegment,
)
from astrbot_plugin_submit_quotes.quote_service import QuoteService
from astrbot_plugin_submit_quotes.store import QuoteRepository
from astrbot_plugin_submit_quotes.tests.fakes import COMPONENTS, FakeEvent


def picture(color="red", size=(900, 600), *, animated=False) -> bytes:
    stream = io.BytesIO()
    first = PILImage.new("RGB", size, color)
    if animated:
        first.save(
            stream,
            format="GIF",
            save_all=True,
            append_images=[PILImage.new("RGB", size, "blue")],
            duration=100,
            loop=0,
        )
    else:
        first.save(stream, format="PNG")
    return stream.getvalue()


def content_text(node: dict) -> str:
    return "".join(
        part["data"]["text"]
        for part in node["data"]["content"]
        if part["type"] == "text"
    )


def content_images(node: dict) -> list[dict]:
    return [part for part in node["data"]["content"] if part["type"] == "image"]


def decode_image(segment: dict) -> PILImage.Image:
    value = segment["data"]["file"]
    if not value.startswith("base64://"):
        raise AssertionError("预览图片必须为内存 base64 图片")
    image = PILImage.open(io.BytesIO(base64.b64decode(value.removeprefix("base64://"))))
    image.load()
    return image


class CatalogPreviewTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = QuoteRepository(self.root)
        self.service = QuoteService(self.repository, None, None, COMPONENTS)
        self.counter = 0

    async def asyncTearDown(self):
        self.temporary.cleanup()

    def save_quote(
        self,
        images: list[bytes] | None = None,
        *,
        session="group_a",
        nested=False,
        forward=False,
        extra_segments: list[QuoteSegment] | None = None,
    ) -> Quote:
        self.counter += 1
        store = self.repository.get_store(session)
        assets = store.load_assets()
        segments = []
        for index, data in enumerate(images or []):
            asset_id = f"img_{self.counter}_{index}"
            path = store.images_dir / f"{asset_id}.png"
            path.write_bytes(data)
            assets.append(
                ImageAsset(
                    asset_id=asset_id,
                    file_name=path.name,
                    rel_path=str(path.relative_to(self.root)),
                    sha256=hashlib.sha256(data).hexdigest(),
                    ref_count=1,
                )
            )
            segments.append(QuoteSegment(type="image", asset_id=asset_id))
        segments.extend(extra_segments or [])
        if not segments:
            segments.append(QuoteSegment(type="text", text="纯文字"))
        if nested:
            segments = [
                QuoteSegment(
                    type="nodes",
                    nodes=[
                        ForwardNode(
                            segments=[
                                ForwardSegment(
                                    type="nodes",
                                    nodes=[
                                        ForwardNode(
                                            segments=[
                                                ForwardSegment.from_dict(item.to_dict())
                                                for item in segments
                                            ]
                                        )
                                    ],
                                )
                            ]
                        )
                    ],
                )
            ]
        quote = Quote(
            id=f"q_catalog_{self.counter}",
            qq="23456",
            name="作者",
            text="原消息",
            created_by="12345",
            created_at=self.counter,
            group=session,
            segments=segments,
        )
        if forward:
            quote.kind = "forward"
            quote.forward_nodes = [
                ForwardNode(
                    segments=[
                        ForwardSegment.from_dict(item.to_dict()) for item in segments
                    ]
                )
            ]
            quote.segments = []
        store.save_snapshot([*store.load_quotes(), quote], assets, [])
        return self.repository.list_quotes(session)[-1]

    async def catalog(self, entries, **event_options):
        event = FakeEvent(**event_options)
        await self.service.send_catalog(
            event, entries, header="投稿列表（第 1 页）", footer="删除投稿 <完整ID>"
        )
        action, params = event.bot.api.calls[-1]
        return event, action, params

    async def test_real_image_and_full_id_are_in_the_same_node(self):
        first = self.save_quote([picture("red")])
        second = self.save_quote([picture("green")])
        _, action, params = await self.catalog(
            [(first, "第一条简介"), (second, "第二条简介")]
        )
        self.assertEqual(action, "send_group_forward_msg")
        self.assertEqual(len(params["messages"]), 4)
        for node, quote, color in zip(
            params["messages"][1:-1], (first, second), ("red", "green")
        ):
            self.assertIn(quote.id, content_text(node))
            self.assertEqual(node["data"]["nickname"], "投稿预览")
            self.assertEqual(node["data"]["user_id"], "99999")
            self.assertIsInstance(node["data"]["time"], int)
            image = decode_image(content_images(node)[0])
            self.assertEqual(image.size, (480, 320))
            expected = (254, 0, 0) if color == "red" else (0, 128, 1)
            self.assertLess(sum(abs(a - b) for a, b in zip(image.getpixel((1, 1)), expected)), 6)

    async def test_nested_forward_images_keep_order_and_have_a_three_image_limit(self):
        quote = self.save_quote(
            [picture(color) for color in ("red", "green", "blue", "white", "black")],
            nested=True,
            forward=True,
        )
        _, _, params = await self.catalog([(quote, "嵌套聊天记录")])
        node = params["messages"][1]
        self.assertIn(quote.id, content_text(node))
        self.assertIn("还有 2 张图片", content_text(node))
        images = [decode_image(item) for item in content_images(node)]
        self.assertEqual(len(images), 3)
        pixels = [image.getpixel((1, 1)) for image in images]
        self.assertGreater(pixels[0][0], 240)
        self.assertGreater(pixels[1][1], 120)
        self.assertGreater(pixels[2][2], 240)

    async def test_nested_nodes_inside_an_ordinary_message_are_previewed(self):
        quote = self.save_quote([picture("blue")], nested=True)
        _, _, params = await self.catalog([(quote, "普通消息里的转发")])
        self.assertEqual(len(content_images(params["messages"][1])), 1)

    async def test_gif_uses_first_frame_and_never_changes_the_original_file(self):
        original = picture(animated=True)
        quote = self.save_quote([original])
        asset = self.repository.find_asset(quote.group, quote.image_ids[0])
        path = self.repository.asset_path(asset)
        before = path.stat().st_mtime_ns
        files_before = sorted(self.root.rglob("*"))
        _, _, params = await self.catalog([(quote, "动画")])
        image = decode_image(content_images(params["messages"][1])[0])
        self.assertEqual(image.format, "JPEG")
        self.assertEqual(getattr(image, "n_frames", 1), 1)
        self.assertGreater(image.getpixel((1, 1))[0], 240)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.assertEqual(sorted(self.root.rglob("*")), files_before)

    async def test_small_transparent_image_is_flattened_without_upscaling(self):
        stream = io.BytesIO()
        PILImage.new("RGBA", (20, 30), (0, 0, 0, 0)).save(stream, format="PNG")
        quote = self.save_quote([stream.getvalue()])
        _, _, params = await self.catalog([(quote, "透明图")])
        image = decode_image(content_images(params["messages"][1])[0])
        self.assertEqual(image.size, (20, 30))
        self.assertEqual(image.getpixel((1, 1)), (255, 255, 255))

    async def test_missing_corrupt_and_missing_index_images_do_not_hide_ids(self):
        missing = self.save_quote([picture()])
        missing_asset = self.repository.find_asset(missing.group, missing.image_ids[0])
        self.repository.asset_path(missing_asset).unlink()
        corrupt = self.save_quote([b"broken image"])
        no_index = self.save_quote([picture()])
        store = self.repository.get_store(no_index.group)
        store.save_snapshot(
            store.load_quotes(),
            [asset for asset in store.load_assets() if asset.asset_id not in no_index.image_ids],
            [],
        )
        healthy = self.save_quote([picture()])
        entries = [(quote, "图片简介") for quote in (missing, corrupt, no_index, healthy)]
        _, _, params = await self.catalog(entries)
        for node, (quote, _) in zip(params["messages"][1:-1], entries):
            self.assertIn(quote.id, content_text(node))
            if quote is healthy:
                self.assertEqual(len(content_images(node)), 1)
            else:
                self.assertIn("无法预览", content_text(node))
                self.assertEqual(content_images(node), [])

    async def test_each_broken_image_is_marked_without_hiding_healthy_siblings(self):
        quote = self.save_quote([b"broken", picture("blue")])
        _, _, params = await self.catalog([(quote, "坏图和好图")])
        node = params["messages"][1]
        self.assertIn("第 1 张图片无法预览", content_text(node))
        self.assertEqual(len(content_images(node)), 1)

    async def test_removed_record_is_marked_and_does_not_read_deleted_files(self):
        quote = self.save_quote([picture()])
        await self.repository.delete_quote(quote.id)
        with patch.object(self.service, "_catalog_thumbnail") as thumbnail:
            _, _, params = await self.catalog([(quote, "旧预览")])
        self.assertIn(quote.id, content_text(params["messages"][1]))
        self.assertIn("已删除", content_text(params["messages"][1]))
        thumbnail.assert_not_called()

    async def test_stored_at_and_other_media_are_never_executed_as_preview(self):
        quote = self.save_quote(
            extra_segments=[
                QuoteSegment(type="at", qq="all"),
                QuoteSegment(type="record", asset_id="missing_record"),
                QuoteSegment(type="video", asset_id="missing_video"),
                QuoteSegment(type="file", asset_id="missing_file"),
            ]
        )
        _, _, params = await self.catalog([(quote, "@所有人 [语音] [视频] [文件]")])
        for node in params["messages"]:
            self.assertTrue(all(item["type"] == "text" for item in node["data"]["content"]))

    async def test_group_and_private_targets_preserve_bot_id_and_footer(self):
        quote = self.save_quote()
        for group, action, target_key, target in (
            ("456", "send_group_forward_msg", "group_id", 456),
            ("", "send_private_forward_msg", "user_id", 12345),
        ):
            with self.subTest(group=group):
                event, actual_action, params = await self.catalog([(quote, "文字")], group=group)
                self.assertEqual(actual_action, action)
                self.assertEqual(params[target_key], target)
                self.assertEqual(params["self_id"], "99999")
                self.assertIn("投稿列表", content_text(params["messages"][0]))
                self.assertEqual(content_text(params["messages"][-1]), "删除投稿 <完整ID>")
                self.assertEqual(event.sent, [])

    async def test_missing_bot_id_uses_requester_as_forward_sender(self):
        quote = self.save_quote()
        _, _, params = await self.catalog([(quote, "文字")], self_id="")
        self.assertNotIn("self_id", params)
        self.assertTrue(all(node["data"]["user_id"] == "12345" for node in params["messages"]))

    async def test_platform_rejections_are_reported(self):
        quote = self.save_quote()
        for response in ({"retcode": 1200}, {"status": "failed", "retcode": 0}):
            event = FakeEvent()
            event.bot.api.call_action = AsyncMock(return_value=response)
            with self.subTest(response=response), self.assertRaisesRegex(SubmissionError, "拒绝发送投稿预览"):
                await self.service.send_catalog(event, [(quote, "文字")], header="头", footer="尾")

    async def test_invalid_target_and_overlarge_page_are_rejected_before_sending(self):
        quote = self.save_quote()
        for entries, event in (
            ([(quote, "文字")], FakeEvent(group="not-id")),
            ([(quote, "文字")] * 11, FakeEvent()),
        ):
            with self.subTest(entries=len(entries)), self.assertRaises(SubmissionError):
                await self.service.send_catalog(event, entries, header="头", footer="尾")
            self.assertEqual(event.bot.api.calls, [])

    async def test_delete_waits_for_thumbnail_generation(self):
        quote = self.save_quote([picture()])
        started = threading.Event()
        release = threading.Event()
        thumbnail = self.service._catalog_thumbnail

        def delayed(path):
            self.assertTrue(self.repository.get_store(quote.group).lock.locked())
            started.set()
            if not release.wait(3):
                raise TimeoutError("测试缩略图未释放")
            return thumbnail(path)

        with patch.object(self.service, "_catalog_thumbnail", side_effect=delayed):
            listing = asyncio.create_task(self.catalog([(quote, "图片")]))
            self.assertTrue(await asyncio.to_thread(started.wait, 3))
            deletion = asyncio.create_task(self.repository.delete_quote(quote.id))
            await asyncio.sleep(0)
            self.assertFalse(deletion.done())
            release.set()
            _, _, params = await listing
            self.assertTrue(await deletion)
        self.assertEqual(len(content_images(params["messages"][1])), 1)
        self.assertEqual(self.repository.list_all_quotes(), [])

    async def test_all_session_locks_are_taken_in_sorted_order_and_released_before_send(self):
        later = self.save_quote([picture()], session="z_group")
        earlier = self.save_quote([picture()], session="a_group")
        acquired = []
        released = []

        class RecordingLock:
            def __init__(self, name):
                self.name = name
                self.active = False

            async def __aenter__(self):
                acquired.append(self.name)
                self.active = True

            async def __aexit__(self, *args):
                released.append(self.name)
                self.active = False

        locks = [RecordingLock(key) for key in ("a_group", "z_group")]
        for lock in locks:
            self.repository.get_store(lock.name).lock = lock
        original_thumbnail = self.service._catalog_thumbnail

        def check_locks(path):
            self.assertTrue(all(lock.active for lock in locks))
            return original_thumbnail(path)

        with patch.object(self.service, "_catalog_thumbnail", side_effect=check_locks):
            await self.catalog([(later, "后"), (earlier, "前")])
        self.assertEqual(acquired, ["a_group", "z_group"])
        self.assertEqual(released, ["z_group", "a_group"])
        self.assertTrue(all(not lock.active for lock in locks))


if __name__ == "__main__":
    unittest.main()
