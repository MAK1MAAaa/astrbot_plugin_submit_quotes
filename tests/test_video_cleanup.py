from __future__ import annotations

import contextlib
import io
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from astrbot_plugin_submit_quotes.errors import SubmissionError
from astrbot_plugin_submit_quotes.maintenance import (
    main,
    remove_video_quotes,
    scan_video_quotes,
)
from astrbot_plugin_submit_quotes.models import (
    PendingForwardNode,
    PendingForwardSegment,
    PendingQuoteSegment,
    PreparedImage,
    PreparedMedia,
    Quote,
    QuoteSegment,
)
from astrbot_plugin_submit_quotes.store import QuoteRepository


class VideoCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = QuoteRepository(self.root)

    async def asyncTearDown(self):
        self.temporary.cleanup()

    def quote(self, quote_id: str, key: str) -> Quote:
        return Quote(
            id=quote_id,
            qq="sensitive-author",
            name="真实作者",
            text="private submission body",
            created_by="sensitive-submitter",
            created_at=1,
            group=key,
            source_key=f"source:{key}:{quote_id}",
        )

    async def add(self, quote_id: str, segments: list, key="group_a") -> Quote:
        result = await self.repository.create_quote_with_segments(
            key, self.quote(quote_id, key), segments
        )
        return result.quote

    @staticmethod
    def video() -> PendingQuoteSegment:
        return PendingQuoteSegment(
            type="video", media=PreparedMedia(b"video", ".mp4", "video")
        )

    @staticmethod
    def image() -> PendingQuoteSegment:
        return PendingQuoteSegment(
            type="image", image=PreparedImage(b"picture", ".png", "sha")
        )

    def snapshot(self) -> dict[Path, bytes]:
        return {
            path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        }

    def rewrite(self, key: str, mutate) -> None:
        path = self.repository.get_store(key).quotes_file
        payload = json.loads(path.read_text())
        mutate(payload)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    async def test_dry_run_finds_video_without_writing_or_exposing_content(self):
        await self.add("video_quote", [self.video()], "private_a")
        await self.add("image_quote", [self.image()])
        before = self.snapshot()
        self.assertEqual(
            [quote.id for quote in scan_video_quotes(self.repository)], ["video_quote"]
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--root", str(self.root)]), 0)
        data = json.loads(output.getvalue())
        self.assertEqual(
            data, {"mode": "dry-run", "count": 1, "quote_ids": ["video_quote"]}
        )
        for private in (
            "sensitive-author",
            "sensitive-submitter",
            "private submission body",
            "真实作者",
        ):
            self.assertNotIn(private, output.getvalue())
        self.assertEqual(self.snapshot(), before)

    async def test_standard_video_is_deleted_with_its_exclusive_media(self):
        await self.add("video_quote", [self.video()])
        deleted = await remove_video_quotes(self.repository)
        self.assertEqual([quote.id for quote in deleted], ["video_quote"])
        store = self.repository.get_store("group_a")
        self.assertEqual(store.load_quotes(), [])
        self.assertEqual(store.load_media_assets(), [])
        self.assertEqual(list(store.media_dir.iterdir()), [])

    async def test_nested_mixed_video_causes_entire_submission_to_be_removed(self):
        quote = await self.add(
            "mixed",
            [
                PendingQuoteSegment(type="text", text="原文"),
                self.image(),
                PendingQuoteSegment(
                    type="nodes",
                    nodes=[
                        PendingForwardNode(
                            segments=[
                                PendingForwardSegment(
                                    type="nodes",
                                    nodes=[
                                        PendingForwardNode(
                                            segments=[
                                                PendingForwardSegment(
                                                    type="video",
                                                    media=PreparedMedia(
                                                        b"clip", ".mp4", "video"
                                                    ),
                                                )
                                            ]
                                        )
                                    ],
                                )
                            ]
                        )
                    ],
                ),
            ],
        )
        await self.add("surviving_image", [self.image()], "private_a")
        self.assertEqual(
            [item.id for item in scan_video_quotes(self.repository)], [quote.id]
        )
        await remove_video_quotes(self.repository)
        self.assertEqual(
            [item.id for item in self.repository.list_all_quotes()], ["surviving_image"]
        )
        store = self.repository.get_store("group_a")
        self.assertEqual(store.load_quotes(), [])
        self.assertEqual(store.load_assets(), [])
        self.assertEqual(store.load_media_assets(), [])
        self.assertEqual(list(store.images_dir.iterdir()), [])
        self.assertEqual(list(store.media_dir.iterdir()), [])

    async def test_video_inside_top_level_forward_nodes_is_found(self):
        await self.repository.create_quote_with_forward_nodes(
            "group_a",
            self.quote("forward", "group_a"),
            [
                PendingForwardNode(
                    segments=[
                        PendingForwardSegment(type="text", text="完整聊天记录"),
                        PendingForwardSegment(
                            type="video", media=PreparedMedia(b"video", ".mp4", "video")
                        ),
                    ]
                )
            ],
        )
        self.assertEqual(
            [quote.id for quote in await remove_video_quotes(self.repository)],
            ["forward"],
        )
        self.assertEqual(self.repository.list_all_quotes(), [])

    async def test_legacy_media_id_only_video_is_found_and_cleaned(self):
        await self.add("legacy_video", [self.video()])

        def legacy(payload):
            payload["quotes"][0].pop("segments")
            payload["quotes"][0].pop("kind")

        self.rewrite("group_a", legacy)
        self.assertEqual(
            [quote.id for quote in scan_video_quotes(self.repository)], ["legacy_video"]
        )
        await remove_video_quotes(self.repository)
        store = self.repository.get_store("group_a")
        self.assertEqual(store.load_media_assets(), [])
        self.assertEqual(list(store.media_dir.iterdir()), [])

    async def test_nonvideo_legacy_content_and_file_extensions_are_preserved(self):
        await self.add("text", [PendingQuoteSegment(type="text", text="video.mp4")])
        await self.add("image", [self.image()])
        await self.add(
            "face_at",
            [
                PendingQuoteSegment(type="face", face_id=1),
                PendingQuoteSegment(type="at", qq="all"),
            ],
        )
        await self.add(
            "file",
            [
                PendingQuoteSegment(
                    type="file", media=PreparedMedia(b"file", ".mp4", "file")
                )
            ],
        )
        await self.add(
            "voice",
            [
                PendingQuoteSegment(
                    type="record", media=PreparedMedia(b"voice", ".mp4", "record")
                )
            ],
        )

        def legacy(payload):
            for entry in payload["quotes"]:
                if entry["id"] in {"file", "voice", "image", "text"}:
                    entry.pop("segments")

        self.rewrite("group_a", legacy)
        before = self.snapshot()
        self.assertEqual(await remove_video_quotes(self.repository), [])
        self.assertEqual(self.snapshot(), before)

    async def test_explicit_file_or_voice_is_not_reclassified_by_conflicting_index(
        self,
    ):
        for kind in ("file", "record"):
            await self.add(
                kind,
                [
                    PendingQuoteSegment(
                        type=kind, media=PreparedMedia(b"content", ".mp4", kind)
                    )
                ],
            )

        def conflict(payload):
            for asset in payload["media"]:
                asset["media_type"] = "video"

        self.rewrite("group_a", conflict)
        before = self.snapshot()
        self.assertEqual(await remove_video_quotes(self.repository), [])
        self.assertEqual(self.snapshot(), before)

    async def test_shared_media_and_images_are_preserved_for_surviving_record(self):
        target = await self.add("video", [self.video(), self.image()])
        store = self.repository.get_store("group_a")
        media = store.load_media_assets()
        images = store.load_assets()
        survivor = self.quote("survivor", "group_a")
        survivor.segments = [
            QuoteSegment(type="image", asset_id=images[0].asset_id),
            QuoteSegment(type="file", asset_id=media[0].asset_id),
        ]
        store.save_snapshot([target, survivor], images, media)
        paths = [self.repository.asset_path(asset) for asset in [*images, *media]]
        self.assertEqual(
            [quote.id for quote in await remove_video_quotes(self.repository)],
            ["video"],
        )
        self.assertEqual([quote.id for quote in store.load_quotes()], ["survivor"])
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertEqual(store.load_assets()[0].ref_count, 1)
        self.assertEqual(store.load_media_assets()[0].ref_count, 1)

    async def test_legacy_shared_video_reference_survives_repository_deletion(self):
        target = await self.add("video", [self.video()])
        store = self.repository.get_store("group_a")
        media = store.load_media_assets()
        survivor = self.quote("legacy_shared", "group_a")
        survivor.media_ids = [media[0].asset_id]
        survivor.segments = [QuoteSegment(type="text", text="保留引用")]
        store.save_snapshot([target, survivor], [], media)
        path = self.repository.asset_path(media[0])
        self.assertTrue(await self.repository.delete_quote("video"))
        self.assertTrue(path.is_file())
        self.assertEqual(store.load_media_assets()[0].ref_count, 1)

    async def test_repeat_cleanup_and_restart_are_idempotent(self):
        await self.add("video", [self.video()])
        await self.add("image", [self.image()])
        self.assertEqual(len(await remove_video_quotes(self.repository)), 1)
        before = self.snapshot()
        restarted = QuoteRepository(self.root)
        self.assertEqual(await remove_video_quotes(restarted), [])
        self.assertEqual(self.snapshot(), before)

    async def test_empty_session_without_snapshot_is_valid_but_lost_index_is_not(self):
        await self.add("video", [self.video()], "a")
        empty = self.repository.get_store("empty_session")
        self.assertFalse(empty.quotes_file.exists())
        self.assertEqual(
            [quote.id for quote in scan_video_quotes(self.repository)], ["video"]
        )
        (empty.media_dir / "orphan.mp4").write_bytes(b"unknown owner")
        before = self.snapshot()
        with self.assertRaises(SubmissionError):
            await remove_video_quotes(self.repository)
        self.assertEqual(self.snapshot(), before)

    async def test_duplicate_id_anywhere_aborts_before_first_delete(self):
        await self.add("first_video", [self.video()], "a")
        await self.add("duplicate", [self.video()], "b")
        await self.add("duplicate", [self.image()], "c")
        before = self.snapshot()
        with self.assertRaisesRegex(SubmissionError, "ID 重复"):
            await remove_video_quotes(self.repository)
        self.assertEqual(self.snapshot(), before)

    async def test_corrupt_unknown_or_missing_data_aborts_before_any_delete(self):
        await self.add("first_video", [self.video()], "a")
        await self.add("survivor", [self.image()], "z")
        path = self.repository.get_store("z").quotes_file
        original = path.read_text()
        bad_payloads = [
            "not json",
            json.dumps({"schema_version": 3, "quotes": [], "images": [], "media": []}),
            json.dumps(
                {"schema_version": 4, "quotes": "bad", "images": [], "media": []}
            ),
        ]
        for raw in bad_payloads:
            with self.subTest(raw=raw):
                path.write_text(raw)
                before = self.snapshot()
                with self.assertRaises(SubmissionError):
                    await remove_video_quotes(self.repository)
                self.assertEqual(self.snapshot(), before)
        path.write_text(original)
        for mutate in (
            lambda p: p["quotes"][0]["segments"].append({"type": "unknown"}),
            lambda p: p["quotes"][0].update(media_ids=["missing_resource"]),
            lambda p: p["quotes"][0].update(segments="bad"),
            lambda p: p["images"][0].update(rel_path="../outside.png"),
        ):
            path.write_text(original)
            self.rewrite("z", mutate)
            before = self.snapshot()
            with self.assertRaises(SubmissionError):
                await remove_video_quotes(self.repository)
            self.assertEqual(self.snapshot(), before)

    async def test_write_failure_propagates_without_success_result(self):
        await self.add("video", [self.video()])
        store = self.repository.get_store("group_a")
        before = self.snapshot()
        with patch.object(store, "save_snapshot", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                await remove_video_quotes(self.repository)
        self.assertEqual(self.snapshot(), before)

    async def test_disappearing_quote_is_reported_as_failure(self):
        await self.add("video", [self.video()])
        with patch.object(self.repository, "delete_quote", return_value=False):
            with self.assertRaisesRegex(SubmissionError, "未全部完成"):
                await remove_video_quotes(self.repository)

    async def test_cli_unknown_directory_fails_without_creating_it(self):
        missing = self.root / "missing"
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            self.assertEqual(main(["--root", str(missing)]), 1)
        self.assertFalse(missing.exists())


class VideoCleanupCLITests(unittest.TestCase):
    def test_apply_cli_removes_records_and_only_prints_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = QuoteRepository(Path(temporary))
            store = repository.get_store("private_identity")
            quote = Quote(
                id="id",
                qq="sensitive_identity",
                name="name",
                text="body",
                created_by="creator",
                created_at=1,
            )
            quote.segments = [QuoteSegment(type="video")]
            store.save_snapshot([quote], [], [])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["--root", temporary, "--apply"]), 0)
            self.assertEqual(
                json.loads(output.getvalue()),
                {"mode": "apply", "count": 1, "quote_ids": ["id"]},
            )
            self.assertEqual(store.load_quotes(), [])
            self.assertNotIn("sensitive_identity", output.getvalue())

    def test_cleanup_warning_causes_nonzero_exit_and_is_redacted(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = QuoteRepository(Path(temporary))
            store = repository.get_store("private_identity")
            quote = Quote(
                id="id",
                qq="author",
                name="name",
                text="body",
                created_by="creator",
                created_at=1,
            )
            quote.segments = [QuoteSegment(type="video")]
            store.save_snapshot([quote], [], [])

            async def warn_cleanup(repository, quote_id):
                logging.getLogger("astrbot_plugin_submit_quotes.store").warning(
                    "sensitive_identity sensitive_body"
                )
                return True

            output = io.StringIO()
            error = io.StringIO()
            with (
                patch.object(QuoteRepository, "delete_quote", warn_cleanup),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(error),
                self.assertLogs(
                    "astrbot_plugin_submit_quotes.store", level="WARNING"
                ) as logs,
            ):
                self.assertEqual(main(["--root", temporary, "--apply"]), 1)
            self.assertEqual(output.getvalue(), "")
            self.assertIn("未全部清理", error.getvalue())
            self.assertNotIn("sensitive_identity", "".join(logs.output))
            self.assertNotIn("sensitive_body", "".join(logs.output))


if __name__ == "__main__":
    unittest.main()
