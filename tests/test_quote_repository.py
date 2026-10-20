from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from astrbot_plugin_submit_quotes.errors import SubmissionError
from astrbot_plugin_submit_quotes.models import (
    ForwardNode,
    ForwardSegment,
    ImageAsset,
    PendingForwardNode,
    PendingForwardSegment,
    PendingQuoteSegment,
    PreparedImage,
    PreparedMedia,
    Quote,
    QuoteSegment,
)
from astrbot_plugin_submit_quotes.store import QuoteRepository


class RepositoryFeaturesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = QuoteRepository(self.root)

    async def asyncTearDown(self):
        self.temporary.cleanup()

    def quote(self, quote_id: str, key: str = "group_a", created_at: float = 0) -> Quote:
        return Quote(
            id=quote_id,
            qq="author",
            name="投稿作者",
            text=quote_id,
            created_by="submitter",
            created_at=created_at,
            group=key,
            source_key=f"source:{key}:{quote_id}",
        )

    async def add_text(
        self, quote_id: str, key: str = "group_a", created_at: float = 0
    ) -> Quote:
        result = await self.repository.create_quote_with_segments(
            key,
            self.quote(quote_id, key, created_at),
            [PendingQuoteSegment(type="text", text=quote_id)],
        )
        return result.quote

    async def add_media(self, quote_id: str, key: str = "group_a") -> Quote:
        result = await self.repository.create_quote_with_segments(
            key,
            self.quote(quote_id, key),
            [
                PendingQuoteSegment(
                    type="image",
                    image=PreparedImage(b"picture", ".png", "hash"),
                ),
                PendingQuoteSegment(
                    type="file",
                    media=PreparedMedia(b"document", ".txt", "file"),
                ),
            ],
        )
        return result.quote

    async def test_global_list_is_stable_and_persisted_across_restart(self):
        await self.add_text("older", "private_a", 1)
        await self.add_text("new_b", "group_b", 3)
        await self.add_text("new_a", "group_a", 3)
        await self.add_text("middle", "private_a", 2)
        restarted = QuoteRepository(self.root)
        self.assertEqual(
            [quote.id for quote in restarted.list_all_quotes()],
            ["new_a", "new_b", "middle", "older"],
        )

    async def test_random_quote_still_draws_from_all_records(self):
        self.assertIsNone(await self.repository.random_quote())
        await self.add_text("a", "group_a")
        await self.add_text("b", "group_a")
        await self.add_text("c", "private_a")
        with patch(
            "astrbot_plugin_submit_quotes.store.secrets.choice",
            side_effect=lambda candidates: candidates[-1],
        ) as draw:
            result = await self.repository.random_quote()
        self.assertEqual(result.id, "c")
        self.assertEqual({quote.id for quote in draw.call_args.args[0]}, {"a", "b", "c"})

    async def test_banquet_rejects_fewer_than_six_without_modifying_records(self):
        for count in range(6):
            with self.subTest(count=count):
                with self.assertRaisesRegex(
                    SubmissionError, rf"至少需要 6 条.*当前 {count} 条"
                ):
                    await self.repository.random_quotes()
                self.assertEqual(len(self.repository.list_all_quotes()), count)
            await self.add_text(str(count))

    async def test_banquet_count_bounds_and_no_replacement(self):
        for count in range(1, 12):
            await self.add_text(str(count), f"session_{count % 2}")
            if count < 6:
                continue
            upper = min(count, 9)
            for offset in range(upper - 6 + 1):
                with self.subTest(total=count, offset=offset):
                    with patch(
                        "astrbot_plugin_submit_quotes.store.secrets.randbelow",
                        return_value=offset,
                    ) as draw:
                        selected = await self.repository.random_quotes()
                    draw.assert_called_once_with(upper - 6 + 1)
                    self.assertEqual(len(selected), 6 + offset)
                    self.assertEqual(len({quote.id for quote in selected}), len(selected))
                    self.assertTrue(
                        {quote.id for quote in selected}.issubset(
                            {str(index) for index in range(1, count + 1)}
                        )
                    )

    async def test_banquet_samples_individual_records_across_sessions(self):
        for index in range(12):
            await self.add_text(str(index), "private_a" if index == 0 else "group_a")
        with patch(
            "astrbot_plugin_submit_quotes.store.secrets.SystemRandom"
        ) as generator:
            generator.return_value.sample.side_effect = (
                lambda candidates, count: candidates[:count]
            )
            result = await self.repository.random_quotes()
        candidates, count = generator.return_value.sample.call_args.args
        self.assertEqual({quote.id for quote in candidates}, {str(index) for index in range(12)})
        self.assertIn(count, range(6, 10))
        self.assertEqual(len(result), count)
        self.assertEqual(len({quote.id for quote in result}), len(result))

    async def test_invalid_sample_range_is_rejected(self):
        for minimum, maximum in [(0, 9), (-1, 6), (8, 7)]:
            with self.subTest(minimum=minimum, maximum=maximum):
                with self.assertRaises(ValueError):
                    await self.repository.random_quotes(minimum, maximum)

    async def test_delete_requires_exact_existing_id(self):
        await self.add_text("full_quote_id")
        for quote_id in ("", "missing", "full_quote", "full_quote_id "):
            self.assertFalse(await self.repository.delete_quote(quote_id))
        self.assertEqual(len(self.repository.list_all_quotes()), 1)

    async def test_delete_is_global_and_persists_after_restart(self):
        await self.add_text("group_quote", "group_a")
        await self.add_text("private_quote", "private_a")
        self.assertTrue(await self.repository.delete_quote("private_quote"))
        restarted = QuoteRepository(self.root)
        self.assertEqual([quote.id for quote in restarted.list_all_quotes()], ["group_quote"])
        self.assertFalse(await restarted.delete_quote("private_quote"))

    async def test_duplicate_id_across_sessions_is_never_deleted(self):
        await self.add_text("same", "group_a")
        await self.add_text("same", "private_a")
        with self.assertRaisesRegex(SubmissionError, "ID 重复"):
            await self.repository.delete_quote("same")
        self.assertEqual(len(self.repository.list_all_quotes()), 2)

    async def test_duplicate_id_in_same_session_is_never_deleted(self):
        quote = await self.add_text("same")
        store = self.repository.get_store("group_a")
        store.save_snapshot([quote, quote], [], [])
        with self.assertRaisesRegex(SubmissionError, "ID 重复"):
            await self.repository.delete_quote("same")
        self.assertEqual(len(store.load_quotes()), 2)

    async def test_concurrent_deletions_remove_a_record_once(self):
        await self.add_media("target")
        results = await asyncio.gather(
            self.repository.delete_quote("target"),
            self.repository.delete_quote("target"),
        )
        self.assertEqual(sorted(results), [False, True])
        store = self.repository.get_store("group_a")
        self.assertEqual(store.load_quotes(), [])
        self.assertEqual(store.load_assets(), [])
        self.assertEqual(store.load_media_assets(), [])
        self.assertEqual(list(store.images_dir.iterdir()), [])
        self.assertEqual(list(store.media_dir.iterdir()), [])

    async def test_delete_rereads_snapshot_after_waiting_for_session_lock(self):
        await self.add_text("target")
        store = self.repository.get_store("group_a")
        await store.lock.acquire()
        deletion = asyncio.create_task(self.repository.delete_quote("target"))
        await asyncio.sleep(0)
        added = self.quote("new", created_at=1)
        added.segments = [QuoteSegment(type="text", text="new")]
        store.save_snapshot([*store.load_quotes(), added], [], [])
        store.lock.release()
        self.assertTrue(await deletion)
        self.assertEqual([quote.id for quote in store.load_quotes()], ["new"])

    async def test_delete_rereads_id_collisions_after_waiting_for_lock(self):
        await self.add_text("target", "group_a")
        store = self.repository.get_store("group_a")
        await store.lock.acquire()
        deletion = asyncio.create_task(self.repository.delete_quote("target"))
        await asyncio.sleep(0)
        await self.add_text("target", "private_a")
        store.lock.release()
        with self.assertRaisesRegex(SubmissionError, "ID 重复"):
            await deletion
        self.assertEqual(len(self.repository.list_all_quotes()), 2)

    async def test_write_failure_preserves_record_indexes_and_media(self):
        await self.add_media("target")
        store = self.repository.get_store("group_a")
        original = {
            path: path.read_bytes()
            for path in store.root.rglob("*")
            if path.is_file()
        }

        def fail_snapshot(*args):
            self.assertTrue(store.lock.locked())
            raise OSError("disk full")

        with (
            patch.object(store, "save_snapshot", side_effect=fail_snapshot),
            patch.object(self.repository, "_cleanup_asset_path") as cleanup,
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                await self.repository.delete_quote("target")
        cleanup.assert_not_called()
        self.assertEqual(
            {path: path.read_bytes() for path in store.root.rglob("*") if path.is_file()},
            original,
        )
        self.assertEqual(len(QuoteRepository(self.root).list_all_quotes()), 1)

    async def test_delete_recurses_into_standard_message_forward_segments(self):
        result = await self.repository.create_quote_with_segments(
            "group_a",
            self.quote("nested"),
            [
                PendingQuoteSegment(
                    type="image", image=PreparedImage(b"top", ".png", "top")
                ),
                PendingQuoteSegment(
                    type="nodes",
                    nodes=[
                        PendingForwardNode(
                            segments=[
                                PendingForwardSegment(
                                    type="image",
                                    image=PreparedImage(b"nested", ".png", "nested"),
                                ),
                                PendingForwardSegment(
                                    type="record",
                                    media=PreparedMedia(b"audio", ".mp3", "record"),
                                ),
                            ]
                        )
                    ],
                ),
            ],
        )
        store = self.repository.get_store("group_a")
        self.assertEqual(len(store.load_assets()), 2)
        self.assertEqual(len(store.load_quotes()[0].image_ids), 1)
        self.assertTrue(await self.repository.delete_quote(result.quote.id))
        self.assertEqual(store.load_assets(), [])
        self.assertEqual(store.load_media_assets(), [])
        self.assertEqual(list(store.images_dir.iterdir()), [])
        self.assertEqual(list(store.media_dir.iterdir()), [])

    async def test_delete_recurses_into_nested_forward_nodes(self):
        result = await self.repository.create_quote_with_forward_nodes(
            "group_a",
            self.quote("forward"),
            [
                PendingForwardNode(
                    segments=[
                        PendingForwardSegment(
                            type="nodes",
                            nodes=[
                                PendingForwardNode(
                                    segments=[
                                        PendingForwardSegment(
                                            type="video",
                                            media=PreparedMedia(b"video", ".mp4", "video"),
                                        )
                                    ]
                                )
                            ],
                        )
                    ]
                )
            ],
        )
        store = self.repository.get_store("group_a")
        self.assertEqual(len(store.load_media_assets()), 1)
        self.assertTrue(await self.repository.delete_quote(result.quote.id))
        self.assertEqual(store.load_media_assets(), [])
        self.assertEqual(list(store.media_dir.iterdir()), [])

    async def test_shared_nested_assets_are_preserved_with_correct_reference_counts(self):
        target = await self.add_media("target")
        store = self.repository.get_store("group_a")
        images = store.load_assets()
        media = store.load_media_assets()
        shared = self.quote("shared")
        shared.segments = [
            QuoteSegment(
                type="nodes",
                nodes=[
                    ForwardNode(
                        segments=[
                            ForwardSegment(type="image", asset_id=images[0].asset_id),
                            ForwardSegment(type="file", asset_id=media[0].asset_id),
                        ]
                    )
                ],
            )
        ]
        images[0].ref_count = media[0].ref_count = 99
        paths = [self.repository.asset_path(asset) for asset in [*images, *media]]
        store.save_snapshot([target, shared], images, media)
        self.assertTrue(await self.repository.delete_quote("target"))
        self.assertEqual(store.load_assets()[0].ref_count, 1)
        self.assertEqual(store.load_media_assets()[0].ref_count, 1)
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertTrue(await self.repository.delete_quote("shared"))
        self.assertEqual(store.load_assets(), [])
        self.assertEqual(store.load_media_assets(), [])
        self.assertTrue(all(not path.exists() for path in paths))

    async def test_different_asset_ids_sharing_a_file_do_not_delete_retained_file(self):
        target = await self.add_media("target")
        store = self.repository.get_store("group_a")
        images = store.load_assets()
        shared_image = ImageAsset.from_dict(images[0].to_dict())
        shared_image.asset_id = "shared_alias"
        shared = self.quote("shared")
        shared.segments = [QuoteSegment(type="image", asset_id=shared_image.asset_id)]
        path = self.repository.asset_path(shared_image)
        store.save_snapshot([target, shared], [*images, shared_image], store.load_media_assets())
        self.assertTrue(await self.repository.delete_quote("target"))
        self.assertTrue(path.is_file())
        self.assertEqual([asset.asset_id for asset in store.load_assets()], ["shared_alias"])
        self.assertTrue(await self.repository.delete_quote("shared"))
        self.assertFalse(path.exists())

    async def test_cleanup_never_follows_unsafe_or_other_session_paths(self):
        protected = self.root / "protected.txt"
        protected.write_bytes(b"keep me")
        other = self.repository.get_store("private_other")
        other_path = other.images_dir / "other.png"
        other_path.write_bytes(b"other session")
        for index, unsafe in enumerate(
            [
                str(protected),
                "../outside.txt",
                "protected.txt",
                str(other_path.relative_to(self.root)),
                "symlink",
                "symlink_directory",
            ]
        ):
            with self.subTest(unsafe=unsafe):
                key = f"group_{index}"
                quote = await self.add_media(f"target_{index}", key)
                store = self.repository.get_store(key)
                assets = store.load_assets()
                if unsafe == "symlink":
                    link = store.images_dir / "link.png"
                    link.symlink_to(protected)
                    assets[0].rel_path = str(link.relative_to(self.root))
                elif unsafe == "symlink_directory":
                    link = store.images_dir / "link_dir"
                    link.symlink_to(self.root, target_is_directory=True)
                    assets[0].rel_path = str((link / protected.name).relative_to(self.root))
                else:
                    assets[0].rel_path = unsafe
                store.save_snapshot([quote], assets, store.load_media_assets())
                with self.assertLogs("astrbot_plugin_submit_quotes.store", level="WARNING"):
                    self.assertTrue(await self.repository.delete_quote(quote.id))
                self.assertEqual(protected.read_bytes(), b"keep me")
                self.assertEqual(other_path.read_bytes(), b"other session")
                self.assertEqual(store.load_quotes(), [])
                self.assertEqual(store.load_assets(), [])

    async def test_cleanup_failure_is_logged_after_successful_commit(self):
        await self.add_media("target")
        store = self.repository.get_store("group_a")
        image = self.repository.asset_path(store.load_assets()[0])
        original_unlink = Path.unlink

        def fail_image(path, *args, **kwargs):
            if path.resolve() == image:
                raise PermissionError("busy")
            return original_unlink(path, *args, **kwargs)

        with (
            patch.object(Path, "unlink", fail_image),
            self.assertLogs("astrbot_plugin_submit_quotes.store", level="ERROR"),
        ):
            self.assertTrue(await self.repository.delete_quote("target"))
        self.assertTrue(image.exists())
        self.assertEqual(store.load_quotes(), [])
        self.assertEqual(store.load_assets(), [])
        self.assertFalse(await self.repository.delete_quote("target"))


if __name__ == "__main__":
    unittest.main()
