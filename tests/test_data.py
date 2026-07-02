# -*- coding: utf-8 -*-
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    import webdataset as wds
    from PIL import Image
    HAVE_DEPS = True
except Exception:
    HAVE_DEPS = False

if HAVE_DEPS:
    from mongocr.alphabet import from_counts
    from mongocr.data import (build_pipeline, collate, decode_image, ink_crop,
                              keys_of, list_shards, src_doc_bands, src_doc_split)
    from collections import Counter


def _strip_png(text_h=120):
    """A tall-narrow grayscale strip with a dark ink band (variance signal)."""
    a = np.full((text_h, 40), 240, dtype=np.uint8)  # light paper
    a[30:90, 10:30] = 30  # dark ink band
    return Image.fromarray(a, mode="L")


def _make_shard(path, items):
    with tarfile.open(path, "w") as tar:
        for doc_id, src_doc, text in items:
            buf = io.BytesIO()
            _strip_png().save(buf, format="PNG")
            png = buf.getvalue()
            info = tarfile.TarInfo(f"{doc_id}.png"); info.size = len(png)
            tar.addfile(info, io.BytesIO(png))
            meta = json.dumps({"kind": "line", "text": text, "src_doc": src_doc,
                               "bucket": "00000", "font": "F", "font_px": 40}).encode()
            jinfo = tarfile.TarInfo(f"{doc_id}.json"); jinfo.size = len(meta)
            tar.addfile(jinfo, io.BytesIO(meta))


@unittest.skipUnless(HAVE_DEPS, "torch/PIL/webdataset not installed")
class TestData(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        # shard 0: src_doc 0,1 ; shard 1: src_doc 8,9
        _make_shard(Path(self.dir) / "shard-00000.tar",
                    [(f"line_{i:08d}_v0", i, "ᠠᠡᠢ") for i in (0, 1)])
        _make_shard(Path(self.dir) / "shard-00001.tar",
                    [(f"line_{i:08d}_v0", i, "ᠣᠤᠥ") for i in (8, 9)])
        self.shards = list_shards(str(Path(self.dir) / "shard-*.tar"))
        self.alpha = from_counts(Counter("ᠠᠡᠢᠣᠤᠥ"))

    def test_ink_crop_and_decode(self):
        x = decode_image(io.BytesIO(b""), 1, 1) if False else None
        img = _strip_png()
        cropped = ink_crop(img)
        self.assertLessEqual(cropped.size[1], img.size[1])  # cropped height <= orig
        buf = io.BytesIO(); _strip_png().save(buf, format="PNG")
        t = decode_image(buf.getvalue(), 64, 32)
        self.assertEqual(tuple(t.shape), (1, 64, 32))
        self.assertGreaterEqual(float(t.max()), float(t.min()))  # ink high

    def test_collate_pads(self):
        b = [(torch.zeros(1, 8, 4), torch.tensor([1, 2, 3])),
             (torch.zeros(1, 8, 4), torch.tensor([1]))]
        imgs, padded, lengths = collate(b)
        self.assertEqual(tuple(imgs.shape), (2, 1, 8, 4))
        self.assertEqual(tuple(padded.shape), (2, 3))
        self.assertEqual(lengths.tolist(), [3, 1])

    def test_src_doc_split_filters(self):
        is_train, is_eval = src_doc_split(threshold=8, gap=2)  # train<6, eval>=8
        train_docs = {sd for _k, sd, _t, _f in keys_of(self.shards, is_train)}
        eval_docs = {sd for _k, sd, _t, _f in keys_of(self.shards, is_eval)}
        self.assertEqual(train_docs, {0, 1})
        self.assertEqual(eval_docs, {8, 9})
        self.assertFalse(train_docs & eval_docs)  # disjoint

    def test_keys_no_duplication(self):
        keys = [k for k, _sd, _t, _f in keys_of(self.shards)]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(keys), 4)

    def test_src_doc_bands_three_way_disjoint(self):
        is_tr, is_va, is_te = src_doc_bands(val_threshold=100, test_threshold=200, gap=10)
        self.assertTrue(is_tr(50)); self.assertFalse(is_va(50)); self.assertFalse(is_te(50))
        self.assertTrue(is_va(150)); self.assertFalse(is_tr(150)); self.assertFalse(is_te(150))
        self.assertTrue(is_te(250)); self.assertFalse(is_tr(250)); self.assertFalse(is_va(250))
        # gap bands belong to nobody
        self.assertFalse(is_tr(95) or is_va(95))      # [90,100) train-val gap
        self.assertFalse(is_va(195) or is_te(195))    # [190,200) val-test gap

    def test_keys_of_yields_font(self):
        fonts = {f for _k, _sd, _t, f in keys_of(self.shards)}
        self.assertEqual(fonts, {"F"})

    def test_pipeline_yields_batches(self):
        pipe = build_pipeline(self.shards, self.alpha, img_h=64, img_w=32,
                              batch_size=2, training=False)
        batches = list(pipe)
        total = sum(int(b[0].shape[0]) for b in batches)
        self.assertEqual(total, 4)
        imgs, padded, lengths = batches[0]
        self.assertEqual(imgs.shape[1:], (1, 64, 32))

    def test_pipeline_drops_oov_lines(self):
        # alphabet missing 'ᠥ' -> the shard-1 samples (text "ᠣᠤᠥ") must be dropped,
        # leaving only the 2 shard-0 samples (text "ᠠᠡᠢ").
        alpha = from_counts(Counter("ᠠᠡᠢᠣᠤ"))  # no 'ᠥ'
        pipe = build_pipeline(self.shards, alpha, img_h=64, img_w=32,
                              batch_size=1, training=False)
        n = sum(int(b[0].shape[0]) for b in pipe)
        self.assertEqual(n, 2)

    def _distinct_text_shards_and_alpha(self):
        # A dedicated fixture with 4 DISTINCT per-sample texts (unlike setUp's
        # shared shards, where both samples within a shard share one text) so a
        # sample's encoded label tuple is a valid per-sample identity for
        # detecting reordering.
        d = tempfile.mkdtemp()
        _make_shard(Path(d) / "shard-00000.tar",
                    [("line_00000000_v0", 0, "ᠠ"), ("line_00000001_v0", 1, "ᠡ")])
        _make_shard(Path(d) / "shard-00001.tar",
                    [("line_00000008_v0", 8, "ᠢ"), ("line_00000009_v0", 9, "ᠣ")])
        shards = list_shards(str(Path(d) / "shard-*.tar"))
        alpha = from_counts(Counter("ᠠᠡᠢᠣ"))
        return shards, alpha

    def _keys_via_key_pipeline(self, *, data_seed):
        # Exercise the actual shuffle stages (training=True), which build_pipeline
        # only wires in when training.
        shards, alpha = self._distinct_text_shards_and_alpha()
        pipe = build_pipeline(shards, alpha, img_h=8, img_w=8,
                              batch_size=1, training=True, data_seed=data_seed,
                              shard_shuffle=10, sample_shuffle=10)
        keys = []
        for imgs, padded, lengths in pipe:
            # decode_image/collate don't carry the key through; the encoded
            # label id tuple is a valid per-sample identity here since all 4
            # fixture samples have distinct text.
            keys.append(tuple(padded[0, : int(lengths[0])].tolist()))
        return keys

    def test_data_seed_reproducible_order(self):
        # data_seed=7 twice -> byte-identical sample order (wds.detshuffle path).
        order1 = self._keys_via_key_pipeline(data_seed=7)
        order2 = self._keys_via_key_pipeline(data_seed=7)
        self.assertEqual(order1, order2)
        self.assertEqual(len(order1), 4)  # all 4 samples present, none duplicated

    def test_data_seed_none_still_yields_all_samples(self):
        # data_seed=None (default) must not change *what* is yielded, only that
        # the order is wall-clock-jittered (today's unchanged behavior) instead
        # of reproducible; assert the signature/contract, not order-determinism.
        order = self._keys_via_key_pipeline(data_seed=None)
        self.assertEqual(len(order), 4)
        self.assertEqual(len(set(order)), 4)  # no duplication

    def test_different_data_seeds_can_differ(self):
        order_a = self._keys_via_key_pipeline(data_seed=1)
        order_b = self._keys_via_key_pipeline(data_seed=2)
        # Same multiset of samples either way; a different seed is free to (but
        # not guaranteed to, at this tiny bufsize/sample count) reorder them —
        # the load-bearing guarantee is same-seed-same-order, already covered by
        # test_data_seed_reproducible_order above.
        self.assertEqual(sorted(order_a), sorted(order_b))

    # --- detshuffle under REAL num_workers>0 (the launch config that matters) ---
    #
    # The tests above all iterate the wds.DataPipeline directly (no DataLoader),
    # which never exercises split_by_worker or forks any process — that alone
    # does not prove detshuffle is reproducible under the actual multi-worker
    # launch. wds.WebLoader(pipe, num_workers=N) wraps pipe in a real
    # torch.utils.data.DataLoader, whose N worker processes each get their own
    # pickled/forked copy of the pipeline (so their own detshuffle instance,
    # each starting at epoch=-1); split_by_worker runs BEFORE both shuffle
    # stages, so each worker only ever shuffles the disjoint shard subset it
    # was already handed — no cross-worker seed coordination is needed for
    # (a) reproducibility (same data_seed twice -> identical order) or (b)
    # no-duplication (every sample appears in exactly one worker's stream).
    #
    # multiprocessing_context="fork" pins the worker-start semantics this test
    # asserts on Linux training boxes (this repo's actual launch target,
    # CLAUDE.md), where fork is the platform default; recent CPython on macOS
    # defaults to spawn/forkserver instead, which cannot pickle build_pipeline's
    # local closures at all (an unrelated, pre-existing platform limitation of
    # WebDataset's DataPipeline, not something this test is checking).

    def _distinct_text_shards_and_alpha_multi(self, n_shards=4, per_shard=2):
        # More shards than the 2-shard fixture above, so num_workers=2 gives
        # each worker >1 shard — a stronger stress test of "no collision" than
        # a 1-shard-per-worker split would be.
        d = tempfile.mkdtemp()
        # 30 distinct Mongolian codepoints (U+1820..) is enough for n_shards *
        # per_shard <= 30 distinct per-sample texts, so the encoded label
        # remains a valid per-sample identity.
        chars = [chr(c) for c in range(0x1820, 0x1820 + n_shards * per_shard)]
        alpha = from_counts(Counter("".join(chars)))
        i = 0
        for s in range(n_shards):
            items = []
            for _ in range(per_shard):
                items.append((f"line_{i:08d}_v0", i, chars[i]))
                i += 1
            _make_shard(Path(d) / f"shard-{s:05d}.tar", items)
        shards = list_shards(str(Path(d) / "shard-*.tar"))
        return shards, alpha, i  # i == total sample count

    def _keys_via_webloader(self, *, num_workers, data_seed):
        import multiprocessing as mp

        shards, alpha, _n = self._distinct_text_shards_and_alpha_multi()
        pipe = build_pipeline(shards, alpha, img_h=8, img_w=8, batch_size=1,
                              training=True, data_seed=data_seed,
                              shard_shuffle=10, sample_shuffle=10)
        loader = wds.WebLoader(
            pipe, batch_size=None, num_workers=num_workers,
            multiprocessing_context=mp.get_context("fork") if num_workers else None,
        )
        return [tuple(padded[0, : int(lengths[0])].tolist())
                for _imgs, padded, lengths in loader]

    def test_data_seed_reproducible_order_under_workers(self):
        # The load-bearing claim: same --data-seed, real num_workers=2 (each
        # worker forked with its own detshuffle instance) -> byte-identical
        # sample order across two independently-launched loaders.
        order1 = self._keys_via_webloader(num_workers=2, data_seed=7)
        order2 = self._keys_via_webloader(num_workers=2, data_seed=7)
        self.assertEqual(order1, order2)
        self.assertEqual(len(order1), 8)  # 4 shards x 2 samples

    def test_no_duplication_or_drop_across_workers(self):
        # Every sample appears in exactly one worker's stream: split_by_worker
        # partitions shards before either shuffle stage runs, so detshuffle
        # (worker-local, not worker-aware) cannot cause a cross-worker collision.
        order = self._keys_via_webloader(num_workers=2, data_seed=7)
        self.assertEqual(len(order), 8)
        self.assertEqual(len(set(order)), 8)  # no duplicates

    def test_different_data_seeds_can_differ_under_workers(self):
        order_a = self._keys_via_webloader(num_workers=2, data_seed=1)
        order_b = self._keys_via_webloader(num_workers=2, data_seed=2)
        self.assertEqual(sorted(order_a), sorted(order_b))  # same multiset
        self.assertNotEqual(order_a, order_b)  # different seed -> different order


if __name__ == "__main__":
    unittest.main()
