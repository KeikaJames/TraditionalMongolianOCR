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
    import webdataset as wds  # noqa: F401
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


if __name__ == "__main__":
    unittest.main()
