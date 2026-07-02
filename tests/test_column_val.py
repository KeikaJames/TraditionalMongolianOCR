# -*- coding: utf-8 -*-
"""Validation L2 plumbing tests: build_column_val's stacking/grouping/label-join
geometry, and eval_column_pipeline's end-to-end wiring (tiny randomly-init CRNN
-> garbage decode expected, but every shape/count/CER computation must be
VALID — no accuracy assertion, this is a plumbing test)."""
import io
import json
import tarfile
import tempfile
import unittest
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    import torch
    from PIL import Image
    HAVE_DEPS = True
except Exception:
    HAVE_DEPS = False

if HAVE_DEPS:
    from mongocr.alphabet import from_counts
    from mongocr.data import list_shards, src_doc_bands
    from mongocr.model import CRNN
    from scripts.build_column_val import (build_columns, group_consecutive_by_src_doc,
                                          stack_column)
    from scripts.eval_column_pipeline import decode_column, load_columns


def _strip_array(h, w, text_len):
    """A tall-narrow grayscale strip: light paper with a dark ink band whose
    WIDTH scales with text_len (a crude stand-in for "more chars -> wider ink
    extent"), so strips a caller stacks have varying widths on purpose."""
    a = np.full((h, w), 235, dtype=np.uint8)
    ink_w = max(4, min(w - 4, 6 * text_len))
    x0 = (w - ink_w) // 2
    a[h // 4: 3 * h // 4, x0:x0 + ink_w] = 25
    return a


def _make_shard(path, items):
    """items: list of (doc_id, src_doc, text, w) — mirrors the real shard
    schema (mongocr/data.py module docstring): <key>.png + <key>.json with a
    "text"/"src_doc" field (plus the usual "kind"/"bucket"/"font"/"font_px")."""
    with tarfile.open(path, "w") as tar:
        for doc_id, src_doc, text, w in items:
            arr = _strip_array(120, w, len(text))
            buf = io.BytesIO()
            Image.fromarray(arr, mode="L").save(buf, format="PNG")
            png = buf.getvalue()
            info = tarfile.TarInfo(f"{doc_id}.png"); info.size = len(png)
            tar.addfile(info, io.BytesIO(png))
            meta = json.dumps({"kind": "line", "text": text, "src_doc": src_doc,
                               "bucket": "00000", "font": "F", "font_px": 40}).encode()
            jinfo = tarfile.TarInfo(f"{doc_id}.json"); jinfo.size = len(meta)
            tar.addfile(jinfo, io.BytesIO(meta))


@unittest.skipUnless(HAVE_DEPS, "torch/PIL/numpy not installed")
class TestGroupConsecutiveBySrcDoc(unittest.TestCase):
    def test_groups_runs_preserving_order(self):
        recs = [{"src_doc": 0, "i": 0}, {"src_doc": 0, "i": 1}, {"src_doc": 5, "i": 2},
                {"src_doc": 5, "i": 3}, {"src_doc": 5, "i": 4}, {"src_doc": 9, "i": 5}]
        groups = group_consecutive_by_src_doc(recs)
        self.assertEqual([len(g) for g in groups], [2, 3, 1])
        self.assertEqual([r["i"] for r in groups[1]], [2, 3, 4])

    def test_non_consecutive_same_src_doc_forms_separate_groups(self):
        # src_doc 0 appears twice but NOT consecutively -> two separate groups,
        # never merged (grouping is stream-order-based, not a global groupby).
        recs = [{"src_doc": 0, "i": 0}, {"src_doc": 1, "i": 1}, {"src_doc": 0, "i": 2}]
        groups = group_consecutive_by_src_doc(recs)
        self.assertEqual(len(groups), 3)

    def test_empty_input(self):
        self.assertEqual(group_consecutive_by_src_doc([]), [])


@unittest.skipUnless(HAVE_DEPS, "torch/PIL/numpy not installed")
class TestStackColumn(unittest.TestCase):
    def test_stacked_height_and_width(self):
        strips = [np.full((50, 20), 10, dtype=np.uint8),
                  np.full((80, 40), 10, dtype=np.uint8),
                  np.full((30, 10), 10, dtype=np.uint8)]
        col = stack_column(strips, line_gap=15)
        self.assertEqual(col.shape[0], 50 + 80 + 30 + 15 * 2)  # k-1 gaps
        self.assertEqual(col.shape[1], 40)  # widest strip

    def test_narrower_strips_centered_and_padded_white(self):
        narrow = np.full((20, 10), 5, dtype=np.uint8)  # all-ink narrow strip
        wide = np.full((20, 40), 5, dtype=np.uint8)
        col = stack_column([narrow, wide], line_gap=0)
        row = col[5]  # inside the narrow strip's row band
        # centered: pad is (40-10)//2 = 15 px of white on each side
        self.assertTrue(np.all(row[:15] == 255))
        self.assertTrue(np.all(row[15:25] == 5))
        self.assertTrue(np.all(row[25:] == 255))

    def test_gap_rows_are_solid_white(self):
        strips = [np.full((10, 10), 5, dtype=np.uint8), np.full((10, 10), 5, dtype=np.uint8)]
        col = stack_column(strips, line_gap=8)
        gap_band = col[10:18]
        self.assertTrue(np.all(gap_band == 255))

    def test_empty_strips_raises(self):
        with self.assertRaises(ValueError):
            stack_column([], line_gap=10)


@unittest.skipUnless(HAVE_DEPS, "torch/PIL/numpy not installed")
class TestBuildColumns(unittest.TestCase):
    def _fake_group(self, src_doc, n, w=30):
        return [{"png": None, "text": f"t{src_doc}_{i}", "src_doc": src_doc} for i in range(n)]

    def setUp(self):
        # Patch _decode_gray at call time isn't needed: build_columns calls
        # scripts.build_column_val._decode_gray on each record's "png" field,
        # so give it real bytes instead of mocking internals.
        pass

    def _fake_group_with_png(self, src_doc, n):
        items = []
        for i in range(n):
            arr = _strip_array(60, 30, 3)
            buf = io.BytesIO()
            Image.fromarray(arr, mode="L").save(buf, format="PNG")
            items.append({"png": buf.getvalue(), "text": f"t{src_doc}_{i}", "src_doc": src_doc})
        return items

    def test_only_groups_meeting_lines_min_are_eligible(self):
        groups = [self._fake_group_with_png(0, 3), self._fake_group_with_png(1, 10)]
        cols = build_columns(groups, n_columns=2, lines_min=5, lines_max=8, seed=0, line_gap=4)
        self.assertEqual(len(cols), 2)
        for c in cols:
            self.assertEqual(c["src_doc"], 1)  # only the len-10 group qualifies
            self.assertGreaterEqual(c["n_lines"], 5)
            self.assertLessEqual(c["n_lines"], 8)

    def test_label_is_space_joined_line_texts(self):
        groups = [self._fake_group_with_png(7, 6)]
        cols = build_columns(groups, n_columns=1, lines_min=3, lines_max=3, seed=1, line_gap=4)
        c = cols[0]
        self.assertEqual(c["text"], " ".join(c["line_texts"]))
        self.assertEqual(len(c["line_texts"]), 3)

    def test_deterministic_given_seed(self):
        groups = [self._fake_group_with_png(0, 10), self._fake_group_with_png(1, 10)]
        a = build_columns(groups, n_columns=5, lines_min=2, lines_max=6, seed=42, line_gap=4)
        b = build_columns(groups, n_columns=5, lines_min=2, lines_max=6, seed=42, line_gap=4)
        self.assertEqual([c["text"] for c in a], [c["text"] for c in b])
        self.assertEqual([c["n_lines"] for c in a], [c["n_lines"] for c in b])

    def test_no_eligible_group_raises(self):
        groups = [self._fake_group_with_png(0, 2)]
        with self.assertRaises(SystemExit):
            build_columns(groups, n_columns=1, lines_min=5, lines_max=8, seed=0, line_gap=4)

    def test_image_geometry_matches_stack_column(self):
        groups = [self._fake_group_with_png(3, 4)]
        cols = build_columns(groups, n_columns=1, lines_min=4, lines_max=4, seed=0, line_gap=10)
        c = cols[0]
        self.assertEqual(c["image"].shape[0], 60 * 4 + 10 * 3)  # 4 strips of h=60, 3 gaps


@unittest.skipUnless(HAVE_DEPS, "torch/PIL/numpy not installed")
class TestBuildColumnValEndToEnd(unittest.TestCase):
    """Build fake shards matching the REAL packed-shard schema (mongocr/data.py:
    <key>.png + <key>.json with text/src_doc), run the actual module-level
    pipeline used by scripts/build_column_val.py's main() (list_shards +
    src_doc_bands filtering + webdataset streaming), and check the columns it
    would write."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        # One doc (src_doc=500, in the test band) with 8 consecutive lines,
        # interleaved in the SAME shard with an unrelated train-band doc
        # (src_doc=0) so keep-filtering + grouping both get exercised.
        items = [(f"line_{0:08d}_{i}", 0, f"train_{i}", 20) for i in range(3)]
        items += [(f"line_{500:08d}_{i}", 500, f"L{i}", 15 + i) for i in range(8)]
        _make_shard(Path(self.dir) / "shard-00000.tar", items)
        self.shards = list_shards(str(Path(self.dir) / "shard-*.tar"))
        self.alpha = from_counts(Counter("".join(f"train_{i}L{i}" for i in range(8)) + "0123456789_"))

    def test_split_filter_and_grouping_via_real_pipeline(self):
        import webdataset as wds

        _is_train, _is_val, is_test = src_doc_bands(val_threshold=400, test_threshold=450, gap=10)
        pipe = wds.DataPipeline(
            wds.SimpleShardList(self.shards),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
        )
        records = []
        for s in pipe:
            meta = json.loads(s["json"])
            sd = int(meta["src_doc"])
            if not is_test(sd):
                continue
            records.append({"png": s["png"], "text": meta["text"], "src_doc": sd})
        self.assertEqual(len(records), 8)  # only src_doc=500's 8 lines
        groups = group_consecutive_by_src_doc(records)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 8)
        cols = build_columns(groups, n_columns=2, lines_min=4, lines_max=8, seed=3, line_gap=6)
        self.assertEqual(len(cols), 2)
        for c in cols:
            self.assertEqual(c["src_doc"], 500)
            self.assertGreaterEqual(c["n_lines"], 4)


def _random_crnn(n_classes=8, img_w=16, lstm_hidden=8):
    torch.manual_seed(0)
    return CRNN(n_classes, lstm_hidden=lstm_hidden, width_collapse="mean", img_w=img_w)


@unittest.skipUnless(HAVE_DEPS, "torch/PIL/numpy not installed")
class TestLoadColumns(unittest.TestCase):
    def test_loads_and_sorts_numerically_not_lexicographically(self):
        d = Path(tempfile.mkdtemp())
        for idx in (2, 10, 1):
            arr = np.full((40, 8), 200, dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(d / f"{idx}.png")
            (d / f"{idx}.json").write_text(
                json.dumps({"text": f"txt{idx}", "n_lines": 2, "src_doc": idx}), encoding="utf-8")
        cols = load_columns(d)
        self.assertEqual([c["idx"] for c in cols], [1, 2, 10])  # numeric, not string, order

    def test_raises_on_empty_dir(self):
        d = Path(tempfile.mkdtemp())
        with self.assertRaises(SystemExit):
            load_columns(d)


@unittest.skipUnless(HAVE_DEPS, "torch/PIL/numpy not installed")
class TestDecodeColumnPlumbing(unittest.TestCase):
    """No accuracy assertion (the CRNN is randomly initialized -> garbage
    decode is EXPECTED); only shape/count/type plumbing is checked."""

    def setUp(self):
        self.alpha = from_counts(Counter("abcdefgh "))
        self.model = _random_crnn(n_classes=self.alpha.n_classes, img_w=16, lstm_hidden=8)
        self.model.eval()

    def test_returns_string_and_matching_tile_count(self):
        col = np.full((1400, 24), 220, dtype=np.uint8)
        col[200:1200:100, :] = 30  # sparse ink so a profile/cut exists
        text, n_tiles, heights = decode_column(
            self.model, self.alpha, col, img_h=64, img_w=16, device=torch.device("cpu"),
            tile_target=400, tile_window=100, min_tile=150)
        self.assertIsInstance(text, str)
        self.assertGreaterEqual(n_tiles, 1)
        self.assertEqual(len(heights), n_tiles)
        self.assertEqual(sum(heights), col.shape[0])

    def test_short_column_single_tile(self):
        col = np.full((300, 24), 220, dtype=np.uint8)
        text, n_tiles, heights = decode_column(
            self.model, self.alpha, col, img_h=64, img_w=16, device=torch.device("cpu"),
            tile_target=900, tile_window=200, min_tile=300)
        self.assertEqual(n_tiles, 1)
        self.assertEqual(heights, [300])


@unittest.skipUnless(HAVE_DEPS, "torch/PIL/numpy not installed")
class TestEvalColumnPipelineEndToEnd(unittest.TestCase):
    """Full smoke: write a directory dataset (build_column_val's output shape),
    load it, decode every column with a tiny fresh CRNN, and score with
    ocr_report — mirrors what scripts/eval_column_pipeline.py's main() does,
    minus argparse/checkpoint I/O. Garbage decode is fine; every stat must be
    computable without crashing and index-0 must be present."""

    def test_end_to_end_plumbing(self):
        from mongocr.metrics import ocr_report

        d = Path(tempfile.mkdtemp())
        alpha = from_counts(Counter("abcdefgh "))
        model = _random_crnn(n_classes=alpha.n_classes, img_w=16, lstm_hidden=8)
        model.eval()

        labels = ["ab cd ef", "gh ab", "cd ef gh ab"]
        for idx, label in enumerate(labels):
            h = 600 + idx * 300
            arr = np.full((h, 20), 225, dtype=np.uint8)
            arr[::80, :] = 40
            Image.fromarray(arr, mode="L").save(d / f"{idx}.png")
            (d / f"{idx}.json").write_text(
                json.dumps({"text": label, "n_lines": 3, "src_doc": idx}), encoding="utf-8")

        columns = load_columns(d)
        self.assertEqual(len(columns), 3)
        self.assertEqual(columns[0]["text"], "ab cd ef")  # index-0 spot-check

        preds, refs, n_tiles_list = [], [], []
        for col in columns:
            pred, n_tiles, heights = decode_column(
                model, alpha, col["image"], img_h=64, img_w=16, device=torch.device("cpu"),
                tile_target=400, tile_window=100, min_tile=150)
            preds.append(pred)
            refs.append(col["text"])
            n_tiles_list.append(n_tiles)
            self.assertEqual(sum(heights), col["image"].shape[0])

        rep = ocr_report(preds, refs)  # must not raise
        self.assertEqual(rep.n, 3)
        self.assertGreaterEqual(rep.norm_cer, 0.0)  # untrained model -> some CER, always >= 0
        self.assertTrue(all(n >= 1 for n in n_tiles_list))
        # index-0 accessible for the spot-check print
        self.assertEqual(refs[0], "ab cd ef")
        self.assertIsInstance(preds[0], str)


if __name__ == "__main__":
    unittest.main()
