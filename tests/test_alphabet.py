# -*- coding: utf-8 -*-
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from mongocr.alphabet import Alphabet, from_counts, load, save, scan_labels


class TestAlphabet(unittest.TestCase):
    def test_from_counts_sorted_and_ids(self):
        counts = Counter({"c": 1, "a": 5, "b": 2})
        alpha = from_counts(counts)
        self.assertEqual(alpha.chars, ["a", "b", "c"])  # sorted by codepoint
        self.assertEqual(alpha.stoi, {"a": 0, "b": 1, "c": 2})
        self.assertEqual(alpha.blank, 3)
        self.assertEqual(alpha.n_classes, 4)

    def test_min_count_curation(self):
        counts = Counter({"a": 100, "b": 50, "x": 2, "🍏": 1})  # x,emoji are noise
        alpha = from_counts(counts, min_count=10)
        self.assertEqual(alpha.chars, ["a", "b"])  # noise tail dropped

    def test_covers(self):
        alpha = from_counts(Counter("abc"))
        self.assertTrue(alpha.covers("abc"))
        self.assertFalse(alpha.covers("abz"))  # 'z' OOV -> line should be dropped

    def test_encode_drops_oov(self):
        alpha = from_counts(Counter("abc"))
        self.assertEqual(alpha.encode("abz"), [0, 1])  # defensive fallback

    def test_save_load_roundtrip_and_sha(self):
        alpha = from_counts(Counter("ᠠᠡabc "))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "alphabet.json"
            save(alpha, p, source="t", n_labels=6)
            loaded = load(p)
            self.assertEqual(loaded.chars, alpha.chars)
            self.assertEqual(loaded.sha256, alpha.sha256)
            # committed file carries no per-char counts
            obj = json.loads(p.read_text(encoding="utf-8"))
            self.assertNotIn("counts", obj)

    def test_load_detects_tampered_sha(self):
        alpha = from_counts(Counter("abc"))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.json"
            save(alpha, p)
            obj = json.loads(p.read_text())
            obj["chars"] = ["a", "b"]  # tamper without fixing sha256
            p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                load(p)

    def test_scan_labels(self):
        with tempfile.TemporaryDirectory() as d:
            mp = Path(d) / "meta.jsonl"
            mp.write_text(
                '{"text": "ab"}\n{"text": "bc"}\n\n{"bad": 1}\n',
                encoding="utf-8",
            )
            counts = scan_labels([mp])
            self.assertEqual(counts, Counter({"b": 2, "a": 1, "c": 1}))


if __name__ == "__main__":
    unittest.main()
