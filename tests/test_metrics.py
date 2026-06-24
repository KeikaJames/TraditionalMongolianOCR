# -*- coding: utf-8 -*-
import unittest

from mongocr.metrics import cer, edit_distance, nominal_fold, ocr_report, wer


class TestMetrics(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance("abc", "abc"), 0)
        self.assertEqual(edit_distance("abc", "abd"), 1)
        self.assertEqual(edit_distance("abc", "ab"), 1)
        self.assertEqual(edit_distance("", "abc"), 3)

    def test_nominal_fold_strips_variation(self):
        # FVS (U+180B), MVS (U+180E), ZWJ (U+200D), BOM dropped; NNBSP -> space
        raw = "a᠋b᠎c‍d e﻿"
        self.assertEqual(nominal_fold(raw), "abcd e")

    def test_cer_normalized_ignores_encoding_noise(self):
        # same nominal text, differ only by FVS -> normalized CER 0, raw CER > 0
        ref = "ᠠᠡ"
        pred = "ᠠ᠋ᠡ"
        self.assertEqual(cer([pred], [ref], normalize=True), 0.0)
        self.assertGreater(cer([pred], [ref], normalize=False), 0.0)

    def test_cer_counts_real_errors(self):
        self.assertAlmostEqual(cer(["abcd"], ["abce"]), 0.25)

    def test_wer(self):
        self.assertAlmostEqual(wer(["a b c"], ["a b d"]), 1 / 3)

    def test_ocr_report_fields(self):
        rep = ocr_report(["abc", "xy"], ["abc", "xz"])
        self.assertEqual(rep.n, 2)
        self.assertEqual(rep.line_exact, 0.5)
        self.assertGreater(rep.norm_cer, 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ocr_report(["a"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
