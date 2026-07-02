# -*- coding: utf-8 -*-
import unittest

from mongocr.metrics import edit_distance
from scripts.eval_confusion import (
    A_CLASS_PAIRS,
    B_CLASS_PAIRS,
    PairStats,
    align_ops,
    edit_distance_via_ops,
    score_confusion,
)


class TestAlignOps(unittest.TestCase):
    """align_ops must agree with mongocr.metrics.edit_distance on distance (same
    cost model), and its backtrace must break ties toward substitution."""

    def test_agrees_with_edit_distance(self):
        cases = [
            ("abc", "abc"), ("abc", "abd"), ("abc", "ab"), ("", "abc"),
            ("abc", ""), ("abcde", "aXcYe"), ("kitten", "sitting"),
            ("t", "d"), ("ᠲ", "ᠳ"), ("ᠠᠡᠢ", "ᠡᠠᠶ"), ("aa", "a"), ("a", "aa"),
            ("", ""), ("同じ", "同じ"),
        ]
        for ref, hyp in cases:
            with self.subTest(ref=ref, hyp=hyp):
                self.assertEqual(edit_distance_via_ops(ref, hyp), edit_distance(ref, hyp))

    def test_single_char_swap_is_one_substitution(self):
        ops = align_ops("ᠲ", "ᠳ")
        self.assertEqual(ops, [("sub", "ᠲ", "ᠳ")])

    def test_tie_breaks_toward_substitution_not_del_ins(self):
        # ref='ab' -> hyp='ba': del(a)+ins(a-at-end) and 2 subs both cost 2 —
        # must prefer the all-substitution backtrace.
        ops = align_ops("ab", "ba")
        self.assertEqual(ops, [("sub", "a", "b"), ("sub", "b", "a")])
        self.assertNotIn("del", [k for k, _r, _h in ops])
        self.assertNotIn("ins", [k for k, _r, _h in ops])

    def test_pure_insertion_and_deletion(self):
        self.assertEqual(align_ops("", "abc"),
                         [("ins", "", "a"), ("ins", "", "b"), ("ins", "", "c")])
        self.assertEqual(align_ops("abc", ""),
                         [("del", "a", ""), ("del", "b", ""), ("del", "c", "")])

    def test_match_ops_are_equal_chars(self):
        ops = align_ops("abc", "abc")
        self.assertTrue(all(kind == "match" and r == h for kind, r, h in ops))

    def test_ops_cover_full_strings(self):
        ref, hyp = "kitten", "sitting"
        ops = align_ops(ref, hyp)
        rebuilt_ref = "".join(r for _k, r, _h in ops)
        rebuilt_hyp = "".join(h for _k, _r, h in ops)
        self.assertEqual(rebuilt_ref, ref)
        self.assertEqual(rebuilt_hyp, hyp)


class TestPairStats(unittest.TestCase):
    def test_directed_counts_not_summed_prematurely(self):
        stats = PairStats([("ᠲ", "ᠳ")])
        # ref has t, model predicts d (t->d), twice; and once d->t.
        stats.update("ᠲᠲ", [("sub", "ᠲ", "ᠳ"), ("match", "ᠲ", "ᠲ")])
        stats.update("ᠳ", [("sub", "ᠳ", "ᠲ")])
        row = stats.pair_row("ᠲ", "ᠳ")
        self.assertEqual(row["ref_to_hyp_count"], 1)  # t->d
        self.assertEqual(row["hyp_to_ref_count"], 1)  # d->t
        self.assertEqual(row["undirected_count"], 2)
        self.assertEqual(row["ref_occurrences"], 3)  # 2 t's + 1 d in refs

    def test_rate_normalizes_by_ref_occurrence(self):
        stats = PairStats([("ᠲ", "ᠳ")])
        stats.update("ᠲ" * 10, [("sub", "ᠲ", "ᠳ")] + [("match", "ᠲ", "ᠲ")] * 9)
        row = stats.pair_row("ᠲ", "ᠳ")
        self.assertAlmostEqual(row["rate"], 1 / 10)

    def test_zero_occurrence_pair_has_zero_rate_no_crash(self):
        stats = PairStats([("ᠲ", "ᠳ")])
        row = stats.pair_row("ᠲ", "ᠳ")
        self.assertEqual(row["undirected_count"], 0)
        self.assertEqual(row["rate"], 0.0)

    def test_pooled_rate_sums_across_pairs(self):
        stats = PairStats(B_CLASS_PAIRS)
        stats.update("ᠲᠬᠠᠢ", [
            ("sub", "ᠲ", "ᠳ"), ("sub", "ᠬ", "ᠭ"),
            ("match", "ᠠ", "ᠠ"), ("match", "ᠢ", "ᠢ"),
        ])
        subs, denom, rate = stats.pooled_rate()
        self.assertEqual(subs, 2)
        self.assertEqual(denom, 4)  # one occurrence of each of the 4 B-CLASS chars
        self.assertAlmostEqual(rate, 0.5)


class TestScoreConfusion(unittest.TestCase):
    def test_a_class_and_b_class_are_disjoint_char_sets(self):
        a_chars = {c for pair in A_CLASS_PAIRS for c in pair}
        b_chars = {c for pair in B_CLASS_PAIRS for c in pair}
        self.assertFalse(a_chars & b_chars)

    def test_all_pair_codepoints_match_spec(self):
        # Guards against silently editing the wrong codepoint later.
        expect_a = {(0x1823, 0x1824), (0x1825, 0x1826)}
        expect_b = {(0x1832, 0x1833), (0x182C, 0x182D), (0x1820, 0x1821), (0x1822, 0x1836)}
        got_a = {(ord(a), ord(b)) for a, b in A_CLASS_PAIRS}
        got_b = {(ord(a), ord(b)) for a, b in B_CLASS_PAIRS}
        self.assertEqual(got_a, expect_a)
        self.assertEqual(got_b, expect_b)

    def test_end_to_end_on_synthetic_lines(self):
        # Perfect predictions -> zero substitutions anywhere.
        refs = ["ᠲᠡᠷᠡ", "ᠭᠠᠵᠠᠷ"]
        preds = list(refs)
        conf = score_confusion(preds, refs)
        a_subs, _a_denom, a_rate = conf["a_class"].pooled_rate()
        b_subs, _b_denom, b_rate = conf["b_class"].pooled_rate()
        self.assertEqual(a_subs, 0)
        self.assertEqual(b_subs, 0)
        self.assertEqual(a_rate, 0.0)
        self.assertEqual(b_rate, 0.0)

    def test_end_to_end_counts_a_deliberate_b_class_confusion(self):
        refs = ["ᠲᠡᠷᠡ"]
        preds = ["ᠳᠡᠷᠡ"]  # t -> d substitution at position 0
        conf = score_confusion(preds, refs)
        row = conf["b_class"].pair_row("ᠲ", "ᠳ")
        self.assertEqual(row["ref_to_hyp_count"], 1)
        self.assertEqual(row["undirected_count"], 1)

    def test_nominal_fold_applied_before_alignment(self):
        # FVS (U+180B) differs between ref/pred but must not count as a
        # substitution anywhere once both sides are nominal-folded.
        refs = ["ᠠ᠋ᠡ"]
        preds = ["ᠠᠡ"]
        conf = score_confusion(preds, refs)
        a_subs, _d, _r = conf["a_class"].pooled_rate()
        b_subs, _d2, _r2 = conf["b_class"].pooled_rate()
        self.assertEqual(a_subs, 0)
        self.assertEqual(b_subs, 0)


if __name__ == "__main__":
    unittest.main()
