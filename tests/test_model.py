# -*- coding: utf-8 -*-
import unittest

try:
    import torch
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

if HAVE_TORCH:
    from mongocr.model import (CRNN, WIDTH_COLLAPSE_MODES, matched_mean_big_hidden,
                               resolve_arch_from_ckpt)

# This repo's frozen alphabet: 613 chars + 1 blank (see alphabet.json).
N_CLASSES = 614
IMG_H = 1024
IMG_W = 64
MEAN_HIDDEN = 384  # this repo's --lstm-hidden default, used by mean/concat/concat_proj
MEAN_BIG_HIDDEN = 462  # matched_mean_big_hidden(614, 64) — see mongocr/model.py


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestWidthCollapseDispatch(unittest.TestCase):
    """The most consequential new code: forward() must produce the right shape
    for all 4 width_collapse modes, and concat's fold must preserve width info
    in the documented channel-major/width-minor order (not silently corrupt or
    scramble it during the permute+reshape)."""

    def _hidden_for(self, mode: str) -> int:
        return MEAN_BIG_HIDDEN if mode == "mean_big" else MEAN_HIDDEN

    def test_forward_shape_all_four_modes(self):
        x = torch.randn(2, 1, IMG_H, IMG_W)
        for mode in WIDTH_COLLAPSE_MODES:
            with self.subTest(width_collapse=mode):
                model = CRNN(N_CLASSES, lstm_hidden=self._hidden_for(mode),
                             width_collapse=mode, img_w=IMG_W)
                model.eval()
                with torch.no_grad():
                    out = model(x)
                self.assertEqual(out.shape[0], 2)
                self.assertEqual(out.shape[2], N_CLASSES)
                # T = H/4 along the reading axis (module docstring); exact
                # value doesn't matter here, just that every mode agrees.
                self.assertGreater(out.shape[1], 0)

    def test_all_four_modes_agree_on_T(self):
        # width_collapse must not change the frame count T (only how the
        # residual width Wp is folded into the per-frame feature) — a
        # regression here would desync T from the CTC input_lengths contract.
        x = torch.randn(1, 1, IMG_H, IMG_W)
        frames = {}
        for mode in WIDTH_COLLAPSE_MODES:
            model = CRNN(N_CLASSES, lstm_hidden=self._hidden_for(mode),
                         width_collapse=mode, img_w=IMG_W)
            model.eval()
            with torch.no_grad():
                frames[mode] = model(x).shape[1]
        self.assertEqual(len(set(frames.values())), 1, frames)

    def test_concat_fold_is_order_preserving_channel_major_width_minor(self):
        # h[b,t] == flatten of f[b,:,t,:] (channel-major, width-minor), where
        # f = model.stem(x) is the pre-fold [B,C,T,Wp] conv feature map — the
        # exact claim in the width_collapse docstring and the forward()
        # dispatch comment. Verified two ways: (1) direct random-index probes
        # against the stem output, (2) end-to-end — manually replaying
        # stem->fold->lstm->proj on the SAME model reproduces forward()'s
        # actual output bit-for-bit, so the fold under test is provably the
        # one forward() actually uses, not a lookalike reimplementation.
        torch.manual_seed(0)
        model = CRNN(N_CLASSES, lstm_hidden=MEAN_HIDDEN, width_collapse="concat",
                     img_w=IMG_W)
        model.eval()
        x = torch.randn(2, 1, IMG_H, IMG_W)
        with torch.no_grad():
            f = model.stem(x)  # [B, C, T, Wp]
            b, c, t, wp = f.shape
            self.assertEqual(wp, model.width_p)
            folded = f.permute(0, 2, 1, 3).reshape(b, t, c * wp)

            # (1) direct probes: h[b,t, ci*Wp + wi] == f[b, ci, t, wi]
            for bi, ti, ci, wi in [(0, 0, 0, 0), (0, 0, 0, wp - 1),
                                   (0, 0, c - 1, 0), (1, t - 1, c - 1, wp - 1),
                                   (1, t // 2, c // 2, wp // 2)]:
                with self.subTest(bi=bi, ti=ti, ci=ci, wi=wi):
                    self.assertTrue(torch.equal(
                        f[bi, ci, ti, wi], folded[bi, ti, ci * wp + wi]))

            # a whole per-timestep row equals the [C,Wp] slice flattened
            # row-major (channel-major since C is the outer/slower axis)
            self.assertTrue(torch.equal(folded[0, 0], f[0, :, 0, :].reshape(-1)))

            # (2) end-to-end: forward() == manual stem -> fold -> lstm -> proj
            lstm_out, _ = model.lstm(folded)
            expected_logits = model.proj(lstm_out)
            actual_logits = model(x)
        self.assertTrue(torch.equal(expected_logits, actual_logits))

    def test_concat_proj_lstm_matches_mean_param_count(self):
        # Info control property from the module docstring: concat_proj's
        # LSTM+proj sub-block must be parameter-identical to mean's (only the
        # fold_proj Linear is extra) — this is what makes it the "clean"
        # width-info-isolation control.
        mean = CRNN(N_CLASSES, lstm_hidden=MEAN_HIDDEN, width_collapse="mean",
                   img_w=IMG_W)
        cproj = CRNN(N_CLASSES, lstm_hidden=MEAN_HIDDEN, width_collapse="concat_proj",
                    img_w=IMG_W)
        mean_lstm_proj = (sum(p.numel() for p in mean.lstm.parameters())
                          + sum(p.numel() for p in mean.proj.parameters()))
        cproj_lstm_proj = (sum(p.numel() for p in cproj.lstm.parameters())
                           + sum(p.numel() for p in cproj.proj.parameters()))
        self.assertEqual(mean_lstm_proj, cproj_lstm_proj)
        # concat_proj's TOTAL param count must still exceed mean's (the extra
        # fold_proj Linear), so it isn't secretly bit-identical either.
        self.assertGreater(
            sum(p.numel() for p in cproj.parameters()),
            sum(p.numel() for p in mean.parameters()))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestMeanBigGuard(unittest.TestCase):
    """BLOCKER fix: mean_big at an unmatched lstm_hidden is a silently-void
    capacity control (bit-identical param count to mean) — CRNN must refuse to
    construct it instead of letting an ablation run quietly compare mean
    against a same-capacity mean_big."""

    def test_matched_hidden_is_462_for_this_repos_alphabet(self):
        self.assertEqual(matched_mean_big_hidden(N_CLASSES, IMG_W), MEAN_BIG_HIDDEN)

    def test_mean_big_at_384_raises(self):
        # 384 is the "mean" default lstm_hidden — this is exactly the
        # silently-void configuration the guard exists to reject.
        with self.assertRaises(ValueError) as ctx:
            CRNN(N_CLASSES, lstm_hidden=384, width_collapse="mean_big", img_w=IMG_W)
        self.assertIn("mean_big", str(ctx.exception))
        self.assertIn("462", str(ctx.exception))

    def test_mean_big_at_matched_hidden_constructs(self):
        model = CRNN(N_CLASSES, lstm_hidden=MEAN_BIG_HIDDEN, width_collapse="mean_big",
                     img_w=IMG_W)
        n_params = sum(p.numel() for p in model.parameters())
        concat = CRNN(N_CLASSES, lstm_hidden=MEAN_HIDDEN, width_collapse="concat",
                      img_w=IMG_W)
        concat_params = sum(p.numel() for p in concat.parameters())
        # exact numbers from the review: mean_big@462=8,747,918,
        # concat@384=8,737,574 (+0.12%)
        self.assertEqual(n_params, 8_747_918)
        self.assertEqual(concat_params, 8_737_574)
        self.assertAlmostEqual((n_params - concat_params) / concat_params, 0.0012,
                               places=3)

    def test_mean_big_param_count_differs_from_mean_at_matched_hidden(self):
        # The defect this guard closes: at the matched hidden, mean_big must
        # NOT be param-identical to mean (that would mean the guard chose a
        # no-op hidden value).
        mean = CRNN(N_CLASSES, lstm_hidden=MEAN_HIDDEN, width_collapse="mean",
                   img_w=IMG_W)
        mean_big = CRNN(N_CLASSES, lstm_hidden=MEAN_BIG_HIDDEN,
                        width_collapse="mean_big", img_w=IMG_W)
        self.assertNotEqual(
            sum(p.numel() for p in mean.parameters()),
            sum(p.numel() for p in mean_big.parameters()))

    def test_other_modes_unaffected_by_guard(self):
        # The guard is mean_big-specific; mean/concat/concat_proj must accept
        # the plain 384 default same as before.
        for mode in ("mean", "concat", "concat_proj"):
            with self.subTest(width_collapse=mode):
                CRNN(N_CLASSES, lstm_hidden=384, width_collapse=mode, img_w=IMG_W)

    def test_matched_hidden_generalizes_to_other_n_classes(self):
        # matched_mean_big_hidden is a closed-form solve, not a hardcoded
        # constant — a different alphabet size must give a different (still
        # constructible) matched hidden.
        h = matched_mean_big_hidden(300, IMG_W)
        self.assertNotEqual(h, MEAN_BIG_HIDDEN)
        CRNN(300, lstm_hidden=h, width_collapse="mean_big", img_w=IMG_W)
        with self.assertRaises(ValueError):
            CRNN(300, lstm_hidden=384, width_collapse="mean_big", img_w=IMG_W)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestLegacyCheckpointLoad(unittest.TestCase):
    """Backward-compat: the production crnn_x2.pt was saved by the OLD
    train_crnn.py, before --width-collapse/--lstm-hidden/--img-w existed as
    ablation flags. Loading it must resolve to width_collapse='mean',
    lstm_hidden=384 — never crash, never silently pick the wrong arch."""

    def _cli_defaults(self):
        # Mirrors eval_crnn.py / eval_confusion.py's argparse defaults.
        return dict(cli_lstm_hidden=384, cli_width_collapse=None, cli_img_w=64)

    def test_args_present_but_missing_new_keys(self):
        # The realistic legacy shape: ckpt["args"] exists (train_crnn.py has
        # always saved vars(args)) but predates the 3 new flags entirely.
        legacy_args = {
            "shards": ["x"], "alphabet": "a.json", "val_threshold": 1,
            "test_threshold": 2, "gap": 200, "steps": 100, "batch_size": 8,
            "lr": 3e-4, "img_h": 1024, "img_w": 64, "lstm_hidden": 384,
            "blank_bias": -3.0, "device": "cpu", "save": "crnn_x2.pt",
        }
        ckpt = {"state_dict": {}, "alphabet": None, "args": legacy_args, "step": 100}
        hidden, wc, img_w = resolve_arch_from_ckpt(ckpt, **self._cli_defaults())
        self.assertEqual((hidden, wc, img_w), (384, "mean", 64))

    def test_args_key_entirely_absent(self):
        ckpt = {"state_dict": {}, "alphabet": None}
        hidden, wc, img_w = resolve_arch_from_ckpt(ckpt, **self._cli_defaults())
        self.assertEqual((hidden, wc, img_w), (384, "mean", 64))

    def test_args_key_present_but_none(self):
        # Defensive: some hypothetical save path could write args=None rather
        # than omitting the key; `ckpt.get("args") or {}` must not crash on
        # calling .get() on None.
        ckpt = {"state_dict": {}, "alphabet": None, "args": None}
        hidden, wc, img_w = resolve_arch_from_ckpt(ckpt, **self._cli_defaults())
        self.assertEqual((hidden, wc, img_w), (384, "mean", 64))

    def test_legacy_resolved_arch_actually_loads_a_legacy_state_dict(self):
        # End-to-end: build a "mean"/384 CRNN (the only architecture that
        # existed pre-ablation), save its state_dict under a legacy-shaped
        # checkpoint (args missing the 3 new keys), resolve, rebuild, and
        # confirm load_state_dict succeeds (no shape mismatch / crash).
        legacy_model = CRNN(N_CLASSES, lstm_hidden=384, width_collapse="mean",
                            img_w=64)
        ckpt = {
            "state_dict": legacy_model.state_dict(),
            "alphabet": None,
            "args": {"lstm_hidden": 384, "img_h": 1024, "img_w": 64,
                     "save": "crnn_x2.pt"},  # no "width_collapse" key
            "step": 100,
        }
        hidden, wc, img_w = resolve_arch_from_ckpt(ckpt, **self._cli_defaults())
        self.assertEqual((hidden, wc), (384, "mean"))
        rebuilt = CRNN(N_CLASSES, lstm_hidden=hidden, width_collapse=wc, img_w=img_w)
        rebuilt.load_state_dict(ckpt["state_dict"])  # must not raise

    def test_new_style_checkpoint_still_honors_saved_width_collapse(self):
        # Non-legacy path must be unaffected: a checkpoint that DOES carry
        # width_collapse='concat' must resolve to concat, not fall back to
        # mean, regardless of CLI flags.
        ckpt = {"args": {"lstm_hidden": 384, "width_collapse": "concat", "img_w": 64}}
        hidden, wc, img_w = resolve_arch_from_ckpt(ckpt, **self._cli_defaults())
        self.assertEqual((hidden, wc, img_w), (384, "concat", 64))

    def test_cli_width_collapse_override_used_only_when_ckpt_silent(self):
        # If the checkpoint has an args dict but it's missing width_collapse,
        # AND the caller explicitly passed a non-None CLI --width-collapse,
        # the CLI value must be honored (documented fallback-of-last-resort
        # behavior, used when a legacy checkpoint's actual mode is known out
        # of band).
        ckpt = {"args": {"lstm_hidden": 384, "img_w": 64}}  # no width_collapse
        hidden, wc, img_w = resolve_arch_from_ckpt(
            ckpt, cli_lstm_hidden=384, cli_width_collapse="concat_proj", cli_img_w=64)
        self.assertEqual(wc, "concat_proj")


if __name__ == "__main__":
    unittest.main()
