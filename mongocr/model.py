# -*- coding: utf-8 -*-

"""Character-level CRNN + CTC for traditional-Mongolian single-column line OCR.

A few Conv2d layers downsample the tall-narrow single-column strip's WIDTH toward
1 while keeping the HEIGHT (reading axis) resolution high, yielding a
``[B, C=256, T=256, Wp=4]`` feature map (``T ~ H / 4`` must exceed target length
``U``); the remaining width ``Wp`` is collapsed by one of four ``width_collapse``
modes (below) into a ``[B, T, C]`` frame sequence; a 2-layer BiLSTM + Linear emits
``alphabet + 1`` log-probs; CTC (blank = alphabet_size) marginalizes the monotonic
alignments.

Cold-start fixes (verified to overfit 4 lines to norm_CER 0.0): the blank logit
bias is initialized NEGATIVE so the untrained model does not collapse to the
all-blank attractor, and ``T = H/4`` (not the original ~9:1) keeps the frame:char
ratio out of the basin where all-blank dominates.

``width_collapse`` modes (ablation: does preserving Wp help, or is it just extra
LSTM capacity?):

- ``"mean"`` (default): current/original behavior, bit-identical. ``f.mean(dim=3)``
  discards Wp, LSTM input width stays ``C=256``. Also the ``crnn_x2`` checkpoint's
  architecture — must stay load-compatible (no state_dict key renaming).
- ``"concat"``: order-preserving fold of Wp into the channel dim instead of
  averaging it away, so ALL width info is kept: ``[B,C,T,Wp] -> permute(0,2,1,3)
  -> [B,T,C,Wp] -> reshape(B,T,C*Wp)``, giving per-timestep ordering
  ``[c0w0,c0w1,c0w2,c0w3, c1w0,c1w1,c1w2,c1w3, ...]`` (channel-major, width-minor
  — an arbitrary but fixed convention the LSTM learns against). ``.reshape()`` is
  required, not ``.view()``: the permute output is non-contiguous and ``.view()``
  raises ``RuntimeError``. LSTM input widens to ``C*Wp=1024`` -> more capacity
  AND more info, confounded by design.
- ``"concat_proj"``: identical fold as ``"concat"``, immediately projected back to
  ``C=256`` via a new ``Linear(C*Wp, C)`` so the LSTM+proj sub-blocks are
  parameter-for-parameter identical to ``"mean"``'s. Info control: isolates
  whether the width info alone (not a wider LSTM) helps.
- ``"mean_big"``: same forward graph as ``"mean"`` (zero width info added) but
  meant to be paired with a larger ``lstm_hidden`` (``matched_mean_big_hidden``,
  462 at this repo's ``n_classes=614, img_w=64``) so total params roughly match
  ``"concat"``'s. Capacity control: no new ops beyond ``"mean"``'s. ``CRNN``
  raises ``ValueError`` at construction if ``"mean_big"`` is launched with an
  unmatched ``lstm_hidden`` (see ``CRNN.__init__``) — at the default
  ``lstm_hidden=384`` (same as ``"mean"``), ``"mean_big"`` silently degenerates
  to a bit-identical param count to ``"mean"``, voiding the capacity control.

Caveat — ``"mean_big"`` is a cruder capacity control than ``"concat_proj"``:
matching total param count is not the same as matching *where* capacity is
added. ``"concat"`` (and by extension what ``"mean_big"`` is trying to mimic)
adds capacity at the LSTM's layer-0 INPUT-facing weights (``lstm_input`` widens
4x, ``C*Wp=1024`` vs ``C=256``); ``"mean_big"`` instead spreads the same
param budget across a wider HIDDEN state (affecting every gate, both layers,
and the output projection). This is a scientific limitation of ``"mean_big"``
as an ablation control, not a code bug — a clean win/loss on ``"mean_big"``
vs ``"mean"`` conflates "more capacity" with "more capacity shaped like
concat's". ``"concat_proj"`` is the cleaner control for isolating width-info
value alone: its LSTM+proj sub-block is parameter-for-parameter IDENTICAL to
``"mean"``'s (only a ``Linear(1024, 256)`` fold projection is added ahead of
it), so any delta vs ``"mean"`` cannot be explained by LSTM capacity at all.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

WIDTH_COLLAPSE_MODES = ("mean", "concat", "concat_proj", "mean_big")


def _lstm_proj_params(lstm_input: int, hidden: int, n_classes: int) -> int:
    """Closed-form param count of ``nn.LSTM(lstm_input, hidden, num_layers=2,
    bidirectional=True)`` + ``nn.Linear(2*hidden, n_classes)`` — the two modules
    whose sizes vary across ``width_collapse`` modes (the conv stem is identical
    for all four). Verified exact against ``sum(p.numel() ...)`` for several
    ``(lstm_input, hidden)`` pairs; used to derive ``matched_mean_big_hidden``
    without instantiating a throwaway model.

    Per (layer, direction): ``weight_ih[4H, layer_input] + weight_hh[4H, H] +
    bias_ih[4H] + bias_hh[4H]``; ``layer_input`` is ``lstm_input`` for layer 0
    and ``2*hidden`` for layer 1 (the two directions' outputs are concatenated).
    """
    lstm = 0
    for layer_input in (lstm_input, 2 * hidden):
        for _direction in (0, 1):
            lstm += 4 * hidden * layer_input + 4 * hidden * hidden + 8 * hidden
    proj = 2 * hidden * n_classes + n_classes  # Linear(2*hidden, n_classes)
    return lstm + proj


def matched_mean_big_hidden(n_classes: int, img_w: int, stem_out_c: int = 256) -> int:
    """The ``lstm_hidden`` that makes ``"mean_big"``'s total param count closest
    to ``"concat"``'s, for the given ``(n_classes, img_w)`` — re-derived here
    (not hardcoded) so it stays correct if either changes.

    Both modes share an identical conv stem, so only the LSTM+proj sub-block
    needs to match: solve ``_lstm_proj_params(stem_out_c, H, n_classes) ==
    _lstm_proj_params(stem_out_c * width_p, base_hidden=384, n_classes)`` for
    ``H``. The LHS is quadratic in ``H`` (from the ``4*H*H`` terms per
    direction/layer), so this is a quadratic-formula solve, not a search;
    ``base_hidden=384`` is the repo's default/original ``lstm_hidden`` that the
    ablation's ``"concat"`` runs are launched with. Round to the nearest int
    (param counts cannot match exactly — ``"concat"`` and ``"mean_big"`` add
    capacity through different weight shapes; see the module docstring's
    "scientific limitation" note).
    """
    base_hidden = 384
    width_p = img_w
    for _ in range(4):
        width_p = (width_p + 1) // 2
    target = _lstm_proj_params(stem_out_c * width_p, base_hidden, n_classes)

    # _lstm_proj_params(stem_out_c, H, n_classes) expands to a*H^2 + b*H + c
    # (verified via sympy.expand/Poly, not hand algebra — the H*H cross terms
    # from 2 layers x 2 directions are easy to undercount by hand):
    #   a = 32
    #   b = 2*n_classes + 8*stem_out_c + 32
    #   c = n_classes
    a = 32.0
    b = 2.0 * n_classes + 8.0 * stem_out_c + 32.0
    c = float(n_classes) - target
    disc = b * b - 4 * a * c
    if disc < 0:
        raise ValueError(
            f"no real solution for matched mean_big hidden at "
            f"n_classes={n_classes}, img_w={img_w}"
        )
    h = (-b + math.sqrt(disc)) / (2 * a)
    return max(1, round(h))


class CRNN(nn.Module):
    """Character-level CRNN: conv stem (squeeze width) -> BiLSTM -> CTC logits.

    The stem applies a stack of ``Conv -> BN -> ReLU`` blocks whose strides
    subsample the HEIGHT so ``T = H / 4`` along the reading axis while reducing
    the WIDTH to a small residual ``Wp`` (``Wp=4`` at the default ``img_w=64``).
    ``width_collapse`` selects how that residual ``Wp`` is turned into a
    ``[B, T, C]`` sequence (see module docstring for the four modes) before the
    2-layer BiLSTM and linear projection to ``n_classes = alphabet + 1`` (blank at
    the top index). The blank logit bias is initialized to ``blank_bias`` (< 0).

    ``lstm_hidden`` sets the LSTM hidden size for the ``"mean"``, ``"concat"``,
    and ``"concat_proj"`` modes; ``"mean_big"`` is the same forward graph as
    ``"mean"`` but MUST be launched with ``lstm_hidden ==
    matched_mean_big_hidden(n_classes, img_w)`` (see module docstring) so its
    total param count param-matches ``"concat"``'s — any other value raises
    ``ValueError`` at construction, since ``"mean_big"`` at an unmatched hidden
    (e.g. the ``"mean"`` default of 384) is a silently-void capacity control
    (bit-identical param count to ``"mean"``).
    """

    def __init__(
        self,
        n_classes: int,
        lstm_hidden: int = 256,
        blank_bias: float = -3.0,
        width_collapse: str = "mean",
        img_w: int = 64,
    ):
        super().__init__()
        if width_collapse not in WIDTH_COLLAPSE_MODES:
            raise ValueError(
                f"width_collapse must be one of {WIDTH_COLLAPSE_MODES}, "
                f"got {width_collapse!r}"
            )
        self.width_collapse = width_collapse

        if width_collapse == "mean_big":
            # BLOCKER guard: "mean_big" at an unmatched lstm_hidden is a
            # silently-void capacity control — e.g. at the "mean" default of
            # 384 it produces a param count BIT-IDENTICAL to "mean" (both are
            # the same forward graph; "mean_big" only differs by lstm_hidden).
            # Fail loudly instead of letting an ablation run quietly compare
            # "mean" against a same-capacity "mean_big" and call it a capacity
            # control. matched_mean_big_hidden is re-derived from
            # (n_classes, img_w), not hardcoded, so this stays correct if
            # either changes.
            expected = matched_mean_big_hidden(n_classes, img_w)
            if lstm_hidden != expected:
                raise ValueError(
                    f"width_collapse='mean_big' requires lstm_hidden="
                    f"{expected} (param-matched to 'concat' at "
                    f"n_classes={n_classes}, img_w={img_w}), got "
                    f"lstm_hidden={lstm_hidden}. 'mean_big' is the same "
                    f"forward graph as 'mean' — at any other lstm_hidden "
                    f"(e.g. 'mean''s default of 384) its param count either "
                    f"matches 'mean' exactly (voiding the capacity control) "
                    f"or is an unmatched, uninterpretable size. If "
                    f"n_classes or img_w changed, re-derive the matched "
                    f"value via matched_mean_big_hidden(n_classes, img_w) "
                    f"instead of reusing a stale constant."
                )

        def block(cin, cout, hstride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=(hstride, 2), padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        stem_out_c = 256
        self.stem = nn.Sequential(
            block(1, 32, hstride=2),           # H/2,  W/2
            block(32, 64, hstride=2),          # H/4,  W/4
            block(64, 128, hstride=1),         # H/4,  W/8
            block(128, stem_out_c, hstride=1), # H/4,  W/16
        )

        # Residual width after the stem's four stride-2-on-width Conv2d blocks
        # (padding=1, kernel=3 -> each halves W, ceil-rounded): Wp=4 at the
        # default img_w=64. Only "concat"/"concat_proj" need it, to size the
        # channel-major width-minor fold ([B,C,T,Wp] -> [B,T,C*Wp]).
        width_p = img_w
        for _ in range(4):
            width_p = (width_p + 1) // 2
        self.width_p = width_p

        if width_collapse == "concat":
            lstm_input = stem_out_c * width_p
            self.fold_proj = None
        elif width_collapse == "concat_proj":
            self.fold_proj = nn.Linear(stem_out_c * width_p, stem_out_c)
            lstm_input = stem_out_c
        else:  # "mean" / "mean_big": identical graph, LSTM input stays C=256
            self.fold_proj = None
            lstm_input = stem_out_c

        self.lstm = nn.LSTM(
            lstm_input, lstm_hidden, num_layers=2, batch_first=True, bidirectional=True
        )
        self.proj = nn.Linear(2 * lstm_hidden, n_classes)
        # Cold-start: bias the blank class DOWN so the model emits glyphs from
        # step 0 instead of falling into the all-blank attractor.
        nn.init.zeros_(self.proj.bias)
        self.proj.bias.data[n_classes - 1] = blank_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W]
        f = self.stem(x)  # [B, C, T, Wp]

        # width_collapse dispatch: "concat"/"concat_proj" fold Wp into the
        # LSTM input (below); "mean"/"mean_big" discard it via f.mean(dim=3)
        # and differ only in lstm_hidden. Caveat (module docstring has the
        # full argument): "mean_big" matches "concat"'s TOTAL param count but
        # not WHERE capacity is added (hidden state, not layer-0 input) — a
        # cruder control than "concat_proj", whose LSTM+proj is
        # parameter-identical to "mean"'s.
        if self.width_collapse in ("concat", "concat_proj"):
            b, c, t, wp = f.shape
            if wp != self.width_p:
                raise RuntimeError(
                    f"{self.width_collapse}: input width produced Wp={wp} but "
                    f"this model was built for Wp={self.width_p} (img_w passed "
                    f"to CRNN() must match the actual input img_w)."
                )
            # [B,C,T,Wp] -> [B,T,C,Wp] (NOT contiguous) -> reshape (not .view(),
            # which would raise on the non-contiguous strides) -> [B,T,C*Wp],
            # channel-major / width-minor per timestep:
            # h[b,t] = [c0w0,c0w1,...,c0w{Wp-1}, c1w0,c1w1,...,c1w{Wp-1}, ...]
            f = f.permute(0, 2, 1, 3).reshape(b, t, c * wp)
            if self.width_collapse == "concat_proj":
                f = self.fold_proj(f)  # [B,T,C*Wp] -> [B,T,C]
        else:  # "mean" / "mean_big": identical graph, differ only in lstm_hidden
            f = f.mean(dim=3)      # squeeze remaining width -> [B, C, T]
            f = f.transpose(1, 2)  # [B, T, C]

        f, _ = self.lstm(f)    # [B, T, 2*hidden]
        logits = self.proj(f)  # [B, T, n_classes]
        return logits


def resolve_arch_from_ckpt(
    ckpt: dict, *, cli_lstm_hidden: int, cli_width_collapse: str | None, cli_img_w: int
) -> tuple[int, str, int]:
    """Resolve ``(lstm_hidden, width_collapse, img_w)`` to build a ``CRNN`` that
    matches a checkpoint's actual trained architecture.

    ``train_crnn.py`` saves the full argparse ``Namespace`` under
    ``ckpt["args"]``, so those values are always preferred over CLI flags/
    defaults when present — loading a ``width_collapse="concat"`` checkpoint
    as the default ``"mean"`` would mismatch state_dict shapes (LSTM input
    1024 vs 256) and crash ``load_state_dict``, or worse, silently produce a
    same-shape-but-wrong-capacity model for the ``mean``/``mean_big`` pair.

    Legacy checkpoints (saved before ``--width-collapse``/``--lstm-hidden``/
    ``--img-w`` existed as flags, e.g. the production ``crnn_x2.pt``) either
    lack the ``"args"`` key entirely or have an ``"args"`` dict missing these
    specific keys — ``ckpt_args.get(key, cli_value)`` falls through to the
    CLI value for each key independently in that case. Both eval scripts'
    CLI default for ``--width-collapse`` is ``None`` (not a fabricated
    "which mode was this trained with" guess), so ``cli_width_collapse or
    "mean"`` is the final fallback: a legacy checkpoint has no
    ``width_collapse`` concept because ``"mean"`` was the only architecture
    that existed, so resolving it to ``"mean"`` is correct, not a default of
    convenience.
    """
    ckpt_args = ckpt.get("args") or {}
    lstm_hidden = ckpt_args.get("lstm_hidden", cli_lstm_hidden)
    width_collapse = ckpt_args.get("width_collapse", cli_width_collapse or "mean")
    img_w = ckpt_args.get("img_w", cli_img_w)
    return lstm_hidden, width_collapse, img_w


def greedy_decode(
    log_probs: torch.Tensor, blank: int, alphabet: list[str]
) -> list[str]:
    """Collapse-repeats greedy CTC decode of ``[B, T, V]`` log-probs -> strings."""
    ids = log_probs.argmax(dim=-1)  # [B, T]
    out = []
    for row in ids.tolist():
        chars = []
        prev = -1
        for c in row:
            if c != prev and c != blank:
                chars.append(alphabet[c])
            prev = c
        out.append("".join(chars))
    return out


__all__ = ["CRNN", "WIDTH_COLLAPSE_MODES", "greedy_decode", "matched_mean_big_hidden",
          "resolve_arch_from_ckpt"]
