# -*- coding: utf-8 -*-

"""Character-level CRNN + CTC for traditional-Mongolian single-column line OCR.

A few Conv2d layers downsample the tall-narrow single-column strip's WIDTH toward
1 while keeping the HEIGHT (reading axis) resolution high, yielding a ``[B, T, C]``
frame sequence with ``T ~ H / 4`` (must exceed target length ``U``); a 2-layer
BiLSTM + Linear emits ``alphabet + 1`` log-probs; CTC (blank = alphabet_size)
marginalizes the monotonic alignments.

Cold-start fixes (verified to overfit 4 lines to norm_CER 0.0): the blank logit
bias is initialized NEGATIVE so the untrained model does not collapse to the
all-blank attractor, and ``T = H/4`` (not the original ~9:1) keeps the frame:char
ratio out of the basin where all-blank dominates.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """Character-level CRNN: conv stem (squeeze width) -> BiLSTM -> CTC logits.

    The stem applies a stack of ``Conv -> BN -> ReLU`` blocks whose strides fully
    collapse the WIDTH (column -> single feature vector per height step) while
    subsampling the HEIGHT so ``T = H / 4`` along the reading axis. Width is
    pooled to 1 after the conv stack, giving a ``[B, T, C]`` sequence fed to a
    2-layer BiLSTM and a linear projection to ``n_classes = alphabet + 1`` (blank
    at the top index). The blank logit bias is initialized to ``blank_bias`` (< 0).
    """

    def __init__(
        self, n_classes: int, lstm_hidden: int = 256, blank_bias: float = -3.0
    ):
        super().__init__()

        def block(cin, cout, hstride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=(hstride, 2), padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(
            block(1, 32, hstride=2),    # H/2,  W/2
            block(32, 64, hstride=2),   # H/4,  W/4
            block(64, 128, hstride=1),  # H/4,  W/8
            block(128, 256, hstride=1), # H/4,  W/16
        )
        self.lstm = nn.LSTM(
            256, lstm_hidden, num_layers=2, batch_first=True, bidirectional=True
        )
        self.proj = nn.Linear(2 * lstm_hidden, n_classes)
        # Cold-start: bias the blank class DOWN so the model emits glyphs from
        # step 0 instead of falling into the all-blank attractor.
        nn.init.zeros_(self.proj.bias)
        self.proj.bias.data[n_classes - 1] = blank_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W]
        f = self.stem(x)                     # [B, C, T, W']
        f = f.mean(dim=3)                    # squeeze remaining width -> [B, C, T]
        f = f.transpose(1, 2)                # [B, T, C]
        f, _ = self.lstm(f)                  # [B, T, 2*hidden]
        logits = self.proj(f)                # [B, T, n_classes]
        return logits


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


__all__ = ["CRNN", "greedy_decode"]
