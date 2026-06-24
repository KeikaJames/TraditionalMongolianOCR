# -*- coding: utf-8 -*-

"""CTC loss over the CRNN frame axis (thin wrapper over ``F.ctc_loss``)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int,
    input_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """CTC loss.

    Args:
        log_probs: ``[B, T, V]`` already-log-softmaxed scores (``V == alphabet + 1``).
        targets: ``[B, U_max]`` padded transcription ids; padding past
            ``target_lengths`` is ignored by CTC.
        target_lengths: ``[B]`` true length ``U`` of each target; each ``U <= T``.
        blank: index of the CTC blank class.
        input_lengths: ``[B]`` frame counts; defaults to full ``T`` per row.

    ``zero_infinity=True`` degrades a bad ``U > T`` row to zero contribution
    instead of NaN-poisoning the batch. Reduction is ``mean`` over the batch.
    """
    bsz, frames, _ = log_probs.shape
    if input_lengths is None:
        input_lengths = torch.full(
            (bsz,), frames, dtype=torch.long, device=log_probs.device
        )
    # F.ctc_loss expects [T, B, V] log-probs.
    lp = log_probs.transpose(0, 1)
    return F.ctc_loss(
        lp,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        zero_infinity=True,
    )


__all__ = ["ctc_loss"]
