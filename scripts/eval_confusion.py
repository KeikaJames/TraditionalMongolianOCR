# -*- coding: utf-8 -*-

"""Per-pair confusion eval for the width_collapse ablation.

Decodes a checkpoint on the held-out split (same selection path as
``eval_crnn.py``: ``src_doc_bands`` + ``build_pipeline(..., training=False)``)
and, on top of the usual headline CER/WER/exact-line numbers, reports directed
and undirected substitution counts + occurrence-normalized rates for two fixed
sets of near-homograph Mongolian character pairs:

- **A-CLASS** (negative control): script-inherent near-homographs disambiguated
  only by context/lexicon, NOT by any visual encoder — o/u (U+1823/U+1824) and
  oe/ue (U+1825/U+1826). Should stay flat across width_collapse modes; if it
  also "improves", that is generic capacity/regularization bleeding into the
  control, not evidence for the width-info hypothesis.
- **B-CLASS** (treatment): small-visual-difference minimal pairs (a tooth/stroke
  apart) that preserving conv-stem WIDTH info could plausibly disambiguate —
  t/d (U+1832/U+1833), k/g (U+182C/U+182D), a/e (U+1820/U+1821),
  i/y (U+1822/U+1836).

Alignment: a from-scratch ``O(len(a)*len(b))`` Levenshtein DP with stored
back-pointers (``mongocr.metrics.edit_distance`` is deliberately scalar-only —
no parent pointers — so per-op classification needs its own DP here). Same cost
model as ``edit_distance`` (ins=del=sub=1, cost 0 iff chars equal) so distances
agree by construction; ties between substitute and insert+delete break toward
substitution, since every A/B-CLASS pair is by construction a single-character
swap and should backtrace as ONE sub op, not a spurious del+ins pair.

Both pred and ref are folded with ``mongocr.metrics.nominal_fold`` before
alignment, matching how norm_CER is computed, so FVS/MVS/joiner/NNBSP noise
never contaminates the per-pair counts.

Usage::

    python3 -m scripts.eval_confusion \
        --eval-shards '/local/eval_cache/test-*.tar' \
        --alphabet alphabet.json --ckpt crnn_concat.pt \
        --val-threshold 434600 --test-threshold 435200 --split test
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from mongocr.alphabet import load as load_alphabet
from mongocr.data import build_pipeline, list_shards, src_doc_bands
from mongocr.metrics import nominal_fold, ocr_report
from mongocr.model import (CRNN, WIDTH_COLLAPSE_MODES, greedy_decode,
                           resolve_arch_from_ckpt)

# (ref_char, hyp_char) unordered pairs. Both directions are tracked separately
# in the directed table; these tuples just fix a canonical (lower codepoint
# first) ordering for the "undirected"/pooled tables.
A_CLASS_PAIRS: list[tuple[str, str]] = [
    ("ᠣ", "ᠤ"),  # MONGOLIAN LETTER O <-> MONGOLIAN LETTER U
    ("ᠥ", "ᠦ"),  # MONGOLIAN LETTER OE <-> MONGOLIAN LETTER UE
]
B_CLASS_PAIRS: list[tuple[str, str]] = [
    ("ᠲ", "ᠳ"),  # MONGOLIAN LETTER T <-> MONGOLIAN LETTER D
    ("ᠬ", "ᠭ"),  # MONGOLIAN LETTER K <-> MONGOLIAN LETTER G
    ("ᠠ", "ᠡ"),  # MONGOLIAN LETTER A <-> MONGOLIAN LETTER E
    ("ᠢ", "ᠶ"),  # MONGOLIAN LETTER I <-> MONGOLIAN LETTER YA (y)
]

_SUB, _DEL, _INS = "sub", "del", "ins"


def align_ops(ref: str, hyp: str) -> list[tuple[str, str, str]]:
    """Levenshtein-align ``ref`` -> ``hyp``, backtrace with ties broken toward
    substitution, return the op list as ``(kind, ref_char_or_"", hyp_char_or_"")``.

    ``kind`` is one of ``"sub"`` (ref_char != hyp_char), ``"match"`` (equal,
    included so the DP/backtrace is easy to verify against total ops == max(len),
    filtered out by callers that only want edits), ``"del"`` (ref_char, no hyp),
    ``"ins"`` (no ref, hyp_char). Same DP cost model as
    ``mongocr.metrics.edit_distance`` (ins=del=sub=1, cost 0 iff equal), so
    ``sum(1 for op in ops if op[0] != "match") == edit_distance(ref, hyp)``.
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            sub_cost = dp[i - 1][j - 1] + cost
            del_cost = dp[i - 1][j] + 1
            ins_cost = dp[i][j - 1] + 1
            dp[i][j] = min(sub_cost, del_cost, ins_cost)

    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                # tie-break toward substitution: prefer this diagonal move
                # whenever it is *a* valid predecessor, even if del/ins also
                # tie, so a single-character swap backtraces as one "sub" (or
                # one "match"), never a spurious del+ins pair.
                ops.append((_SUB if cost else "match", ref[i - 1], hyp[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append((_DEL, ref[i - 1], ""))
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append((_INS, "", hyp[j - 1]))
            j -= 1
            continue
        raise AssertionError("unreachable: DP backtrace ran out of valid moves")
    ops.reverse()
    return ops


def edit_distance_via_ops(ref: str, hyp: str) -> int:
    """Distance implied by ``align_ops`` — used only to unit-test agreement with
    ``mongocr.metrics.edit_distance`` (see tests/test_eval_confusion.py)."""
    return sum(1 for kind, _r, _h in align_ops(ref, hyp) if kind != "match")


class PairStats:
    """Accumulates directed substitution counts + per-char ref occurrence counts
    for a fixed set of (charA, charB) pairs, across many aligned (ref, hyp) lines."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs
        self.chars = {c for pair in pairs for c in pair}
        self.directed: Counter[tuple[str, str]] = Counter()  # (ref_char,hyp_char)->n
        self.ref_occurrences: Counter[str] = Counter()  # ref_char -> n seen in refs

    def update(self, ref: str, ops: list[tuple[str, str, str]]) -> None:
        for ch in ref:
            if ch in self.chars:
                self.ref_occurrences[ch] += 1
        for kind, r, h in ops:
            if kind == _SUB and r in self.chars and h in self.chars:
                self.directed[(r, h)] += 1

    def pair_row(self, a: str, b: str) -> dict:
        ab = self.directed.get((a, b), 0)
        ba = self.directed.get((b, a), 0)
        undirected = ab + ba
        denom = self.ref_occurrences.get(a, 0) + self.ref_occurrences.get(b, 0)
        rate = undirected / denom if denom else 0.0
        return {
            "pair": f"U+{ord(a):04X}<->U+{ord(b):04X}",
            "chars": f"{a}<->{b}",
            "ref_to_hyp_count": ab,
            "hyp_to_ref_count": ba,
            "undirected_count": undirected,
            "ref_occurrences": denom,
            "rate": rate,
        }

    def pooled_rate(self) -> tuple[int, int, float]:
        """(pooled substitution count, pooled ref-occurrence denominator, rate)
        summed across ALL pairs in this set — the primary discriminating stat
        given per-pair sparsity."""
        total_subs = sum(self.pair_row(a, b)["undirected_count"] for a, b in self.pairs)
        total_denom = sum(self.pair_row(a, b)["ref_occurrences"] for a, b in self.pairs)
        rate = total_subs / total_denom if total_denom else 0.0
        return total_subs, total_denom, rate


def score_confusion(preds: list[str], refs: list[str]) -> dict:
    """Fold pred/ref to nominal Unicode, align, accumulate A/B-CLASS pair stats."""
    a_stats = PairStats(A_CLASS_PAIRS)
    b_stats = PairStats(B_CLASS_PAIRS)
    for pred, ref in zip(preds, refs):
        nref = nominal_fold(ref)
        nhyp = nominal_fold(pred)
        ops = align_ops(nref, nhyp)
        a_stats.update(nref, ops)
        b_stats.update(nref, ops)
    return {"a_class": a_stats, "b_class": b_stats}


def print_pair_table(name: str, stats: PairStats) -> None:
    print(f"\n[eval_confusion] {name} per-pair table "
          f"(directed ref->hyp counts NOT summed; occurrence-normalized rate):",
          flush=True)
    header = (f"  {'pair':<22}{'ref->hyp':>10}{'hyp->ref':>10}"
              f"{'undirected':>12}{'ref_occ':>10}{'rate':>10}")
    print(header, flush=True)
    for a, b in stats.pairs:
        row = stats.pair_row(a, b)
        print(f"  {row['pair']:<22}{row['ref_to_hyp_count']:>10}"
              f"{row['hyp_to_ref_count']:>10}{row['undirected_count']:>12}"
              f"{row['ref_occurrences']:>10}{row['rate']:>10.4%}", flush=True)
    subs, denom, rate = stats.pooled_rate()
    print(f"  {'POOLED':<22}{'':>10}{'':>10}{subs:>12}{denom:>10}{rate:>10.4%}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--eval-shards", action="append", required=True,
                    help="dedicated eval shards (extract_eval_shards output), "
                         "e.g. test-*.tar for the headline split")
    ap.add_argument("--alphabet", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-threshold", type=int, required=True)
    ap.add_argument("--test-threshold", type=int, required=True)
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--split", choices=["val", "test"], default="test",
                    help="which held-out split the --eval-shards contain "
                         "(default: test = headline)")
    ap.add_argument("--img-h", type=int, default=1024)
    ap.add_argument("--img-w", type=int, default=64)
    ap.add_argument("--lstm-hidden", type=int, default=384,
                    help="fallback for checkpoints saved before --lstm-hidden "
                         "was recorded in the checkpoint's args")
    ap.add_argument("--width-collapse", default=None, choices=[*WIDTH_COLLAPSE_MODES, None],
                    help="fallback for checkpoints saved before --width-collapse "
                         "was recorded in the checkpoint's args (default: mean)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap eval batches (default: None = full eval-shards set)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--show", type=int, default=15,
                    help="index-0 + this many strided samples printed FIRST as a "
                         "spot-check, before trusting the pooled counts")
    args = ap.parse_args()

    device = (torch.device("cuda") if (args.device in ("auto", "cuda") and torch.cuda.is_available())
              else torch.device("cpu"))
    alpha = load_alphabet(Path(args.alphabet))
    shard_urls = list_shards(*args.eval_shards)
    if not shard_urls:
        raise SystemExit(f"no eval shards matched: {args.eval_shards}")
    _is_train, is_val, is_test = src_doc_bands(args.val_threshold, args.test_threshold,
                                               args.gap)
    is_eval = is_test if args.split == "test" else is_val
    print(f"[eval_confusion] scoring {args.split.upper()} split, "
          f"{len(shard_urls)} eval shards", flush=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    # Reconstruct the checkpoint's actual architecture — see
    # mongocr.model.resolve_arch_from_ckpt for the legacy-checkpoint fallback
    # reasoning. A width_collapse mismatch would either crash load_state_dict
    # (LSTM input shape differs) or, for mean vs mean_big, load successfully
    # with the WRONG param count silently — this script prints params below
    # specifically so that silent mismatch is always visible.
    lstm_hidden, width_collapse, img_w = resolve_arch_from_ckpt(
        ckpt, cli_lstm_hidden=args.lstm_hidden,
        cli_width_collapse=args.width_collapse, cli_img_w=args.img_w)
    print(f"[eval_confusion] arch: lstm_hidden={lstm_hidden} "
          f"width_collapse={width_collapse} img_w={img_w} (from checkpoint args: "
          f"{'yes' if ckpt.get('args') else 'no, using CLI/defaults'})", flush=True)

    model = CRNN(alpha.n_classes, lstm_hidden=lstm_hidden,
                 width_collapse=width_collapse, img_w=img_w).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eval_confusion] params={n_params/1e6:.3f}M ({n_params:,})", flush=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if ckpt.get("alphabet") and ckpt["alphabet"] != alpha.chars:
        raise SystemExit("checkpoint alphabet != provided alphabet.json")

    import webdataset as wds
    pipe = build_pipeline(shard_urls, alpha, img_h=args.img_h, img_w=img_w,
                          batch_size=args.batch_size, keep_src_doc=is_eval, training=False)
    loader = wds.WebLoader(pipe, batch_size=None, num_workers=args.num_workers)

    preds, refs = [], []
    with torch.no_grad():
        for bi, (imgs, padded, lengths) in enumerate(loader):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            log_probs = model(imgs.to(device))
            preds.extend(greedy_decode(log_probs.float().cpu(), alpha.blank, alpha.chars))
            for t, n in zip(padded, lengths):
                refs.append("".join(alpha.chars[i] for i in t[: int(n)].tolist()))

    if not preds:
        raise SystemExit("no samples decoded — check --eval-shards / split thresholds")

    # Spot-check FIRST (index-0 + a deterministic strided sample), before trusting
    # the pooled rates: a silently-broken forward pass (e.g. a garbled T-ordering
    # from a reshape bug) could otherwise masquerade as a CER delta.
    print("\n[eval_confusion] index-0 + samples (decoded text vs GT) — SPOT-CHECK "
          "FIRST:", flush=True)
    step = max(1, len(preds) // max(1, args.show))
    for i in range(0, len(preds), step):
        print(f"  [{i}] GT : {refs[i][:70]}", flush=True)
        print(f"       OUT: {preds[i][:70]}", flush=True)

    rep = ocr_report(preds, refs)
    print(f"\n[eval_confusion] HEADLINE n={rep.n} norm_CER={rep.norm_cer:.4f} "
          f"raw_CER={rep.raw_cer:.4f} WER={rep.wer:.4f} "
          f"line_exact={rep.line_exact:.4f}", flush=True)
    print("[eval_confusion] a variant that improves per-pair rates but regresses "
          "this headline norm_CER is not a clean win.", flush=True)

    conf = score_confusion(preds, refs)
    print_pair_table("A-CLASS (negative control — should stay ~flat)", conf["a_class"])
    print_pair_table("B-CLASS (treatment — width info could help)", conf["b_class"])

    a_subs, a_denom, a_rate = conf["a_class"].pooled_rate()
    b_subs, b_denom, b_rate = conf["b_class"].pooled_rate()
    print(f"\n[eval_confusion] SUMMARY  pooled_A_rate={a_rate:.4%} "
          f"(n_sub={a_subs}/{a_denom})  pooled_B_rate={b_rate:.4%} "
          f"(n_sub={b_subs}/{b_denom})  norm_CER={rep.norm_cer:.4%}", flush=True)
    print("[eval_confusion] cold numbers only — no cross-variant verdict here; "
          "compare this line's pooled_B_rate/pooled_A_rate/norm_CER across all 4 "
          "checkpoints' runs per the ablation's success criteria (treat per-pair "
          "deltas smaller than roughly sqrt(count) as noise).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
