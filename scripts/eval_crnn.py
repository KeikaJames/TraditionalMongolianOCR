# -*- coding: utf-8 -*-

"""Evaluate a CRNN checkpoint on the held-out (src_doc) split.

Reports normalized + raw CER, WER, exact-line rate, and prints index-0 plus a
deterministic sample of pred-vs-GT pairs (trust decoded text, not any proxy).

Usage::

    python3 -m scripts.eval_crnn \
        --shards '/path/to/shards/shard-*.tar' \
        --alphabet alphabet.json --ckpt crnn.pt \
        --val-threshold 434600 --test-threshold 435200 --split test --max-batches 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mongocr.alphabet import load as load_alphabet
from mongocr.data import build_pipeline, list_shards, src_doc_bands
from mongocr.metrics import ocr_report
from mongocr.model import (CRNN, WIDTH_COLLAPSE_MODES, greedy_decode,
                           resolve_arch_from_ckpt)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--shards", action="append", required=True)
    ap.add_argument("--alphabet", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-threshold", type=int, required=True)
    ap.add_argument("--test-threshold", type=int, required=True)
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--split", choices=["val", "test"], default="test",
                    help="which held-out split to score (default: test = headline)")
    ap.add_argument("--img-h", type=int, default=1024)
    ap.add_argument("--img-w", type=int, default=64)
    ap.add_argument("--lstm-hidden", type=int, default=384,
                    help="overridden by the checkpoint's saved --lstm-hidden when "
                         "present (older checkpoints predating this arg use this "
                         "flag's value)")
    ap.add_argument("--width-collapse", default=None, choices=[*WIDTH_COLLAPSE_MODES, None],
                    help="overridden by the checkpoint's saved --width-collapse "
                         "when present; only needed as a fallback for checkpoints "
                         "saved before this flag existed (default: mean)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=200)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--show", type=int, default=15)
    args = ap.parse_args()

    device = (torch.device("cuda") if (args.device in ("auto", "cuda") and torch.cuda.is_available())
              else torch.device("cpu"))
    alpha = load_alphabet(Path(args.alphabet))
    shard_urls = list_shards(*args.shards)
    _is_train, is_val, is_test = src_doc_bands(args.val_threshold, args.test_threshold,
                                               args.gap)
    is_eval = is_test if args.split == "test" else is_val
    print(f"[eval] scoring {args.split.upper()} split", flush=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    # Reconstruct the exact architecture the checkpoint was trained with —
    # see mongocr.model.resolve_arch_from_ckpt for the legacy-checkpoint
    # fallback reasoning (e.g. the production crnn_x2.pt, saved before
    # --width-collapse/--lstm-hidden/--img-w existed as flags).
    lstm_hidden, width_collapse, img_w = resolve_arch_from_ckpt(
        ckpt, cli_lstm_hidden=args.lstm_hidden,
        cli_width_collapse=args.width_collapse, cli_img_w=args.img_w)
    print(f"[eval] arch: lstm_hidden={lstm_hidden} width_collapse={width_collapse} "
          f"img_w={img_w} (from checkpoint args: "
          f"{'yes' if ckpt.get('args') else 'no, using CLI/defaults'})", flush=True)

    model = CRNN(alpha.n_classes, lstm_hidden=lstm_hidden,
                 width_collapse=width_collapse, img_w=img_w).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eval] params={n_params/1e6:.2f}M ({n_params:,})", flush=True)
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
            if bi >= args.max_batches:
                break
            log_probs = model(imgs.to(device))
            preds.extend(greedy_decode(log_probs.float().cpu(), alpha.blank, alpha.chars))
            for t, n in zip(padded, lengths):
                refs.append("".join(alpha.chars[i] for i in t[: int(n)].tolist()))

    rep = ocr_report(preds, refs)
    print(f"\n[eval] n={rep.n} norm_CER={rep.norm_cer:.4f} raw_CER={rep.raw_cer:.4f} "
          f"WER={rep.wer:.4f} line_exact={rep.line_exact:.4f}", flush=True)
    print("\n[eval] index-0 + samples (decoded text vs GT):", flush=True)
    step = max(1, len(preds) // max(1, args.show))
    for i in range(0, len(preds), step):
        print(f"  [{i}] GT : {refs[i][:70]}", flush=True)
        print(f"       OUT: {preds[i][:70]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
