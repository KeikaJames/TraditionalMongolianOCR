# -*- coding: utf-8 -*-

"""Evaluate a CRNN checkpoint on the held-out (src_doc) split.

Reports normalized + raw CER, WER, exact-line rate, and prints index-0 plus a
deterministic sample of pred-vs-GT pairs (trust decoded text, not any proxy).

Usage::

    python3 -m scripts.eval_crnn \
        --shards '/path/to/shards/shard-*.tar' \
        --alphabet alphabet.json --ckpt crnn.pt \
        --eval-threshold 430000 --gap 200 --max-batches 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mongocr.alphabet import load as load_alphabet
from mongocr.data import build_pipeline, list_shards, src_doc_split
from mongocr.metrics import ocr_report
from mongocr.model import CRNN, greedy_decode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--shards", action="append", required=True)
    ap.add_argument("--alphabet", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-threshold", type=int, required=True)
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--img-h", type=int, default=1024)
    ap.add_argument("--img-w", type=int, default=64)
    ap.add_argument("--lstm-hidden", type=int, default=384)
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
    _, is_eval = src_doc_split(args.eval_threshold, args.gap)

    model = CRNN(alpha.n_classes, lstm_hidden=args.lstm_hidden).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    if ckpt.get("alphabet") and ckpt["alphabet"] != alpha.chars:
        raise SystemExit("checkpoint alphabet != provided alphabet.json")

    import webdataset as wds
    pipe = build_pipeline(shard_urls, alpha, img_h=args.img_h, img_w=args.img_w,
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
