# -*- coding: utf-8 -*-

"""Pre-training data verification (QA gate, run before any scaled run).

Checks, in order:
1. index-0 / deterministic-sample decode: pull the first few train samples,
   print their label text + image shape so a human can eyeball image<->GT.
2. all expected fonts appear in a sample window.
3. T >= U: encoder frames must exceed the longest target in the window.
4. NO sample duplication across DataLoader workers (the silent webdataset trap):
   collect keys with num_workers>=2 and assert set size == list size.
5. train/eval src_doc disjointness: the two splits share zero src_doc.

Usage::

    python3 -m scripts.verify_dataloader \
        --shards '/path/to/shards/shard-*.tar' \
        --alphabet alphabet.json --eval-threshold 430000 --gap 200
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from mongocr.alphabet import load as load_alphabet
from mongocr.data import (build_pipeline, key_pipeline, keys_of, list_shards,
                          src_doc_bands)
from mongocr.model import CRNN


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--shards", action="append", required=True)
    ap.add_argument("--alphabet", required=True)
    ap.add_argument("--val-threshold", type=int, required=True)
    ap.add_argument("--test-threshold", type=int, required=True)
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--img-h", type=int, default=1024)
    ap.add_argument("--img-w", type=int, default=64)
    ap.add_argument("--lstm-hidden", type=int, default=384)
    ap.add_argument("--window", type=int, default=2000, help="samples to inspect")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    alpha = load_alphabet(Path(args.alphabet))
    shard_urls = list_shards(*args.shards)
    if not shard_urls:
        raise SystemExit(f"no shards matched: {args.shards}")
    print(f"[verify] {len(shard_urls)} shards, alphabet={len(alpha.chars)} "
          f"sha256={alpha.sha256[:12]}", flush=True)
    is_train, is_val, is_test = src_doc_bands(args.val_threshold, args.test_threshold,
                                              args.gap)

    # 1. index-0 / sample decode preview --------------------------------------
    import webdataset as wds
    pipe = build_pipeline(shard_urls, alpha, img_h=args.img_h, img_w=args.img_w,
                          batch_size=1, keep_src_doc=is_train, training=False)
    print("\n[verify] first samples (eyeball image shape <-> label):", flush=True)
    fonts = Counter()
    u_max = 0
    seen = 0
    for imgs, padded, lengths in pipe:
        if seen < 5:
            ids = padded[0][: int(lengths[0])].tolist()
            text = "".join(alpha.chars[i] for i in ids)
            print(f"  [{seen}] img={tuple(imgs.shape)} U={int(lengths[0])} "
                  f"text={text[:60]!r}", flush=True)
        u_max = max(u_max, int(lengths.max()))
        seen += 1
        if seen >= args.window:
            break

    # 2. font coverage + OOV-drop rate -----------------------------------------
    fonts = Counter()
    oov = scanned = 0
    for _k, _sd, txt, font in keys_of(shard_urls, is_train, limit=args.window):
        fonts[font] += 1
        scanned += 1
        if not alpha.covers(txt):
            oov += 1
    print(f"\n[verify] fonts in first {scanned} train samples: {dict(fonts)}",
          flush=True)
    print(f"[verify] OOV lines (would be dropped): {oov}/{scanned} "
          f"({100*oov/max(scanned,1):.3f}%)", flush=True)

    # 3. T >= U ----------------------------------------------------------------
    model = CRNN(alpha.n_classes, lstm_hidden=args.lstm_hidden)
    with torch.no_grad():
        t_frames = model(torch.zeros(1, 1, args.img_h, args.img_w)).shape[1]
    ok_tu = t_frames >= u_max
    print(f"\n[verify] T(frames)={t_frames} U_max(window)={u_max} "
          f"-> {'OK T>=U' if ok_tu else 'FAIL T<U'}", flush=True)

    # 4. no duplication ACROSS WORKERS (run the key pipeline through a real
    #    multi-worker WebLoader so split_by_worker is actually exercised) -------
    print(f"\n[verify] multi-worker duplication check (num_workers={args.num_workers}) ...",
          flush=True)
    kloader = wds.WebLoader(key_pipeline(shard_urls, is_train),
                            batch_size=None, num_workers=args.num_workers)
    keys = []
    for k in kloader:
        keys.append(k)
        if len(keys) >= args.window:
            break
    dup = len(keys) - len(set(keys))
    print(f"[verify] collected {len(keys)} keys across {args.num_workers} workers, "
          f"duplicates={dup} -> {'OK no dup' if dup == 0 else 'FAIL duplicates'}",
          flush=True)

    # 5. train/val/test src_doc disjointness -----------------------------------
    docs = {}
    for name, pred in (("train", is_train), ("val", is_val), ("test", is_test)):
        s = {sd for _k, sd, _t, _f in keys_of(shard_urls, pred, limit=args.window)}
        docs[name] = s
        print(f"\n[verify] {name:5s} src_doc: n={len(s)} "
              f"[min={min(s, default=-1)},max={max(s, default=-1)}]", flush=True)
    leaks = {f"{a}&{b}": len(docs[a] & docs[b])
             for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    no_leak = all(v == 0 for v in leaks.values())
    print(f"[verify] pairwise intersections {leaks} -> "
          f"{'OK disjoint' if no_leak else 'FAIL leak'}", flush=True)

    ok = ok_tu and dup == 0 and no_leak
    print(f"\n[verify] {'ALL PASS' if ok else 'FAILURES PRESENT'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
