# -*- coding: utf-8 -*-

"""Streaming CRNN+CTC trainer over WebDataset shards.

Streams hundreds of millions of line strips from sharded tars (no meta in RAM,
no per-image stat), splits train/eval by ``src_doc``, freezes the alphabet,
trains the from-scratch CRNN with the verified cold-start fixes, and reports
held-out normalized CER on a fixed eval subset frequently + the full eval set
sparsely. Coverage target is >= 1 epoch over all samples; stops on max-steps or
held-out CER plateau (early stop), whichever first.

Usage (from repo root)::

    python3 -m scripts.train_crnn \
        --shards '/path/to/image2text/shards/shard-*.tar' \
        --shards '/path/to/image2text/hanshi_shards/shard-*.tar' \
        --alphabet alphabet.json \
        --eval-threshold 430000 --gap 200 \
        --batch-size 64 --device cuda --save crnn.pt
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from mongocr.alphabet import load as load_alphabet
from mongocr.data import build_pipeline, list_shards, src_doc_split
from mongocr.losses import ctc_loss
from mongocr.metrics import ocr_report
from mongocr.model import CRNN, greedy_decode


def pick_device(flag: str) -> torch.device:
    if flag == "cpu":
        return torch.device("cpu")
    if flag in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    if flag in ("cuda", "auto"):
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as e:
            print(f"[crnn] cuda init failed -> CPU: {type(e).__name__}: {e}", flush=True)
    return torch.device("cpu")


def make_loader(shard_urls, alpha, keep, *, img_h, img_w, batch_size, num_workers,
                training):
    import webdataset as wds
    pipe = build_pipeline(
        shard_urls, alpha, img_h=img_h, img_w=img_w, batch_size=batch_size,
        keep_src_doc=keep, training=training,
    )
    return wds.WebLoader(pipe, batch_size=None, num_workers=num_workers)


@torch.no_grad()
def evaluate(model, loader, blank, alphabet, device, max_batches=None):
    model.eval()
    preds, refs = [], []
    for bi, (imgs, padded, lengths) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        log_probs = model(imgs.to(device))
        preds.extend(greedy_decode(log_probs.float().cpu(), blank, alphabet))
        for t, n in zip(padded, lengths):
            ids = t[: int(n)].tolist()
            refs.append("".join(alphabet[i] for i in ids))
    model.train()
    if not preds:
        return None
    return ocr_report(preds, refs).norm_cer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--shards", action="append", required=True,
                    help="glob for shard tars (repeatable)")
    ap.add_argument("--alphabet", required=True, help="alphabet.json from build_alphabet")
    ap.add_argument("--eval-threshold", type=int, required=True,
                    help="src_doc >= this -> eval; < this-gap -> train")
    ap.add_argument("--gap", type=int, default=200, help="src_doc gap band dropped")
    ap.add_argument("--steps", type=int, default=2_000_000, help="max optimizer steps")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-subset-batches", type=int, default=120,
                    help="batches for the frequent fixed eval")
    ap.add_argument("--full-eval-every", type=int, default=50000)
    ap.add_argument("--early-stop-patience", type=int, default=10,
                    help="stop after N evals with no norm_CER improvement")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--img-h", type=int, default=1024)
    ap.add_argument("--img-w", type=int, default=64)
    ap.add_argument("--lstm-hidden", type=int, default=384)
    ap.add_argument("--blank-bias", type=float, default=-3.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="", help="checkpoint path")
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--resume", default="", help="resume from checkpoint")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    alpha = load_alphabet(Path(args.alphabet))
    blank = alpha.blank
    print(f"[crnn] alphabet chars={len(alpha.chars)} blank={blank} "
          f"sha256={alpha.sha256[:12]}", flush=True)

    shard_urls = list_shards(*args.shards)
    if not shard_urls:
        raise SystemExit(f"no shards matched: {args.shards}")
    print(f"[crnn] {len(shard_urls)} shards", flush=True)

    is_train, is_eval = src_doc_split(args.eval_threshold, args.gap)
    train_loader = make_loader(shard_urls, alpha, is_train, img_h=args.img_h,
                               img_w=args.img_w, batch_size=args.batch_size,
                               num_workers=args.num_workers, training=True)

    def eval_loader():
        return make_loader(shard_urls, alpha, is_eval, img_h=args.img_h,
                           img_w=args.img_w, batch_size=args.batch_size,
                           num_workers=max(2, args.num_workers // 2), training=False)

    model = CRNN(alpha.n_classes, lstm_hidden=args.lstm_hidden,
                 blank_bias=args.blank_bias).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def lr_at(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        # cosine to 0 over the remaining budget
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    start_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_step = ckpt.get("step", 0)
        print(f"[crnn] resumed from {args.resume} @ step {start_step}", flush=True)

    with torch.no_grad():
        probe = torch.zeros(1, 1, args.img_h, args.img_w, device=device)
        t_frames = model(probe).shape[1]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[crnn] device={device} params={n_params/1e6:.2f}M T(frames)={t_frames} "
          f"img={args.img_h}x{args.img_w} batch={args.batch_size}", flush=True)

    best_cer = float("inf")
    no_improve = 0
    seen = start_step * args.batch_size
    t0 = time.time()
    step = start_step
    model.train()
    data_iter = iter(train_loader)
    while step < args.steps:
        try:
            imgs, padded, lengths = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            imgs, padded, lengths = next(data_iter)
        imgs = imgs.to(device)
        log_probs = model(imgs).log_softmax(dim=-1)
        loss = ctc_loss(log_probs, padded.to(device), lengths, blank=blank)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        step += 1
        seen += args.batch_size

        if step % 50 == 0:
            rate = (step - start_step) / (time.time() - t0)
            print(f"[crnn] step {step:7d} ctc={loss.item():.3f} "
                  f"lr={sched.get_last_lr()[0]:.2e} seen={seen:,} "
                  f"({rate:.1f} it/s)", flush=True)

        if step % args.eval_every == 0:
            full = step % args.full_eval_every == 0
            cer = evaluate(model, eval_loader(), blank, alpha.chars, device,
                           max_batches=None if full else args.eval_subset_batches)
            tag = "FULL" if full else f"sub({args.eval_subset_batches}b)"
            print(f"[crnn] step {step:7d} held-out norm_CER[{tag}] = "
                  f"{cer:.4f}  (best {best_cer:.4f})", flush=True)
            if cer is not None and cer < best_cer - 1e-4:
                best_cer = cer
                no_improve = 0
                if args.save:
                    torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(),
                                "alphabet": alpha.chars, "args": vars(args), "step": step,
                                "norm_cer": cer}, args.save)
            else:
                no_improve += 1
                if no_improve >= args.early_stop_patience:
                    print(f"[crnn] early stop: {no_improve} evals w/o improvement "
                          f"(best norm_CER={best_cer:.4f} @ seen={seen:,})", flush=True)
                    break

        if args.save and step % args.save_every == 0:
            torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(),
                        "alphabet": alpha.chars, "args": vars(args), "step": step},
                       args.save + ".last")

    print(f"[crnn] done: step={step} seen={seen:,} best_norm_CER={best_cer:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
