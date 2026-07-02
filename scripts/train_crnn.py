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
        --eval-shards '/local/eval_cache/val-*.tar' \
        --alphabet alphabet.json \
        --val-threshold 434600 --test-threshold 435200 --gap 200 \
        --batch-size 128 --device cuda --save crnn.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from mongocr.alphabet import load as load_alphabet
from mongocr.data import build_pipeline, list_shards, src_doc_bands
from mongocr.losses import ctc_loss
from mongocr.metrics import ocr_report
from mongocr.model import (CRNN, WIDTH_COLLAPSE_MODES, greedy_decode,
                           matched_mean_big_hidden)


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
                training, data_seed=None):
    import webdataset as wds
    pipe = build_pipeline(
        shard_urls, alpha, img_h=img_h, img_w=img_w, batch_size=batch_size,
        keep_src_doc=keep, training=training, data_seed=data_seed,
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


def checkpoint_dict(model, opt, plateau, alpha, args, step, best_cer, no_improve,
                    norm_cer=None):
    """Shared checkpoint payload for the best/.last/.final saves — keeps the
    three call sites (best-on-val, periodic .last, guaranteed-final) from
    drifting out of sync on which keys they carry."""
    d = {"state_dict": model.state_dict(), "opt": opt.state_dict(),
         "plateau": plateau.state_dict(), "alphabet": alpha.chars,
         "args": vars(args), "step": step, "best_cer": best_cer,
         "no_improve": no_improve}
    if norm_cer is not None:
        d["norm_cer"] = norm_cer
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--shards", action="append", required=True,
                    help="glob for shard tars (repeatable)")
    ap.add_argument("--alphabet", required=True, help="alphabet.json from build_alphabet")
    ap.add_argument("--val-threshold", type=int, required=True,
                    help="val = val_threshold <= src_doc < test_threshold-gap")
    ap.add_argument("--test-threshold", type=int, required=True,
                    help="test = src_doc >= test_threshold (reserved, untouched here)")
    ap.add_argument("--gap", type=int, default=200, help="src_doc gap band dropped")
    ap.add_argument("--eval-shards", action="append", default=None,
                    help="dedicated val shards (from extract_eval_shards) read for "
                         "evaluation; STRONGLY recommended — without it eval filters "
                         "the full corpus and streams every shard to find the val band")
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
    ap.add_argument("--lr-patience", type=int, default=4,
                    help="halve LR after this many subset-evals with no val improvement")
    ap.add_argument("--lr-decay", type=float, default=0.5)
    ap.add_argument("--min-lr-divisor", type=float, default=64.0,
                    help="floor LR at base_lr / this; stop once floored and stalled")
    ap.add_argument("--img-h", type=int, default=1024)
    ap.add_argument("--img-w", type=int, default=64)
    ap.add_argument("--lstm-hidden", type=int, default=384)
    ap.add_argument("--width-collapse", default="mean", choices=WIDTH_COLLAPSE_MODES,
                    help="how the conv stem's residual width Wp is folded into "
                         "the BiLSTM input (default: mean = current behavior). "
                         "See mongocr/model.py CRNN docstring for the 4 modes.")
    ap.add_argument("--blank-bias", type=float, default=-3.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=None,
                    help="seed for the WebDataset shard-order + sample-buffer "
                         "shuffles (wds.detshuffle). Default: None = unseeded "
                         "wds.shuffle, today's wall-clock-jittered behavior, "
                         "unchanged. Set this (e.g. equal to --seed) for a "
                         "reproducible data order across independently launched "
                         "runs — required for a fair multi-variant ablation, "
                         "since each run's shuffle RNG is otherwise seeded from "
                         "PID+wall-clock and NOT from --seed/torch.manual_seed.")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="disable the LR-floored-and-plateaued early-stop break "
                         "so training always runs to --steps. best_cer tracking, "
                         "checkpoint saving, and the ReduceLROnPlateau LR decay "
                         "itself are unaffected — this only skips the `break`. "
                         "Use for a fixed-budget ablation across variants so a "
                         "wider model that plateaus later cannot stop earlier "
                         "than a narrower one under the same --early-stop-patience.")
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

    is_train, is_val, _is_test = src_doc_bands(
        args.val_threshold, args.test_threshold, args.gap)
    print(f"[crnn] split: train src_doc<{args.val_threshold - args.gap} | "
          f"val [{args.val_threshold},{args.test_threshold - args.gap}) | "
          f"test src_doc>={args.test_threshold} (reserved)", flush=True)
    train_loader = make_loader(shard_urls, alpha, is_train, img_h=args.img_h,
                               img_w=args.img_w, batch_size=args.batch_size,
                               num_workers=args.num_workers, training=True,
                               data_seed=args.data_seed)

    # Eval reads dedicated val shards if given (fast: only val data, no filtering
    # over the full corpus). Falls back to filtering the main shards otherwise.
    if args.eval_shards:
        eval_shard_urls = list_shards(*args.eval_shards)
        if not eval_shard_urls:
            raise SystemExit(f"no eval shards matched: {args.eval_shards}")
        print(f"[crnn] eval reads {len(eval_shard_urls)} dedicated val shards", flush=True)
    else:
        eval_shard_urls = shard_urls
        print("[crnn] WARNING: no --eval-shards; eval will filter the full corpus "
              "(slow). Use extract_eval_shards.", flush=True)

    def eval_loader():
        return make_loader(eval_shard_urls, alpha, is_val, img_h=args.img_h,
                           img_w=args.img_w, batch_size=args.batch_size,
                           num_workers=max(2, args.num_workers // 2), training=False)

    model = CRNN(alpha.n_classes, lstm_hidden=args.lstm_hidden,
                 blank_bias=args.blank_bias, width_collapse=args.width_collapse,
                 img_w=args.img_w).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # LR: linear warmup, then hold; halve on val-CER plateau (ReduceLROnPlateau),
    # floored at base_lr/min_lr_divisor. Early-stop only once LR is floored AND
    # still not improving — so the schedule anneals into the plateau wherever it
    # lands (unknown horizon), instead of a cosine stretched over a step cap the
    # run never reaches.
    base_lr = args.lr
    min_lr = base_lr / args.min_lr_divisor
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=args.lr_decay, patience=args.lr_patience, min_lr=min_lr)

    start_step = 0
    best_cer = float("inf")
    no_improve = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if "plateau" in ckpt:
            plateau.load_state_dict(ckpt["plateau"])
        start_step = ckpt.get("step", 0)
        # Restore early-stop accounting so a resume never overwrites the saved
        # best checkpoint with a worse model nor resets the patience counter.
        best_cer = ckpt.get("best_cer", ckpt.get("norm_cer", float("inf")))
        no_improve = ckpt.get("no_improve", 0)
        print(f"[crnn] resumed from {args.resume} @ step {start_step} "
              f"(best_cer={best_cer:.4f}, no_improve={no_improve})", flush=True)

    with torch.no_grad():
        probe = torch.zeros(1, 1, args.img_h, args.img_w, device=device)
        t_frames = model(probe).shape[1]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[crnn] device={device} params={n_params/1e6:.2f}M T(frames)={t_frames} "
          f"img={args.img_h}x{args.img_w} batch={args.batch_size} "
          f"width_collapse={args.width_collapse} lstm_hidden={args.lstm_hidden}",
          flush=True)
    # Cheap extra guard beyond CRNN's own mean_big ValueError: pin down the
    # launched variant's param count against a second, independent
    # computation (matched_mean_big_hidden), so a future refactor that
    # changes per-mode parameter shapes without updating the guard trips here
    # instead of silently shipping an unmatched ablation arm.
    if args.width_collapse == "mean_big":
        expected_hidden = matched_mean_big_hidden(alpha.n_classes, args.img_w)
        assert args.lstm_hidden == expected_hidden, (
            f"mean_big launched with lstm_hidden={args.lstm_hidden}, expected "
            f"{expected_hidden} (CRNN.__init__ should have already raised — "
            f"this assert is a second, independent check)")

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
        # linear warmup, then LR is governed by ReduceLROnPlateau (below)
        if step < args.warmup:
            for g in opt.param_groups:
                g["lr"] = base_lr * (step + 1) / args.warmup
        imgs = imgs.to(device)
        log_probs = model(imgs).log_softmax(dim=-1)
        loss = ctc_loss(log_probs, padded.to(device), lengths, blank=blank)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        step += 1
        seen += args.batch_size

        if step % 50 == 0:
            rate = (step - start_step) / (time.time() - t0)
            print(f"[crnn] step {step:7d} ctc={loss.item():.3f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} seen={seen:,} "
                  f"({rate:.1f} it/s)", flush=True)

        if step % args.eval_every == 0:
            full = step % args.full_eval_every == 0
            cer = evaluate(model, eval_loader(), blank, alpha.chars, device,
                           max_batches=None if full else args.eval_subset_batches)
            if cer is None:
                print(f"[crnn] step {step:7d} val eval returned no samples — "
                      f"skipping (check val band populates these shards)", flush=True)
            else:
                tag = "FULL" if full else f"sub({args.eval_subset_batches}b)"
                print(f"[crnn] step {step:7d} val norm_CER[{tag}] = "
                      f"{cer:.4f}  (best {best_cer:.4f}) "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
            # Only the (deterministic, frequent) SUBSET eval drives LR decay,
            # best-checkpoint, and early-stop. FULL eval is a logged sanity number
            # on a different population — mixing it into the same counter is
            # apples-to-oranges.
            if cer is not None and not full and step >= args.warmup:
                plateau.step(cer)
                if cer < best_cer - 1e-4:
                    best_cer = cer
                    no_improve = 0
                    if args.save:
                        torch.save(checkpoint_dict(model, opt, plateau, alpha, args,
                                                   step, best_cer, no_improve,
                                                   norm_cer=cer), args.save)
                else:
                    no_improve += 1
                    cur_lr = opt.param_groups[0]["lr"]
                    if (not args.no_early_stop and cur_lr <= min_lr * 1.001
                            and no_improve >= args.early_stop_patience):
                        print(f"[crnn] early stop: LR floored ({cur_lr:.2e}) and "
                              f"{no_improve} evals w/o improvement "
                              f"(best norm_CER={best_cer:.4f} @ seen={seen:,})", flush=True)
                        break

        if args.save and step % args.save_every == 0:
            # resumable latest checkpoint — carries early-stop accounting so a
            # restart from .last continues without clobbering the best model.
            torch.save(checkpoint_dict(model, opt, plateau, alpha, args, step,
                                       best_cer, no_improve), args.save + ".last")

    if args.save:
        # Guaranteed terminal-step checkpoint: the loop above only saves on a
        # val-CER improvement (args.save) or every --save-every steps
        # (args.save + ".last"), so the EXACT step the loop exits at (whether
        # by hitting --steps or by the early-stop `break`) can fall between
        # both and never get torch.save'd. Without this, a --no-early-stop
        # --steps N run has no guarantee the step-N model itself is ever on
        # disk, which breaks an equal-step-budget comparison across ablation
        # variants (only a stale .last or a possibly-earlier "best" would be
        # available). Always write it, unconditional on step % save_every.
        torch.save(checkpoint_dict(model, opt, plateau, alpha, args, step,
                                   best_cer, no_improve), args.save + ".final")
        print(f"[crnn] wrote terminal checkpoint -> {args.save}.final "
              f"(step={step})", flush=True)

    print(f"[crnn] done: step={step} seen={seen:,} best_norm_CER={best_cer:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
