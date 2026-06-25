# -*- coding: utf-8 -*-

"""Extract the val/test bands into small dedicated shards (one-time).

The val/test src_doc bands live only in the highest-src_doc (last) source
shards. Filtering them out of the full shard list at eval time forces a stream
over the whole corpus to reach them (catastrophic over a network mount). Instead
we copy the val and test samples ONCE into their own small shards (kept on fast
local disk) so every evaluation reads only eval data — no filtering, no
full-corpus scan. The split is unchanged; these are the same samples the
src_doc bands define, just physically separated for fast access.

Usage::

    python3 -m scripts.extract_eval_shards \
        --src '/path/to/shards/shard-*.tar' --tail 40 \
        --out-dir /local/eval_cache \
        --val-threshold 434600 --test-threshold 435200 --gap 200
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

from mongocr.data import list_shards, src_doc_bands


class RollingTar:
    """Write (png, json) samples into rolling tar shards of ``per_shard`` items."""

    def __init__(self, out_dir: Path, prefix: str, per_shard: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.per_shard = per_shard
        self.idx = 0
        self.n = 0
        self.total = 0
        self.tar: tarfile.TarFile | None = None

    def _open(self):
        path = self.out_dir / f"{self.prefix}-{self.idx:04d}.tar"
        self.tar = tarfile.open(path, "w")
        self.n = 0

    def add(self, key: str, png: bytes, meta: dict):
        if self.tar is None:
            self._open()
        for ext, data in (("png", png),
                          ("json", json.dumps(meta, ensure_ascii=False).encode())):
            info = tarfile.TarInfo(name=f"{key}.{ext}")
            info.size = len(data)
            self.tar.addfile(info, io.BytesIO(data))
        self.n += 1
        self.total += 1
        if self.n >= self.per_shard:
            self.tar.close()
            self.idx += 1
            self.tar = None

    def close(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--src", action="append", required=True, help="source shard glob")
    ap.add_argument("--tail", type=int, default=40,
                    help="scan only the last N source shards (where val/test live)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alphabet", required=True,
                    help="alphabet.json; OOV lines are dropped here too so the "
                         "cached eval set equals the population eval actually scores")
    ap.add_argument("--val-threshold", type=int, required=True)
    ap.add_argument("--test-threshold", type=int, required=True)
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--per-shard", type=int, default=50000)
    args = ap.parse_args()

    import webdataset as wds
    from mongocr.alphabet import load as load_alphabet

    alpha = load_alphabet(Path(args.alphabet))
    shards = list_shards(*args.src)[-args.tail:]
    if not shards:
        raise SystemExit("no source shards matched")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] scanning {len(shards)} tail shards -> {out_dir} "
          f"(alphabet {len(alpha.chars)} chars, OOV lines dropped)", flush=True)

    _is_train, is_val, is_test = src_doc_bands(args.val_threshold, args.test_threshold, args.gap)
    val_w = RollingTar(out_dir, "val", args.per_shard)
    test_w = RollingTar(out_dir, "test", args.per_shard)

    pipe = wds.DataPipeline(
        wds.SimpleShardList(shards),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
    )
    seen = 0
    window_min_sd = None                 # lowest src_doc anywhere in the scanned window
    vmin = vmax = tmin = tmax = None
    vdocs, tdocs = set(), set()
    for s in pipe:
        if "json" not in s or "png" not in s:
            continue
        try:
            meta = json.loads(s["json"])
            sd = int(meta["src_doc"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        window_min_sd = sd if window_min_sd is None else min(window_min_sd, sd)
        # same OOV-drop policy as the eval loader, so cached == scored population
        if not alpha.covers(meta["text"]):
            continue
        if is_val(sd):
            val_w.add(s["__key__"], s["png"], meta)
            vmin = sd if vmin is None else min(vmin, sd); vmax = sd if vmax is None else max(vmax, sd)
            vdocs.add(sd)
        elif is_test(sd):
            test_w.add(s["__key__"], s["png"], meta)
            tmin = sd if tmin is None else min(tmin, sd); tmax = sd if tmax is None else max(tmax, sd)
            tdocs.add(sd)
        seen += 1
        if seen % 200000 == 0:
            print(f"[extract] scanned {seen:,}; val={val_w.total:,} test={test_w.total:,}",
                  flush=True)

    val_w.close()
    test_w.close()
    print(f"[extract] DONE: val={val_w.total:,} samples / {len(vdocs)} docs "
          f"[{vmin},{vmax}] ({val_w.idx + 1} shards); "
          f"test={test_w.total:,} samples / {len(tdocs)} docs [{tmin},{tmax}] "
          f"({test_w.idx + 1} shards); window_min_src_doc={window_min_sd}", flush=True)

    # Coverage guard: the scan window must open BELOW the val floor, which proves
    # the contiguous high-src_doc tail spans the entire val+test range (nothing
    # below the floor was left in un-scanned earlier shards). Without this a too-
    # small --tail silently truncates the eval set and the early-stop CER is wrong.
    if val_w.total == 0 or test_w.total == 0:
        raise SystemExit(f"[extract] FAIL: empty band (val={val_w.total} test={test_w.total}); "
                         f"check thresholds / increase --tail")
    if window_min_sd is None or window_min_sd >= args.val_threshold:
        raise SystemExit(
            f"[extract] FAIL: window_min_src_doc={window_min_sd} did not open below "
            f"val_threshold={args.val_threshold} — tail too small, val/test may be "
            f"truncated. Increase --tail and rerun.")
    print("[extract] coverage guard PASS (window opened below val floor)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
