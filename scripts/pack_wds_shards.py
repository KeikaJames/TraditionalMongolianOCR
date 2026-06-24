# -*- coding: utf-8 -*-
"""Pack rendered OCR strips into WebDataset-style tar shards (parallel).

Each worker writes shards to LOCAL disk, then a background thread moves completed
shards to the output dir. This lets packing continue unblocked while the (possibly
networked) copy runs. Input layout is ``meta.jsonl`` + ``pages/<bucket>/<doc_id>.png``;
output is ``shard-NNNNN.tar`` with ``<doc_id>.png`` + ``<doc_id>.json`` per sample.

Usage::

    python3 -m scripts.pack_wds_shards \
        --input  /path/to/rendered_dataset \
        --output /path/to/shards \
        --shard-mb 500 --workers 8
"""

import argparse
import io
import json
import os
import shutil
import tarfile
import time
import threading
import queue
import multiprocessing as mp


def _bucket(doc_id: str) -> str:
    n = int(doc_id.split("_")[1])
    return f"{n // 10000:05d}"


def _mover_thread(q, output_dir, worker_id):
    while True:
        path = q.get()
        if path is None:
            break
        try:
            dst = os.path.join(output_dir, os.path.basename(path))
            shutil.copy2(path, dst)
            os.remove(path)
        except Exception as e:
            print(f"[w{worker_id}] move error: {e}", flush=True)
        q.task_done()


def _pack_chunk(args_tuple):
    chunk_path, pages_dir, output_dir, local_tmp, shard_mb, worker_id, shard_offset = args_tuple
    shard_bytes = shard_mb * 1024 * 1024
    shard_idx = shard_offset
    cur_tar = None
    cur_size = 0
    cur_local_path = None
    packed = 0
    errors = 0
    t0 = time.time()

    move_q = queue.Queue(maxsize=3)
    mover = threading.Thread(target=_mover_thread,
                             args=(move_q, output_dir, worker_id),
                             daemon=True)
    mover.start()

    def _open():
        nonlocal cur_tar, cur_size, shard_idx, cur_local_path
        name = f"shard-{shard_idx:05d}.tar"
        cur_local_path = os.path.join(local_tmp, name)
        cur_tar = tarfile.open(cur_local_path, "w")
        cur_size = 0

    def _close():
        nonlocal cur_tar, shard_idx, cur_local_path
        if cur_tar:
            cur_tar.close()
            cur_tar = None
            if cur_local_path and os.path.exists(cur_local_path):
                move_q.put(cur_local_path)
            shard_idx += 1

    _open()

    with open(chunk_path) as f:
        for line in f:
            row = json.loads(line)
            doc_id = row["doc_id"]
            bucket = row.get("bucket") or _bucket(doc_id)
            img_path = os.path.join(pages_dir, bucket, f"{doc_id}.png")

            if not os.path.exists(img_path):
                errors += 1
                continue

            img_data = open(img_path, "rb").read()

            info = tarfile.TarInfo(name=f"{doc_id}.png")
            info.size = len(img_data)
            cur_tar.addfile(info, io.BytesIO(img_data))
            cur_size += info.size + 512

            meta = {k: v for k, v in row.items() if k != "doc_id"}
            meta_bytes = json.dumps(meta, ensure_ascii=False).encode()
            jinfo = tarfile.TarInfo(name=f"{doc_id}.json")
            jinfo.size = len(meta_bytes)
            cur_tar.addfile(jinfo, io.BytesIO(meta_bytes))
            cur_size += jinfo.size + 512

            packed += 1

            if cur_size >= shard_bytes:
                _close()
                _open()

            if packed % 100000 == 0:
                elapsed = time.time() - t0
                speed = packed / elapsed
                print(f"[w{worker_id}] {packed:,} packed  "
                      f"{speed:.0f}/s  shard {shard_idx}  err={errors}  "
                      f"moveQ={move_q.qsize()}",
                      flush=True)

    _close()
    move_q.join()
    move_q.put(None)
    mover.join()

    elapsed = time.time() - t0
    shards_written = shard_idx - shard_offset
    print(f"[w{worker_id}] DONE: {packed:,} -> {shards_written} shards, "
          f"{errors} err, {elapsed:.0f}s ({packed/max(elapsed,1):.0f}/s)", flush=True)
    return packed, shards_written, errors


def pack_parallel(input_dir: str, output_dir: str, shard_mb: int,
                  workers: int) -> None:
    meta_path = os.path.join(input_dir, "meta.jsonl")
    pages_dir = os.path.join(input_dir, "pages")
    os.makedirs(output_dir, exist_ok=True)

    local_tmp = os.path.join(input_dir, "_shard_tmp")
    os.makedirs(local_tmp, exist_ok=True)

    for f in os.listdir(output_dir):
        if f.startswith("shard-") and f.endswith(".tar"):
            os.remove(os.path.join(output_dir, f))
    for f in os.listdir(local_tmp):
        os.remove(os.path.join(local_tmp, f))
    print("[pack] cleaned old shards + tmp", flush=True)

    print("[pack] counting meta.jsonl ...", flush=True)
    total = 0
    with open(meta_path) as f:
        for _ in f:
            total += 1
    print(f"[pack] {total:,} rows, splitting into {workers} chunks", flush=True)

    chunk_size = (total + workers - 1) // workers
    samples_per_shard = 50000
    shards_per_chunk = (chunk_size // samples_per_shard) + 100

    tmp_dir = os.path.join(input_dir, "_chunks")
    os.makedirs(tmp_dir, exist_ok=True)
    chunk_paths = []
    print("[pack] splitting meta.jsonl into chunks ...", flush=True)
    with open(meta_path) as f:
        for w in range(workers):
            cpath = os.path.join(tmp_dir, f"chunk_{w:02d}.jsonl")
            chunk_paths.append(cpath)
            with open(cpath, "w") as out:
                for i in range(chunk_size):
                    line = f.readline()
                    if not line:
                        break
                    out.write(line)
    print(f"[pack] chunks ready: {[os.path.getsize(p)//1024//1024 for p in chunk_paths]} MB",
          flush=True)

    tasks = []
    for w in range(workers):
        if not os.path.exists(chunk_paths[w]) or os.path.getsize(chunk_paths[w]) == 0:
            continue
        shard_offset = w * shards_per_chunk
        tasks.append((chunk_paths[w], pages_dir, output_dir, local_tmp,
                       shard_mb, w, shard_offset))

    t0 = time.time()
    print(f"[pack] launching {len(tasks)} workers "
          f"(async move) ...", flush=True)

    with mp.Pool(len(tasks)) as pool:
        results = pool.map(_pack_chunk, tasks)

    elapsed = time.time() - t0
    total_packed = sum(r[0] for r in results)
    total_shards = sum(r[1] for r in results)
    total_errors = sum(r[2] for r in results)
    print(f"[pack] ALL DONE: {total_packed:,} samples in {total_shards} shards, "
          f"{total_errors} errors, {elapsed:.0f}s "
          f"({total_packed/max(elapsed,1):.0f} samples/s)", flush=True)

    for p in chunk_paths:
        if os.path.exists(p):
            os.remove(p)
    os.rmdir(tmp_dir)
    shutil.rmtree(local_tmp, ignore_errors=True)
    print("[pack] cleaned up temp files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard-mb", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    pack_parallel(args.input, args.output, args.shard_mb, args.workers)


if __name__ == "__main__":
    main()
