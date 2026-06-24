# TraditionalMongolianOCR

From-scratch **CRNN + CTC** line recognizer for traditional (vertical) Mongolian
script. Character-level, trained on synthetic rendered line strips streamed from
WebDataset shards.

## Why CRNN

A compact (~6M-param) character-level CRNN+CTC trained on enough rendered data
recognizes vertical Mongolian lines at low character error rate, while staying
cheap to train and run. It reads the **height** axis of a tall-narrow
single-column strip: a conv stem collapses the width to one feature vector per
height step (`T = H/4` frames), a 2-layer BiLSTM models the sequence, and CTC
marginalizes the monotonic alignment to the character labels.

Two cold-start fixes make the from-scratch model converge instead of collapsing
to the all-blank CTC attractor: the blank logit bias starts negative, and the
frame:character ratio is kept near 3:1 (not ~9:1) via `T = H/4` over
ink-cropped strips.

## Layout

```
mongocr/
  model.py      CRNN + greedy CTC decode
  losses.py     ctc_loss (F.ctc_loss wrapper, zero_infinity)
  metrics.py    normalized/raw CER, WER, exact-line (torch-free)
  alphabet.py   frozen character vocab (build / save / load)
  data.py       WebDataset streaming pipeline + preprocessing
scripts/
  build_alphabet.py   full-scan labels -> alphabet.json
  verify_dataloader.py  QA gate: index-0, no-dup-across-workers, src_doc split
  train_crnn.py       streaming trainer (two-tier eval, early stop, resume)
  eval_crnn.py        evaluate a checkpoint on the held-out split
  pack_wds_shards.py  pack rendered pages/ + meta.jsonl -> tar shards
  render_mn_synth.py  synthetic line/page renderer (Playwright)
```

## Data format

Line strips are packed into WebDataset tar shards; each sample is a
`<key>.png` (grayscale vertical strip) + `<key>.json`:

```json
{"kind": "line", "text": "<label>", "src_doc": 1234, "bucket": "00012", "font": "...", "font_px": 47}
```

`text` is the nominal-Unicode training label. `src_doc` is the source document
id, used for a **document-level** train/eval split (no adjacent-line leakage):
eval = `src_doc >= threshold`, train = `src_doc < threshold - gap`, with a gap
band dropped so a document straddling a shard boundary cannot leak.

## Usage

```bash
pip install -e .

# 1) freeze the alphabet from the full label set (never sampled)
python3 -m scripts.build_alphabet --meta /path/to/meta.jsonl --out alphabet.json

# 2) verify the dataloader before any scaled run
python3 -m scripts.verify_dataloader \
    --shards '/path/to/shards/shard-*.tar' \
    --alphabet alphabet.json --eval-threshold 430000 --gap 200

# 3) train (streams all shards; >=1 epoch coverage, CER early stop)
python3 -m scripts.train_crnn \
    --shards '/path/to/shards/shard-*.tar' \
    --alphabet alphabet.json --eval-threshold 430000 --gap 200 \
    --batch-size 64 --device cuda --save crnn.pt

# 4) evaluate a checkpoint
python3 -m scripts.eval_crnn \
    --shards '/path/to/shards/shard-*.tar' \
    --alphabet alphabet.json --ckpt crnn.pt --eval-threshold 430000
```

## CER yardstick

The primary metric is **normalized CER**: predictions and references are folded
to nominal Mongolian Unicode (drop FVS/MVS/joiners/BOM, NNBSP -> space) before
edit distance, so encoding/rendering variation is not charged as recognition
error. Raw CER is reported alongside.

## License

MIT
