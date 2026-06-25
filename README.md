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
id, used for a **document-level** train/val/test split (no adjacent-line
leakage): test = `src_doc >= test_threshold`, val = `[val_threshold,
test_threshold - gap)`, train = `src_doc < val_threshold - gap`. Gap bands are
dropped between splits so a document straddling a shard boundary cannot leak.
Early stopping selects on val; the headline CER is reported once on test.

## Charset curation & data cleaning

Web-scraped Mongolian text carries a long noise tail of stray code points (stray
emoji, foreign-script fragments, box-drawing, hapax CJK). The alphabet is curated
by a frequency floor (`build_alphabet --min-count`, default 10): characters
seen fewer than that across the whole corpus are dropped rather than turned into
dead CTC classes. A **line containing any dropped character is removed from
training** — the rendered image still shows that glyph, so keeping the line and
silently dropping the char from the target would teach the model to skip glyphs.
At `--min-count 10` the dropped code points are < 0.001% of all occurrences.

## Usage

```bash
pip install -e .

# 1) freeze the alphabet from the full label set (never sampled)
python3 -m scripts.build_alphabet --meta /path/to/meta.jsonl --out alphabet.json

# 2) verify the dataloader before any scaled run
python3 -m scripts.verify_dataloader \
    --shards '/path/to/shards/shard-*.tar' \
    --alphabet alphabet.json --val-threshold 431718 --test-threshold 433718 --gap 200

# 3) extract val/test into small local shards (one-time; the val/test src_doc
#    bands live only in the last source shards, so copy them out once for fast
#    eval instead of streaming the whole corpus to find them every evaluation)
python3 -m scripts.extract_eval_shards \
    --src '/path/to/shards/shard-*.tar' --tail 40 --out-dir /local/eval_cache \
    --val-threshold 431718 --test-threshold 433718 --gap 200

# 4) train (streams all shards for train; evaluates on the local val shards)
python3 -m scripts.train_crnn \
    --shards '/path/to/shards/shard-*.tar' \
    --eval-shards '/local/eval_cache/val-*.tar' \
    --alphabet alphabet.json --val-threshold 431718 --test-threshold 433718 --gap 200 \
    --batch-size 128 --device cuda --save crnn.pt

# 5) evaluate a checkpoint on the untouched test split (local test shards)
python3 -m scripts.eval_crnn \
    --shards '/local/eval_cache/test-*.tar' \
    --alphabet alphabet.json --ckpt crnn.pt \
    --val-threshold 431718 --test-threshold 433718 --split test
```

## CER yardstick

The primary metric is **normalized CER**: predictions and references are folded
to nominal Mongolian Unicode (drop FVS/MVS/joiners/BOM, NNBSP -> space) before
edit distance, so encoding/rendering variation is not charged as recognition
error. Raw CER is reported alongside.

## License

MIT
