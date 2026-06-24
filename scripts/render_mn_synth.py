# -*- coding: utf-8 -*-
"""Synthetic traditional-Mongolian OCR data renderer.

Two independent modes (run separately; different scales):

  --mode recog   RECOGNITION data. One corpus line -> one single-column,
                 vertical-lr strip, NOT rotated (the CRNN reads along the HEIGHT
                 axis: width is squeezed to 1, T=H/4). Output:
                   pages/<bucket>/<doc_id>.png
                   meta.jsonl  {doc_id, kind, text, src_doc, bucket, font, font_px}
                 Pack these into WebDataset shards with pack_wds_shards.py, then
                 train with train_crnn.py. RECOGNITION needs full glyph-n-gram
                 coverage, so this is the mode you run on the whole corpus.

  --mode detect  DETECTION data. Several lines -> one multi-column page with
                 per-column bboxes (programmatic, zero human labeling):
                   det_pages/<page_id>.png
                   det.jsonl   {page_id, image, image_size, font,
                                bboxes:[{x,y,w,h,text,render_text}]}
                 DETECTION learns LAYOUT, not glyphs; layout diversity saturates
                 in ~1e5-1e6 pages, so do NOT run this on the full corpus -- cap
                 with --max-pages. No DBNet trainer consumes this yet; it is
                 staged for that second phase.

Rendering: Playwright Chromium + CSS writing-mode:vertical-lr (correct cursive
shaping; PIL ttb+rotate does not shape Mongolian). Fonts are base64-EMBEDDED in
@font-face so the requested face actually loads (a file:// url silently falls
back to whatever Mongolian face fontconfig exposes -- verified Onon->Noto).

text = nominal Unicode (MVS suffix separator; the training/CTC label).
render_text = presentation form (_to_presentation: MVS->NNBSP) actually shaped.

Usage (from repo root):

  # recognition, full corpus
  PYTHONPATH=. python3 -m scripts.render_mn_synth --mode recog \
    --input /path/to/mn_traditional.clean.jsonl \
    --fonts /path/to/Onon.ttf,/path/to/Noto.ttf --out /path/to/recog_out --workers 16

  # detection, capped
  PYTHONPATH=. python3 -m scripts.render_mn_synth --mode detect \
    --input /path/to/mn_traditional.clean.jsonl \
    --fonts /path/to/Onon.ttf,/path/to/Noto.ttf --out /path/to/detect_out \
    --workers 16 --max-pages 300000
"""
from __future__ import annotations

import argparse
import atexit
import base64
import io
import json
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_WORD_SEP = " "
_MVS = "᠎"
_NNBSP = " "
_SEP_VOWELS = ("ᠠ", "ᠡ")
_MN_LETTER_LO, _MN_LETTER_HI = "ᠠ", "ᢪ"
_GLYPH_V_FACTOR = 0.62
_SC_LINE_BOX = 1.15


def _to_presentation(text: str) -> str:
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch != _MVS:
            out.append(ch)
            continue
        nxt = text[i + 1] if i + 1 < n else ""
        nxt2 = text[i + 2] if i + 2 < n else ""
        is_sep_vowel = nxt in _SEP_VOWELS and not (
            nxt2 and _MN_LETTER_LO <= nxt2 <= _MN_LETTER_HI
        )
        out.append(_MVS if is_sep_vowel else _NNBSP)
    return "".join(out)


def _normalize(text: str) -> str:
    out = []
    prev_space = False
    for ch in text:
        if ch in ("\n", "\r", "\t", " "):
            if not prev_space:
                out.append(" ")
            prev_space = True
        else:
            out.append(ch)
            prev_space = False
    return "".join(out).strip()


def _iter_docs(path, min_chars):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text")
            if isinstance(text, str) and len(text) >= min_chars:
                yield text


def _iter_lines(path, min_glyphs, max_glyphs, min_doc_chars):
    """Yield (src_doc_idx, line_text). src_doc_idx lets a downstream split be
    document-level (avoid adjacent-line train/val leakage)."""
    for doc_idx, raw_text in enumerate(_iter_docs(path, min_doc_chars)):
        words = _normalize(raw_text).split(_WORD_SEP)
        cur, cur_len = [], 0
        for w in words:
            wlen = len(w)
            if wlen > max_glyphs:
                if cur and cur_len >= min_glyphs:
                    yield doc_idx, _WORD_SEP.join(cur)
                cur, cur_len = [], 0
                continue
            add = wlen + (1 if cur else 0)
            if cur and cur_len + add > max_glyphs:
                if cur_len >= min_glyphs:
                    yield doc_idx, _WORD_SEP.join(cur)
                cur, cur_len = [w], wlen
            else:
                cur.append(w)
                cur_len += add
        if cur and cur_len >= min_glyphs:
            yield doc_idx, _WORD_SEP.join(cur)


def _geometry(font_px, line_height, margin_px, max_glyphs):
    col_w = int(round(font_px * line_height * _SC_LINE_BOX)) + 2 * margin_px
    col_h = int(round(max_glyphs * font_px * _GLYPH_V_FACTOR * 1.6)) + 2 * margin_px
    return col_w, col_h


def _embed_html(font_b64, view_w, view_h, text, font_px, line_height, margin_px, fg, bg):
    pres = _to_presentation(text)
    pres = pres.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
@font-face {{ font-family: "MnFont";
  src: url("data:font/ttf;base64,{font_b64}") format("truetype"); }}
* {{ margin: 0; box-sizing: border-box; }}
body {{
  width: {view_w}px; height: {view_h}px; overflow: hidden;
  writing-mode: vertical-lr; white-space: nowrap;
  font-family: "MnFont";
  font-size: {font_px}px; line-height: {line_height};
  padding: {margin_px}px;
  color: {fg}; background: {bg};
}}
</style></head><body>{pres}</body></html>"""


def _validate_fonts(font_paths):
    """cmap covers Mongolian block + GSUB has medial+final joining, else raise."""
    from fontTools.ttLib import TTFont
    for p in font_paths:
        t = TTFont(p, fontNumber=0, lazy=True)
        cmap = set(t.getBestCmap().keys())
        mn = [c for c in cmap if 0x1820 <= c <= 0x1842]
        feats = set()
        if "GSUB" in t and t["GSUB"].table.FeatureList:
            for fr in t["GSUB"].table.FeatureList.FeatureRecord:
                feats.add(fr.FeatureTag)
        # Two valid shaping paths: classic joining (init/medi/fina/isol, e.g.
        # Onon/Noto) OR the USE path via ccmp+rlig (+ stylistic sets/vert), e.g.
        # the Hanshi handwriting font. Both still need render-verification;
        # Menksoft has NEITHER (verified: cmap only, no ccmp/rlig) so it stays out.
        shapes = {"medi", "fina"}.issubset(feats) or {"ccmp", "rlig"}.issubset(feats)
        if len(mn) < 20 or not shapes:
            raise RuntimeError(
                f"font unfit for Mongolian shaping: {Path(p).name} "
                f"(mn_cmap={len(mn)}, joining={sorted(feats & {'init','medi','fina','isol'})}, "
                f"ccmp_rlig={ {'ccmp','rlig'}.issubset(feats) })")
        t.close()


def _tight_crop(img, pad=6):
    import numpy as np
    a = np.asarray(img.convert("L"), dtype=np.int16)
    border = np.concatenate(
        [a[:3].ravel(), a[-3:].ravel(), a[:, :3].ravel(), a[:, -3:].ravel()])
    bg = int(np.percentile(border, 90))
    ink = a < (bg - 45)
    rows = np.nonzero(ink.sum(axis=1) > 0)[0]
    cols = np.nonzero(ink.sum(axis=0) > 0)[0]
    if rows.size == 0 or cols.size == 0:
        return None
    t = max(0, int(rows[0]) - pad)
    b = min(a.shape[0], int(rows[-1]) + 1 + pad)
    le = max(0, int(cols[0]) - pad)
    r = min(a.shape[1], int(cols[-1]) + 1 + pad)
    return img.crop((le, t, r, b))


def _degrade(img, rng, *, allow_geometric=True, bleed_src=None):
    """Physics print+scan degradation (DocCreator/Kanungo/STRAug/ocrodeg):
    ink erode/dilate, paper substrate, illumination, fold-crease shadow, verso
    bleed, sensor noise + codec, plus perspective/skew/wrinkle.

    allow_geometric=False skips every pixel-MOVING step so page-level bboxes stay
    valid; the colour-only steps are kept. False for detection pages, True for an
    isolated recognition strip (whole strip is the input)."""
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    try:
        import cv2
    except Exception:
        cv2 = None

    npr = np.random.default_rng(rng.randint(0, 1 << 30))
    g = np.asarray(img.convert("L"), dtype=np.float32)
    H, W = g.shape
    ink_frac = float((g < 128).mean())

    if cv2 is not None and rng.random() < 0.8:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rng.choice([2, 3]),) * 2)
        g = cv2.erode(g, k) if rng.random() < 0.5 else cv2.dilate(g, k)

    if bleed_src is not None and ink_frac > 0.04 and rng.random() < 0.5:
        b = np.asarray(bleed_src.convert("L").resize((W, H)), dtype=np.float32)[:, ::-1]
        if cv2 is not None:
            b = cv2.GaussianBlur(b, (0, 0), rng.uniform(1.0, 2.2))
        g = np.clip(g - (255.0 - b) * rng.uniform(0.05, 0.14), 0, 255)

    if rng.random() < 0.9:
        lf = npr.normal(0, 1, (max(2, H // 32), max(2, W // 32))).astype(np.float32)
        lf = (cv2.resize(lf, (W, H), interpolation=cv2.INTER_CUBIC)
              if cv2 is not None else np.kron(lf, np.ones((32, 32)))[:H, :W])
        g = g * (1.0 + (lf - lf.mean()) * rng.uniform(0.015, 0.04))
        g = np.where(g > 180, g + npr.normal(0, rng.uniform(2, 6), g.shape), g)

    if rng.random() < 0.75:
        gy, gx = np.mgrid[0:H, 0:W]
        gx = gx.astype(np.float32) / W - 0.5
        gy = gy.astype(np.float32) / H - 0.5
        g = g * (1.0 - rng.uniform(0.06, 0.20)
                 * (gx * rng.uniform(-1, 1) + gy * rng.uniform(-1, 1) + 0.5))
        if rng.random() < 0.5:
            r = np.sqrt(gx * gx + gy * gy)
            g = g * (1.0 - rng.uniform(0.0, 0.22) * np.clip(r * 1.6 - 0.4, 0, 1))

    if cv2 is not None and allow_geometric and rng.random() < 0.6:
        gh, gw = max(2, H // 48), max(2, W // 16)
        amp = rng.uniform(2.0, 7.0)
        dx = cv2.resize(npr.normal(0, 1, (gh, gw)).astype(np.float32), (W, H),
                        interpolation=cv2.INTER_CUBIC) * amp
        dy = cv2.resize(npr.normal(0, 1, (gh, gw)).astype(np.float32), (W, H),
                        interpolation=cv2.INTER_CUBIC) * amp
        xx, yy = np.meshgrid(np.arange(W, dtype=np.float32),
                             np.arange(H, dtype=np.float32))
        g = cv2.remap(g, xx + dx, yy + dy, interpolation=cv2.INTER_LINEAR,
                      borderValue=250.0)
        shade = 1.0 - rng.uniform(0.05, 0.14) * np.abs(cv2.Laplacian(dx + dy, cv2.CV_32F))
        g = g * np.clip(shade, 0.7, 1.05)

    if rng.random() < 0.3:
        sig = rng.uniform(3.0, 8.0)
        if rng.random() < 0.5:
            y0 = rng.randint(int(H * 0.2), int(H * 0.8))
            band = np.exp(-((np.arange(H) - y0) ** 2) / (2 * sig ** 2))
            g = g * (1.0 - rng.uniform(0.10, 0.25) * band[:, None])
        else:
            x0 = rng.randint(int(W * 0.2), int(W * 0.8))
            band = np.exp(-((np.arange(W) - x0) ** 2) / (2 * sig ** 2))
            g = g * (1.0 - rng.uniform(0.10, 0.25) * band[None, :])

    out = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8), "L")

    if allow_geometric:
        if cv2 is not None and rng.random() < 0.7:
            j = rng.uniform(0.02, 0.06)
            src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
            dst = (src + npr.uniform(-j, j, src.shape).astype(np.float32)
                   * np.array([W, H], np.float32)).astype(np.float32)
            warp = cv2.warpPerspective(
                np.asarray(out), cv2.getPerspectiveTransform(src, dst),
                (W, H), borderValue=250, flags=cv2.INTER_LINEAR)
            out = Image.fromarray(warp, "L")
        elif rng.random() < 0.5:
            out = out.rotate(rng.uniform(-0.9, 0.9), fillcolor=250, expand=False)

    if rng.random() < 0.7:
        out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.4)))
    if rng.random() < 0.7:
        gg = (np.asarray(out, dtype=np.float32)
              + npr.normal(0, rng.uniform(4, 14), (out.height, out.width)))
        out = Image.fromarray(np.clip(gg, 0, 255).astype(np.uint8), "L")
    if rng.random() < 0.5:
        out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.9, 1.12))

    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=rng.randint(30, 72))
    buf.seek(0)
    out = Image.open(buf).convert("L")
    out.load()
    return out


def _compose_page(col_imgs, rng, page_w, page_h, bg_val, margin=30, col_gap=18):
    """Paste vertical column strips left-to-right; (page, bboxes, placed).
    bg_val == the columns' own bg grey -> seamless paste (no rectangle the
    detector could cheat on)."""
    from PIL import Image
    page = Image.new("L", (page_w, page_h), bg_val)
    bboxes = []
    x_cursor = margin
    for cimg in col_imgs:
        cw, ch = cimg.size
        if ch > page_h - 2 * margin:
            scale = (page_h - 2 * margin) / ch
            cw = max(1, int(cw * scale))
            ch = page_h - 2 * margin
            cimg = cimg.resize((cw, ch))
        if x_cursor + cw + margin > page_w:
            break
        y = margin + rng.randint(0, max(0, page_h - 2 * margin - ch))
        page.paste(cimg, (x_cursor, y))
        bboxes.append((x_cursor, y, cw, ch))
        x_cursor += cw + col_gap
    return page, bboxes, len(bboxes)


# --- per-worker Playwright ---------------------------------------------------
_W = {}


def _base_html(font_b64s):
    """Page loaded ONCE per worker: all fonts embedded, empty vertical-lr body.
    Each strip then only swaps text/style via DOM (no re-parsing the font, no
    set_content) -- the 5x win the profiler showed."""
    faces = "".join(
        f'@font-face{{font-family:"F{i}";'
        f'src:url("data:font/ttf;base64,{b}") format("truetype")}}'
        for i, b in enumerate(font_b64s))
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{faces}'
            '*{margin:0;box-sizing:border-box}'
            'body{writing-mode:vertical-lr;white-space:nowrap;overflow:hidden}'
            '</style></head><body></body></html>')


def _setup_base_page(page, font_b64s):
    page.set_content(_base_html(font_b64s), timeout=20000)
    page.evaluate("() => document.fonts.ready")


def _worker_init(font_paths, view_w, view_h):
    from playwright.sync_api import sync_playwright
    _validate_fonts(font_paths)
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
    page = browser.new_page(viewport={"width": view_w, "height": view_h})
    font_b64 = [base64.b64encode(Path(p).read_bytes()).decode() for p in font_paths]
    _W.update({
        "pw": pw, "browser": browser, "page": page,
        "font_b64": font_b64,
        "font_names": [Path(p).stem for p in font_paths],
        "view_w": view_w, "view_h": view_h,
    })
    _setup_base_page(page, font_b64)  # embed fonts ONCE; strips swap text/style only
    atexit.register(_worker_close)  # stdlib Pool has no per-worker teardown hook


def _worker_close():
    if _W:
        try:
            _W["browser"].close()
            _W["pw"].stop()
        except Exception:
            pass
        _W.clear()


def _ensure_page():
    if _W.get("browser") is None or not _W["browser"].is_connected():
        _W["browser"] = _W["pw"].chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-gpu"])
        _W["page"] = None
    if _W.get("page") is None:
        _W["page"] = _W["browser"].new_page(
            viewport={"width": _W["view_w"], "height": _W["view_h"]})
        _setup_base_page(_W["page"], _W["font_b64"])  # re-embed fonts on rebuild
    return _W["page"]


def _reset_page():
    try:
        if _W.get("page") is not None:
            _W["page"].close()
    except Exception:
        pass
    _W["page"] = None


def _render_column(text, font_idx, fg, bg, font_px, line_height, margin_px, retries=3):
    """Render one vertical-lr column; tight-cropped L image (or None on blank).
    Retries on transient Chromium errors, rebuilding the page between tries."""
    from PIL import Image
    pres = _to_presentation(text)
    clip = {"x": 0, "y": 0, "width": _W["view_w"], "height": _W["view_h"]}
    style = {"ff": f"F{font_idx}", "fs": font_px, "lh": line_height,
             "mp": margin_px, "fg": fg, "bg": bg, "t": pres}
    last = None
    for _ in range(retries):
        try:
            page = _ensure_page()
            # swap only text + style on the pre-loaded font page (no set_content,
            # no font re-parse, no fixed wait). textContent auto-escapes.
            page.evaluate(
                """(a)=>{const b=document.body;
                b.style.fontFamily=a.ff;b.style.fontSize=a.fs+'px';
                b.style.lineHeight=a.lh;b.style.padding=a.mp+'px';
                b.style.color=a.fg;b.style.background=a.bg;
                b.textContent=a.t;}""", style)
            png = page.screenshot(clip=clip)
            img = Image.open(io.BytesIO(png)).convert("L")
            img.load()
            return _tight_crop(img)
        except Exception as exc:
            last = exc
            _reset_page()
    raise RuntimeError(f"render_column failed after {retries} tries: {last}")


_PALETTE_FG = ["#101010", "#1a1a2a", "#262626", "#30281e"]
_PALETTE_BG = ["#ffffff", "#faf6ee", "#f4ead8", "#f0e6d2", "#efefef"]


def _pick_style(rng, fpx_min, fpx_max, lh_min, lh_max):
    return {
        "fg": rng.choice(_PALETTE_FG),
        "bg": rng.choice(_PALETTE_BG),
        "font_px": rng.randint(fpx_min, fpx_max),
        "line_height": round(rng.uniform(lh_min, lh_max), 2),
    }


# --- recognition mode: one line -> one NON-rotated single-column strip --------
def _render_recog(task):
    try:
        return _render_recog_impl(task)
    except Exception as exc:
        sys.stderr.write(f"[render] DROP {task[0]}: {type(exc).__name__}: {exc}\n")
        sys.stderr.flush()
        return "DROPPED"


def _render_recog_impl(task):
    (doc_id, src_doc, text, font_idx, seed, degrade_ratio, out_dir,
     margin_px, fpx_min, fpx_max, lh_min, lh_max) = task
    rng = random.Random(seed)
    st = _pick_style(rng, fpx_min, fpx_max, lh_min, lh_max)
    # font_idx is variant-controlled by the stream so each text covers all faces
    cimg = _render_column(text, font_idx, st["fg"], st["bg"],
                          st["font_px"], st["line_height"], margin_px)
    if cimg is None or cimg.width < 6 or cimg.height < 10:
        return None
    if rng.random() < degrade_ratio:
        cimg = _degrade(cimg, rng, allow_geometric=True)
    bucket = _doc_bucket(doc_id)
    bset = _W.setdefault("buckets", set())
    pdir = Path(out_dir, "pages", bucket)
    if bucket not in bset:
        pdir.mkdir(parents=True, exist_ok=True)
        bset.add(bucket)
    pdir.joinpath(f"{doc_id}.png").write_bytes(_png_bytes(cimg))
    meta_row = {
        "doc_id": doc_id, "kind": "line", "text": text, "src_doc": src_doc,
        "bucket": bucket, "font": _W["font_names"][font_idx], "font_px": st["font_px"],
    }
    return ("REC", meta_row)


# --- detection mode: several lines -> one multi-column page + bboxes ----------
def _render_detect(task):
    try:
        return _render_detect_impl(task)
    except Exception as exc:
        sys.stderr.write(f"[render] DROP {task[0]}: {type(exc).__name__}: {exc}\n")
        sys.stderr.flush()
        return "DROPPED"


def _render_detect_impl(task):
    import numpy as np
    (page_id, line_texts, seed, degrade_ratio, out_dir,
     page_w, page_h, margin_px, fpx_min, fpx_max, lh_min, lh_max) = task
    rng = random.Random(seed)
    st = _pick_style(rng, fpx_min, fpx_max, lh_min, lh_max)
    font_idx = rng.randrange(len(_W["font_names"]))

    col_imgs, valid_texts = [], []
    for text in line_texts:
        cimg = _render_column(text, font_idx, st["fg"], st["bg"],
                              st["font_px"], st["line_height"], margin_px)
        if cimg is None or cimg.width < 6 or cimg.height < 10:
            continue
        col_imgs.append(cimg)
        valid_texts.append(text)
    if not col_imgs:
        return None

    bg_val = int(np.asarray(col_imgs[0])[:4, :4].mean())
    page_img, bboxes, placed = _compose_page(col_imgs, rng, page_w, page_h, bg_val)
    if placed == 0:
        return None
    valid_texts = valid_texts[:placed]

    if rng.random() < degrade_ratio:
        page_img = _degrade(page_img, rng, allow_geometric=False)  # keep bboxes
    Path(out_dir, "det_pages", f"{page_id}.png").write_bytes(_png_bytes(page_img))

    det_row = {
        "page_id": page_id, "image": f"det_pages/{page_id}.png",
        "image_size": [page_img.height, page_img.width],
        "font": _W["font_names"][font_idx],
        "bboxes": [{"x": b[0], "y": b[1], "w": b[2], "h": b[3],
                    "text": t, "render_text": _to_presentation(t)}
                   for b, t in zip(bboxes, valid_texts)],
    }
    return ("DET", det_row)


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _derive_seed(base, idx):
    """Deterministic per-item seed independent of any global RNG stream, so a
    resumed run reproduces identical items (no RNG desync)."""
    return (base * 1000003 + idx * 2654435761 + 12345) & 0x3FFFFFFF


_BUCKET_SIZE = 10000


def _doc_bucket(doc_id):
    """Shard sub-directory for a recog doc_id ('line_<n>'). At full scale ~1e8
    strips would overflow a single ext4 dir (htree ~1e7 cap, large_dir off by
    default); ~1e4 strips per bucket keeps every dir small. The packer derives
    the identical bucket to find the image."""
    try:
        n = int(doc_id.split("_")[1])  # line_<n> or line_<n>_v<k>
    except (IndexError, ValueError):
        return "00000"
    return f"{n // _BUCKET_SIZE:05d}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthetic Mongolian OCR renderer (Playwright)")
    ap.add_argument("--mode", choices=["recog", "detect"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--fonts", required=True, help="comma-separated abs .ttf paths")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-docs", type=int, default=0, help="recog: cap lines (0=all)")
    ap.add_argument("--augment", type=int, default=1,
                    help="recog: render each line K times with different face/size/"
                    "degrade for diversity (K=2 -> both faces per text, ~2x disk)")
    ap.add_argument("--max-pages", type=int, default=0, help="detect: cap pages (0=all)")
    ap.add_argument("--font-px-min", type=int, default=30)
    ap.add_argument("--font-px-max", type=int, default=50)
    ap.add_argument("--line-height-min", type=float, default=1.4)
    ap.add_argument("--line-height-max", type=float, default=1.7)
    ap.add_argument("--margin-px", type=int, default=16)
    ap.add_argument("--max-glyphs", type=int, default=28)
    ap.add_argument("--min-glyphs", type=int, default=8)
    ap.add_argument("--page-w", type=int, default=800)
    ap.add_argument("--page-h", type=int, default=1200)
    ap.add_argument("--cols-per-page-min", type=int, default=3)
    ap.add_argument("--cols-per-page-max", type=int, default=8)
    ap.add_argument("--degrade-ratio", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    font_paths = [p.strip() for p in args.fonts.split(",") if p.strip()]
    for fp in font_paths:
        if not Path(fp).exists():
            print(f"font not found: {fp}", file=sys.stderr)
            return 1
    _validate_fonts(font_paths)
    print(f"[render] mode={args.mode} fonts validated: {[Path(p).name for p in font_paths]}")

    # viewport sized for the largest possible column (max font/glyphs); smaller
    # styles render inside it and are tight-cropped out.
    view_w, view_h = _geometry(args.font_px_max, args.line_height_max,
                               args.margin_px, args.max_glyphs)

    out_dir = Path(args.out)
    img_subdir = "pages" if args.mode == "recog" else "det_pages"
    (out_dir / img_subdir).mkdir(parents=True, exist_ok=True)
    rec_file = out_dir / ("meta.jsonl" if args.mode == "recog" else "det.jsonl")
    id_key = "doc_id" if args.mode == "recog" else "page_id"

    done_ids = set()
    if args.resume and rec_file.exists():
        with open(rec_file, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done_ids.add(json.loads(line)[id_key])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[render] resume: {len(done_ids)} items already done")

    def _stream_recog():
        n = 0
        K = max(1, args.augment)
        nf = len(font_paths)
        for src_doc, text in _iter_lines(args.input, args.min_glyphs, args.max_glyphs, 20):
            if args.max_docs and n >= args.max_docs:
                break
            for k in range(K):
                # variant k uses face k%nf so K>=nf covers every face per text;
                # size/degrade differ via the per-variant seed. Both variants of
                # one text share src_doc -> document-level split keeps them together.
                doc_id = f"line_{n:08d}_v{k}"
                if doc_id not in done_ids:
                    yield (doc_id, src_doc, text, k % nf,
                           _derive_seed(args.seed, n * K + k),
                           args.degrade_ratio, str(out_dir), args.margin_px,
                           args.font_px_min, args.font_px_max,
                           args.line_height_min, args.line_height_max)
            n += 1

    def _stream_detect():
        n, batch = 0, []
        for _src_doc, text in _iter_lines(args.input, args.min_glyphs, args.max_glyphs, 20):
            if args.max_pages and n >= args.max_pages:
                break
            batch.append(text)
            # cols target derived deterministically (no global RNG draw -> no desync)
            ncols = args.cols_per_page_min + (
                _derive_seed(args.seed, n) % (args.cols_per_page_max - args.cols_per_page_min + 1))
            if len(batch) >= ncols:
                page_id = f"page_{n:08d}"
                if page_id not in done_ids:
                    yield (page_id, batch, _derive_seed(args.seed, n),
                           args.degrade_ratio, str(out_dir), args.page_w, args.page_h,
                           args.margin_px, args.font_px_min, args.font_px_max,
                           args.line_height_min, args.line_height_max)
                n += 1
                batch = []

    worker_fn = _render_recog if args.mode == "recog" else _render_detect
    stream = _stream_recog if args.mode == "recog" else _stream_detect

    print(f"[render] viewport {view_w}x{view_h}, {args.workers} workers", flush=True)
    rec_fh = open(rec_file, "a", encoding="utf-8")
    n_ok = n_drop = n_empty = 0
    # explicit close()/join() (not `with`, which terminate()s workers mid-task)
    # lets each worker exit cleanly so its atexit(_worker_close) shuts Chromium.
    pool = Pool(args.workers, initializer=_worker_init,
                initargs=(font_paths, view_w, view_h))
    try:
        for result in pool.imap_unordered(worker_fn, stream(), chunksize=8):
            if result == "DROPPED":
                n_drop += 1
                continue
            if result is None:
                n_empty += 1
                continue
            _kind, row = result
            rec_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1
            if n_ok % 500 == 0:
                rec_fh.flush()
                print(f"[render] {n_ok} items ({n_drop} dropped, {n_empty} empty)",
                      flush=True)
        pool.close()
        pool.join()
    finally:
        pool.terminate()
        rec_fh.close()

    print(f"[render] done: {n_ok} items -> {out_dir} "
          f"({n_drop} dropped on error, {n_empty} empty-skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
