"""Turn COCO instance annotations into a per-image label matrix for the 80 objects.

Produces the ground truth that ``eval_coco80_correspondence.py`` uses to check
whether the learned image-to-text permutation actually links the same concept.

Two things this script decides, both of which change the answer:

*Salience.* A CLIP image embedding summarizes the whole photo in one vector, so
a 20-pixel object in a corner is not something the embedding represents. An
image counts as positive for a category only when that category covers at least
``--area-frac`` of the frame (0.05 by default). The unfiltered counts are
printed too, so the sensitivity is visible.

*Splitting the population.* Image-side and text-side statistics must not come
from the same photograph, or agreement between them could be an artifact of the
pairing rather than evidence about the permutation. Images are split by
``md5(image_id) % 2``: half A feeds the image side, half B feeds the text side.

Run it to check feasibility before writing any evaluation code — the summary it
prints says how many categories survive with enough positives on both sides.

    python scripts/real_alpha/build_coco80_labels.py \
        --instances cache/coco_annotations/instances_val2014.json \
        --cache-dir cache/clip_b32_coco \
        --captions cache/coco_karpathy_captions.json \
        --out cache/coco80_labels.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(_HERE))
from coco80_synonyms import COCO_80, matches  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--instances", required=True,
                   help="instances_val2014.json from annotations_trainval2014.zip")
    p.add_argument("--cache-dir", required=True,
                   help="CLIP COCO embedding cache; its splits.json defines the usable ids")
    p.add_argument("--captions", default="cache/coco_karpathy_captions.json")
    p.add_argument("--out", default="cache/coco80_labels.npz")
    p.add_argument("--area-frac", type=float, default=0.05)
    p.add_argument("--min-count", type=int, default=50,
                   help="minimum positives per side for a category to be usable")
    return p.parse_args()


def image_half(image_id: int) -> int:
    """Stable 0/1 assignment. md5 rather than hash() so it survives restarts."""
    return int(hashlib.md5(str(image_id).encode()).hexdigest(), 16) % 2


def cached_image_ids(cache_dir: str) -> set[int]:
    """Every image id the embedding cache can serve, across all splits."""
    with open(Path(cache_dir) / "splits.json") as f:
        splits = json.load(f)
    ids: set[int] = set()
    for pairs in splits.values():
        for iid, _cap_idx in pairs:
            ids.add(int(iid))
    return ids


def main() -> None:
    args = parse_args()

    logger.info("loading %s", args.instances)
    with open(args.instances) as f:
        inst = json.load(f)

    # COCO category ids are not contiguous (they run 1..90 with gaps), so map
    # them onto positions in COCO_80 by name.
    name_to_col = {c: i for i, c in enumerate(COCO_80)}
    catid_to_col = {}
    for c in inst["categories"]:
        if c["name"] in name_to_col:
            catid_to_col[c["id"]] = name_to_col[c["name"]]
    missing = set(COCO_80) - {c["name"] for c in inst["categories"]}
    if missing:
        logger.warning("categories absent from the annotation file: %s", sorted(missing))

    frame_area = {im["id"]: float(im["width"] * im["height"]) for im in inst["images"]}
    annotated_ids = set(frame_area)

    cache_ids = cached_image_ids(args.cache_dir)
    usable = sorted(annotated_ids & cache_ids)
    logger.info("annotated images %d, cache images %d, intersection %d",
                len(annotated_ids), len(cache_ids), len(usable))

    row_of = {iid: r for r, iid in enumerate(usable)}
    M = len(usable)
    area_px = np.zeros((M, 80), dtype=np.float64)
    present = np.zeros((M, 80), dtype=bool)

    for a in inst["annotations"]:
        col = catid_to_col.get(a["category_id"])
        row = row_of.get(a["image_id"])
        if col is None or row is None:
            continue
        present[row, col] = True
        # Crowd regions are counted toward area (they do cover the frame) but
        # they are still just one annotation, so no separate instance counting.
        area_px[row, col] += float(a.get("area", 0.0))

    denom = np.array([frame_area[i] for i in usable], dtype=np.float64)[:, None]
    area_frac = np.clip(area_px / denom, 0.0, 1.0).astype(np.float32)
    Y = present & (area_frac >= args.area_frac)

    image_ids = np.array(usable, dtype=np.int64)
    halves = np.array([image_half(int(i)) for i in image_ids], dtype=np.int8)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        image_ids=image_ids,
        Y=Y,
        Y_unfiltered=present,
        area_frac=area_frac,
        half=halves,
        area_frac_threshold=np.float32(args.area_frac),
    )
    logger.info("wrote %s  (M=%d images x 80 categories)", out, M)

    # ---- feasibility summary -------------------------------------------------
    captions_by_image: dict[int, list[str]] = defaultdict(list)
    cap_path = Path(args.captions)
    if cap_path.exists():
        with open(cap_path) as f:
            caps = json.load(f)
        for key, text in caps.items():
            iid = int(key.split("::")[0] if "::" in key else key.split("_")[0])
            captions_by_image[iid].append(text)
        logger.info("captions loaded for %d images", len(captions_by_image))
    else:
        logger.warning("captions file %s missing; skipping the text-side count", cap_path)

    img_half = halves == 0
    txt_half = halves == 1
    n_img = Y[img_half].sum(axis=0)
    n_txt_ann = Y[txt_half].sum(axis=0)

    # Text side with the extra lexical requirement: the caption must actually
    # mention the object. A chair in the frame that no caption names carries no
    # signal on the text side.
    n_txt_lex = np.zeros(80, dtype=np.int64)
    if captions_by_image:
        txt_rows = np.where(txt_half)[0]
        for col, cat in enumerate(COCO_80):
            hits = 0
            for r in txt_rows:
                if not Y[r, col]:
                    continue
                iid = int(image_ids[r])
                if any(matches(t, cat) for t in captions_by_image.get(iid, ())):
                    hits += 1
            n_txt_lex[col] = hits

    ok_sym = int(((n_img >= args.min_count) & (n_txt_ann >= args.min_count)).sum())
    ok_lex = int(((n_img >= args.min_count) & (n_txt_lex >= args.min_count)).sum())
    unfiltered_img = present[img_half].sum(axis=0)

    print()
    print(f"images usable                : {M}")
    print(f"area filter                  : >= {args.area_frac:.2f} of the frame")
    print(f"positives per category (img) : median {int(np.median(n_img))}, "
          f"min {int(n_img.min())}, max {int(n_img.max())} "
          f"(before area filter: median {int(np.median(unfiltered_img))})")
    print(f"categories with >= {args.min_count} on both sides")
    print(f"  annotation labels on both sides : {ok_sym} / 80")
    print(f"  text side also requires the word: {ok_lex} / 80")
    thin = [f"{COCO_80[i]}({int(n_img[i])}/{int(n_txt_ann[i])})"
            for i in range(80) if n_img[i] < args.min_count or n_txt_ann[i] < args.min_count]
    if thin:
        print(f"  too few positives: {', '.join(thin)}")
    print()
    print("KILL CHECK:", "PASS" if ok_sym >= 60 else "FAIL — fall back to caption-derived labels")


if __name__ == "__main__":
    main()
