"""On the server: load HF dataset namkha1032/coco-karpathy and dump the
images for the iids in needed_iids.json as JPG files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iids-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="test")
    args = p.parse_args()

    iids = set(int(i) for i in json.load(open(args.iids_json)))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("namkha1032/coco-karpathy", split=args.split)
    print(f"dataset {args.split}: {len(ds)} rows; need {len(iids)} images")

    saved = 0
    for row in tqdm(ds):
        iid = int(row["image_id"]) if "image_id" in row else int(row["id"])
        if iid not in iids:
            continue
        path = out / f"{iid}.jpg"
        if path.exists():
            saved += 1
            continue
        img = row["image"]
        # PIL Image → save as JPEG
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(path, "JPEG", quality=85)
        saved += 1
    print(f"saved {saved}/{len(iids)} images → {out}")


if __name__ == "__main__":
    main()
