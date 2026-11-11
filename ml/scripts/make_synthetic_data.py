"""Generate a small SYNTHETIC look-alike dataset for CI and pipeline smoke tests.

It mimics the NIH folder layout and file-naming scheme (so patient-level splitting is exercised), with
per-patient stain colour shifts. It is NOT medical data and results on it mean nothing beyond
"the pipeline runs end to end".

    python scripts/make_synthetic_data.py --out data/synthetic --patients 30 --cells 40
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def make_cell(rng: random.Random, infected: bool, tint: tuple[int, int, int]) -> Image.Image:
    size = rng.randint(100, 150)
    img = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = rng.randint(6, 14)
    base = tuple(max(0, min(255, c + rng.randint(-12, 12))) for c in (214, 160, 160))
    color = tuple(max(0, min(255, b + t)) for b, t in zip(base, tint))
    d.ellipse([pad, pad + rng.randint(-4, 4), size - pad, size - pad + rng.randint(-4, 4)], fill=color)
    # stain artefacts appear in both classes (as in the real data)
    for _ in range(rng.randint(0, 2)):
        x, y = rng.randint(pad + 10, size - pad - 10), rng.randint(pad + 10, size - pad - 10)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(color[0] - 30, color[1] - 30, color[2] - 10))
    if infected:  # ring-form trophozoites: small dark purple rings/dots
        for _ in range(rng.randint(1, 3)):
            r = rng.randint(4, 8)
            x, y = rng.randint(pad + 15, size - pad - 15), rng.randint(pad + 15, size - pad - 15)
            d.ellipse([x - r, y - r, x + r, y + r], outline=(95, 30, 120), width=3)
            d.ellipse([x + r - 3, y - 2, x + r + 1, y + 2], fill=(70, 15, 100))
    return img.filter(ImageFilter.GaussianBlur(0.8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--patients", type=int, default=30, help="infected patients (healthy donors = 1/3 of this)")
    ap.add_argument("--cells", type=int, default=40, help="cells per class per patient")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    for cls in ("Parasitized", "Uninfected"):
        (out / cls).mkdir(parents=True, exist_ok=True)
    n = 0
    # infected patients contribute cells to BOTH classes, like the real dataset
    for pid in range(1, a.patients + 1):
        tint = (rng.randint(-25, 25), rng.randint(-20, 20), rng.randint(-15, 25))
        slide = 30 + pid
        for i in range(a.cells):
            for cls, inf in (("Parasitized", True), ("Uninfected", False)):
                name = f"C{slide}P{pid}ThinF_IMG_20150101_{pid:06d}_cell_{i}.png"
                make_cell(rng, inf, tint).save(out / cls / name)
                n += 1
    for hid in range(1, max(2, a.patients // 3) + 1):  # healthy donors: uninfected only
        tint = (rng.randint(-25, 25), rng.randint(-20, 20), rng.randint(-15, 25))
        for i in range(a.cells):
            make_cell(rng, False, tint).save(out / "Uninfected" / f"C{hid}_thinF_IMG_20150101_{hid:06d}_cell_{i}.png")
            n += 1
    print(f"Wrote {n} synthetic cells to {out}")


if __name__ == "__main__":
    main()
