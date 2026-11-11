"""Download and extract the NIH/NLM malaria cell dataset (~337 MB, 27,558 PNGs).

    python scripts/download_data.py               # -> data/cell_images/{Parasitized,Uninfected}

If the NIH server is unreachable, use the Kaggle mirror instead:
    kaggle datasets download -d iarunava/cell-images-for-detecting-malaria -p data --unzip
(the training code handles the mirror's duplicated nested folder automatically).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

NIH_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
EXPECTED = 27_558


def _progress(block, size, total):
    if total > 0:
        done = min(block * size, total)
        sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--url", default=NIH_URL)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / "cell_images.zip"
    if not zpath.exists():
        print(f"Downloading {a.url}")
        urllib.request.urlretrieve(a.url, zpath, _progress)
        print()
    print("Extracting…")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out)
    n = sum(1 for p in (out / "cell_images").rglob("*.png"))
    print(f"Found {n:,} PNG files under {out / 'cell_images'} (expected {EXPECTED:,}).")
    if n < EXPECTED:
        print("WARNING: fewer images than expected — the download may be incomplete.")


if __name__ == "__main__":
    main()
