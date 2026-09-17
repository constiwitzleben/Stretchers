"""Regenerate the README comparison figure.

Matches a real pair of deformed pig liver images with SuperPoint+LightGlue and
with Stretcher+LightGlue, and writes a side-by-side panel to docs/. This is the
qualitative result of Sec. 3.3 of the paper: Stretcher recovers correspondences
across the organ surface where the baseline clusters them in near-rigid regions.

    python scripts/make_readme_figure.py
"""

import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matching_util import draw_matches  # noqa: E402
from src.notebook_utils import (  # noqa: E402
    extract_superpoint_keypoints,
    get_best_device,
    matching,
    stretch_descriptions,
    stretched_matching,
)

BASE = "data/medical_deformed/pl_rest.png"
DEFORMED = "data/medical_deformed/pl_def3.png"
MODEL = "models/stretcher_superpoint.pth"
OUT = "docs/comparison.png"
WIDTH = 1400  # final panel width, px


def load_rgb(path):
    arr = np.array(Image.open(path))
    return arr[:, :, :3] if arr.ndim == 3 and arr.shape[-1] == 4 else arr


def caption(img, text):
    """Add a caption bar under a panel."""
    bar = 42
    out = Image.new("RGB", (img.width, img.height + bar), (255, 255, 255))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((14, img.height + 13), text, fill=(20, 20, 20))
    return out


def main():
    device = get_best_device(verbose=True)
    base, deformed = load_rgb(BASE), load_rgb(DEFORMED)

    feats0, feats1 = extract_superpoint_keypoints(BASE, DEFORMED, device, num_keypoints=2048)

    _, b_base, b_def = matching("lightglue", feats0, feats1, base, deformed, device, image=False)
    stretched = stretch_descriptions(feats0, device, MODEL)
    _, s_base, s_def = stretched_matching(
        "lightglue", feats0, feats1, stretched, base, deformed, device, image=False
    )

    panels = [
        caption(Image.fromarray(draw_matches(base, b_base.cpu(), deformed, b_def.cpu())),
                f"SuperPoint + LightGlue  -  {len(b_base)} matches"),
        caption(Image.fromarray(draw_matches(base, s_base.cpu(), deformed, s_def.cpu())),
                f"Stretcher + LightGlue  -  {len(s_base)} matches"),
    ]

    gap = 16
    w = max(p.width for p in panels)
    fig = Image.new("RGB", (w, sum(p.height for p in panels) + gap), (255, 255, 255))
    y = 0
    for p in panels:
        fig.paste(p, (0, y))
        y += p.height + gap

    fig = fig.resize((WIDTH, int(fig.height * WIDTH / fig.width)), Image.LANCZOS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.save(OUT, optimize=True)
    print(f"\nwrote {OUT}  ({fig.width}x{fig.height}, {os.path.getsize(OUT)/1e6:.2f} MB)")
    print(f"baseline {len(b_base)} matches -> Stretcher {len(s_base)} matches")


if __name__ == "__main__":
    main()
