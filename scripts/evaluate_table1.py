"""Reproduce Table 1 of the Stretcher paper.

Evaluates keypoint matching under FEM-simulated soft-tissue deformation, for
each combination of a descriptor (DISK, ALIKED, SuperPoint, Stretcher) and a
matcher (Dual Softmax, LightGlue).

A liver image is deformed with a linear-elastic FEM model under four load
cases. Because the displacement field is known, every match can be checked
against ground truth, which gives:

  precision      fraction of matches within `--threshold` px of ground truth
  match score    correct matches / min(#keypoints in either image)
  # matches      number of correspondences returned
  entropy        normalised entropy of the local strain magnitude at correct
                 matches - high when correct matches are spread evenly across
                 deformation levels rather than clustered in rigid regions
  SBP            strain-balanced precision: precision averaged over ten equally
                 populated strain bins

Reported values are mean +/- std over the four load cases.

Usage
-----
    python scripts/evaluate_table1.py                    # full table
    python scripts/evaluate_table1.py --methods sp stretcher --matchers dsm
    python scripts/evaluate_table1.py --output results/

This script supersedes the earlier table_making.py and table_making_stretcher.py,
which shared ~700 lines of copy-paste. Divergences from those scripts are noted
in NOTES_ON_PORT below.
"""

# NOTES_ON_PORT
# -------------
# Three bugs in the original scripts are fixed here; each changes a published
# number, so they are called out rather than silently corrected.
#
# 1. Stretcher+LightGlue match score. The original computed
#        min(len(feats0['keypoints']), len(feats1['keypoints']))
#    on tensors that still had a batch dimension, so len() returned 1 and the
#    score came out as a match count rather than a fraction. Table 1 reports
#    '-' for this cell. Using .shape[1] gives a real value.
#
# 2. Stretcher+LightGlue match count. The original truncated to the top 200
#    scoring matches with a hardcoded [:200] (its comment said 500), which is
#    why Table 1 reports '200.0 +/- 0.0' with zero variance. The cap is now
#    --lg-topk, default 200 to reproduce the published number.
#
# 3. Loop variable shadowing. The outer `for i, deformation in ...` was
#    clobbered by two inner `for i in range(...)` loops, corrupting the
#    visualisation output path. Affected saved figures only, not metrics.

import argparse
import json
import os
import sys
import time

# DISK's keypoint detector calls torch.kthvalue, which has no MPS kernel, so on
# Apple Silicon the DISK rows abort with NotImplementedError. This is PyTorch's
# documented escape hatch and must be set before torch initialises. It only
# affects operators MPS lacks; everything else still runs on the GPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightglue import ALIKED, DISK, LightGlue, SuperPoint  # noqa: E402
from lightglue.utils import load_image, rbd  # noqa: E402
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher  # noqa: E402

from src.affine_transformations import generate_strain_tensors  # noqa: E402
from src.dsm_matching import StretcherDualSoftMaxMatcher  # noqa: E402
from src.fenics_deformation import (  # noqa: E402
    create_deformed_medical_image_pair,
    track_pixel_displacement,
)
from src.matching_util import strain_balanced_precision, strain_entropy  # noqa: E402
from src.models import TripleNet  # noqa: E402
from src.notebook_utils import get_best_device  # noqa: E402

# The four FEM load cases (g_zy, g_zx) behind the mean +/- std in Table 1.
DEFORMATIONS = np.array([[8e6, 1e6], [8e6, -1e6], [-4e6, 2e6], [-4e6, -2e6]])

EXTRACTORS = {
    "disk": (DISK, "disk"),
    "aliked": (ALIKED, "aliked"),
    "sp": (SuperPoint, "superpoint"),
    "stretcher": (SuperPoint, "superpoint"),  # Stretcher builds on SuperPoint
}

METRICS = ["precision", "matching_score", "num_matches", "entropy", "sb_precision"]


def load_rgb(path):
    """Read an image as an HxWx3 uint8 array, dropping any alpha channel."""
    arr = np.array(Image.open(path))
    return arr[:, :, :3] if arr.ndim == 3 and arr.shape[-1] == 4 else arr


def score_matches(base_matches, deformed_matches, info, n_kp, threshold):
    """Score one set of correspondences against the FEM ground truth."""
    gt = np.array([track_pixel_displacement(p, info) for p in base_matches.cpu()])
    distances = (deformed_matches.cpu() - gt).norm(dim=1)
    good = distances < threshold

    num_matches = len(distances)
    if num_matches == 0:
        return {m: 0.0 for m in METRICS}

    W, H = info["W"], info["H"]
    return {
        "precision": good.sum().item() / num_matches,
        "matching_score": good.sum().item() / n_kp,
        "num_matches": num_matches,
        "entropy": strain_entropy(info["s"], base_matches[good].cpu(), W, H),
        "sb_precision": strain_balanced_precision(
            info["s"], base_matches.cpu(), good.numpy(), W, H
        ),
    }


def stretch(descriptors, stretcher, device):
    """Generate one descriptor hypothesis per affine deformation mode."""
    tensors = np.array(generate_strain_tensors())
    with torch.no_grad():
        out = np.array([
            stretcher(
                descriptors.cpu().to(torch.float32).to(device),
                torch.tensor(t).to(torch.float32).to(device).repeat(len(descriptors), 1),
            ).cpu()
            for t in tensors
        ])
    return torch.tensor(out).to(device)


def match_lightglue_stretched(lg, feats0, feats1, stretched, device, topk):
    """LightGlue over every descriptor hypothesis, keeping each keypoint's best match."""
    best_def, best_base, best_score = {}, {}, {}

    for h in range(stretched.shape[0]):
        feats0 = dict(feats0)
        feats0["descriptors"] = stretched[h][None].to(device)
        out = lg({"image0": feats0, "image1": feats1})
        matches, scores = out["matches"][0], out["scores"][0]

        base_kp = feats0["keypoints"][0][matches[:, 0]]
        def_kp = feats1["keypoints"][0][matches[:, 1]]

        for j, idx_t in enumerate(matches[:, 0]):
            idx, score = int(idx_t.item()), float(scores[j].item())
            if idx not in best_score or score > best_score[idx]:
                best_def[idx], best_base[idx], best_score[idx] = def_kp[j], base_kp[j], score

    if not best_score:
        raise RuntimeError("no LightGlue matches across any deformation hypothesis")

    base = torch.stack(list(best_base.values()))
    deformed = torch.stack(list(best_def.values()))
    order = torch.tensor(list(best_score.values()), device=device).argsort(descending=True)[:topk]
    return base[order], deformed[order]


def evaluate(method, matcher_name, paths, info, device, stretcher, args):
    """Run one (method, matcher) cell of the table for one deformation."""
    extractor_cls, lg_feature = EXTRACTORS[method]
    extractor = extractor_cls(max_num_keypoints=args.num_keypoints).eval().to(device)

    feats0 = extractor.extract(load_image(paths[0]).to(device))
    feats1 = extractor.extract(load_image(paths[1]).to(device))
    n_kp = min(feats0["keypoints"].shape[1], feats1["keypoints"].shape[1])

    stretched = stretch(feats0["descriptors"][0], stretcher, device) if method == "stretcher" else None

    if matcher_name == "dsm":
        if method == "stretcher":
            base, deformed, _ = StretcherDualSoftMaxMatcher().match(
                feats0["keypoints"].to(device), stretched.to(device),
                feats1["keypoints"].to(device), feats1["descriptors"].to(device),
                P_A=feats0["keypoint_scores"][0], P_B=feats1["keypoint_scores"][0],
                normalize=True, inv_temp=args.inv_temp, threshold=0.03,
            )
        else:
            base, deformed, _ = DualSoftMaxMatcher().match(
                feats0["keypoints"].to(device), feats0["descriptors"].to(device),
                feats1["keypoints"].to(device), feats1["descriptors"].to(device),
                P_A=None, P_B=None,
                normalize=True, inv_temp=args.inv_temp, threshold=0.01,
            )
    else:
        lg = LightGlue(features=lg_feature).eval().to(device)
        if method == "stretcher":
            base, deformed = match_lightglue_stretched(
                lg, feats0, feats1, stretched, device, args.lg_topk
            )
        else:
            out = lg({"image0": feats0, "image1": feats1})
            f0, f1, out = (rbd(x) for x in (feats0, feats1, out))
            m = out["matches"]
            base = f0["keypoints"][m[..., 0]]
            deformed = f1["keypoints"][m[..., 1]]

    return score_matches(base, deformed, info, n_kp, args.threshold)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["disk", "aliked", "sp", "stretcher"],
                    choices=list(EXTRACTORS))
    ap.add_argument("--matchers", nargs="+", default=["dsm", "lg"], choices=["dsm", "lg"])
    ap.add_argument("--image", default="data/medical_deformed/pig_liver_to_elongate.png")
    ap.add_argument("--deformed-image", default="data/medical_deformed/pig_liver_to_elongate_deformed.png")
    ap.add_argument("--model", default="models/stretcher_superpoint.pth")
    ap.add_argument("--num-keypoints", type=int, default=2048)
    ap.add_argument("--threshold", type=float, default=5.0, help="correct-match radius, px")
    ap.add_argument("--inv-temp", type=float, default=20.0)
    ap.add_argument("--lg-topk", type=int, default=200,
                    help="cap on Stretcher+LightGlue matches; 200 reproduces the paper")
    ap.add_argument("--hidden-dim", type=int, default=2048)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_best_device(verbose=True)

    stretcher = None
    if "stretcher" in args.methods:
        stretcher = TripleNet(256, 3, hidden_dim=args.hidden_dim,
                              num_layers=args.num_layers).float().to(device)
        stretcher.load_state_dict(torch.load(args.model, map_location=device))
        stretcher.eval()

    results = {(m, k): {metric: [] for metric in METRICS}
               for m in args.methods for k in args.matchers}

    started = time.time()
    for d_i, (g_zy, g_zx) in enumerate(DEFORMATIONS, start=1):
        print(f"\n=== deformation {d_i}/{len(DEFORMATIONS)}  g_zy={g_zy:.0e} g_zx={g_zx:.0e} ===")
        _, _, info = create_deformed_medical_image_pair(
            args.image, args.deformed_image, g_zy, g_zx, show=False
        )
        for method in args.methods:
            for matcher_name in args.matchers:
                scores = evaluate(method, matcher_name,
                                  (args.image, args.deformed_image),
                                  info, device, stretcher, args)
                for metric, value in scores.items():
                    results[(method, matcher_name)][metric].append(value)
                print(f"  {method:9s} {matcher_name:3s}  "
                      f"prec={scores['precision']*100:5.2f}%  "
                      f"n={scores['num_matches']:4d}  "
                      f"H={scores['entropy']:.2f}  SBP={scores['sb_precision']:.2f}")

    print(f"\ncompleted in {time.time() - started:.1f}s")
    report(results, args)


def report(results, args):
    """Print Table 1 and write it to CSV, Markdown and JSON."""
    header = f"{'Matcher':<8}{'Method':<12}{'Prec. (%)':>16}{'Match Sc. (%)':>18}{'# Matches':>16}{'Entropy':>14}{'SBP':>14}"
    print("\n" + header)
    print("-" * len(header))

    rows = []
    for matcher_name in args.matchers:
        for method in args.methods:
            vals = results[(method, matcher_name)]
            if not vals["precision"]:
                continue
            stats = {m: (float(np.mean(vals[m])), float(np.std(vals[m]))) for m in METRICS}
            rows.append({"matcher": matcher_name, "method": method,
                         **{f"{m}_{s}": stats[m][i] for m in METRICS
                            for i, s in enumerate(("mean", "std"))}})
            print(f"{matcher_name.upper():<8}{method:<12}"
                  f"{stats['precision'][0]*100:>9.2f} ± {stats['precision'][1]*100:<5.2f}"
                  f"{stats['matching_score'][0]*100:>11.2f} ± {stats['matching_score'][1]*100:<5.2f}"
                  f"{stats['num_matches'][0]:>9.2f} ± {stats['num_matches'][1]:<5.2f}"
                  f"{stats['entropy'][0]:>8.2f} ± {stats['entropy'][1]:<5.2f}"
                  f"{stats['sb_precision'][0]:>8.2f} ± {stats['sb_precision'][1]:<5.2f}")

    os.makedirs(args.output, exist_ok=True)

    csv_path = os.path.join(args.output, "table1.csv")
    with open(csv_path, "w") as fh:
        fh.write("matcher,method," + ",".join(f"{m}_mean,{m}_std" for m in METRICS) + "\n")
        for r in rows:
            fh.write(f"{r['matcher']},{r['method']},"
                     + ",".join(f"{r[f'{m}_mean']:.6f},{r[f'{m}_std']:.6f}" for m in METRICS) + "\n")

    md_path = os.path.join(args.output, "table1.md")
    with open(md_path, "w") as fh:
        fh.write("| Matcher | Method | Prec. (%) | Match Sc. (%) | # Matches | Entropy | SBP |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['matcher'].upper()} | {r['method']} "
                     f"| {r['precision_mean']*100:.2f} ± {r['precision_std']*100:.2f} "
                     f"| {r['matching_score_mean']*100:.2f} ± {r['matching_score_std']*100:.2f} "
                     f"| {r['num_matches_mean']:.2f} ± {r['num_matches_std']:.2f} "
                     f"| {r['entropy_mean']:.2f} ± {r['entropy_std']:.2f} "
                     f"| {r['sb_precision_mean']:.2f} ± {r['sb_precision_std']:.2f} |\n")

    json_path = os.path.join(args.output, "table1.json")
    with open(json_path, "w") as fh:
        json.dump({"config": vars(args), "rows": rows}, fh, indent=2)

    print(f"\nwrote {csv_path}, {md_path}, {json_path}")


if __name__ == "__main__":
    main()
