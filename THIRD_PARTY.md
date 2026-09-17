# Third-Party Code and Licenses

Stretcher itself (everything in `src/`, the notebooks, and `scripts/`) is released
under the MIT License — see [LICENSE](LICENSE).

This repository also **vendors** portions of several third-party projects so that the
published results can be reproduced against the exact code they were produced with.
Those files are **not** covered by the MIT License above. Each retains the license of
its original project, reproduced below.

---

## LightGlue — `lightglue/`

- **Upstream:** https://github.com/cvg/LightGlue
- **License:** Apache License 2.0 — full text in [`lightglue/LICENSE`](lightglue/LICENSE)
- **Copyright:** ETH Zurich Computer Vision and Geometry Group
- **Reference:** Lindenberger, Sarlin and Pollefeys, *LightGlue: Local Feature Matching
  at Light Speed*, ICCV 2023.

Vendored files: `__init__.py`, `aliked.py`, `disk.py`, `lightglue.py`, `superpoint.py`,
`utils.py`, `viz2d.py`.

### ⚠️ SuperPoint — `lightglue/superpoint.py`

This file requires separate attention. Although it is distributed as part of the
Apache-2.0 licensed LightGlue repository, it **retains Magic Leap, Inc.'s original
copyright banner**, which is more restrictive than Apache-2.0.

SuperPoint was released by Magic Leap for **non-commercial research use only**
(see https://github.com/magicleap/SuperPointPretrainedNetwork). If you intend to use
this repository for anything beyond academic research, consult Magic Leap's terms
directly. The MIT License on Stretcher does not, and cannot, grant you rights to
SuperPoint.

The same restriction applies to the pretrained SuperPoint weights that LightGlue
downloads at runtime.

- **Reference:** DeTone, Malisiewicz and Rabinovich, *SuperPoint: Self-Supervised
  Interest Point Detection and Description*, CVPRW 2018.

---

## DeDoDe — `DeDoDe/`

- **Upstream:** https://github.com/Parskatt/DeDoDe
- **License:** MIT — full text in [`DeDoDe/LICENSE`](DeDoDe/LICENSE)
- **Copyright:** (c) 2023 Johan Edstedt
- **Reference:** Edstedt, Bökman, Wadenbäck and Felsberg, *DeDoDe: Detect, Don't
  Describe — Describe, Don't Detect for Local Feature Matching*, 3DV 2024.

Vendored files: `utils.py`, `matchers/__init__.py`, `matchers/dual_softmax_matcher.py`.

Only the dual-softmax matcher and a small set of coordinate utilities are used.
Stretcher's own matcher, `src/dsm_matching.py`, is derived from
`DeDoDe/matchers/dual_softmax_matcher.py` and extends it to select the best-scoring
match across a set of deformation hypotheses.

---

## Other dependencies

The following are ordinary runtime dependencies, installed via `environment.yml`
rather than vendored. Their licenses apply as distributed:

| Package | License |
|---|---|
| PyTorch | BSD-3-Clause |
| FEniCS (2019.1.0) | LGPL-3.0-or-later |
| PyVista | MIT |
| OpenCV (`opencv-python`) | Apache-2.0 |
| scikit-image | BSD-3-Clause |
| Kornia | Apache-2.0 |

---

## Datasets

**Cholec80 / CholecSeg8k** — used to train the descriptor transformation network.
Not redistributed here. It must be obtained from its maintainers under their own
terms: http://camma.u-strasbg.fr/datasets

**Pig liver images** (`data/medical_deformed/`) — acquired by the authors and
released under the MIT License together with this repository.
