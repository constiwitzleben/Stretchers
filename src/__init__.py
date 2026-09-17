"""Stretcher: deformation-robust keypoint descriptors.

Modules
-------
affine_transformations  local affine strain model and image warping (Sec. 2.2.1)
models                  the descriptor transformation network (Sec. 2.2.2)
descriptors             SuperPoint detection and dense descriptor sampling
dsm_matching            dual-softmax matching over deformation hypotheses (Sec. 2.1)
fenics_deformation      FEM synthetic deformation and ground-truth tracking
matching_util           drawing helpers and the strain-aware metrics
notebook_utils          high-level steps used by the four notebooks
"""
