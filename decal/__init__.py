"""Decal projection package."""

from .project import (
    DecalError,
    blend_decal,
    build_matrix,
    clip_to_box,
    depth_bias,
    fade_falloff,
    project_batch,
    sort_decals,
    validate_decal,
)

__all__ = [
    "DecalError",
    "blend_decal",
    "build_matrix",
    "clip_to_box",
    "depth_bias",
    "fade_falloff",
    "project_batch",
    "sort_decals",
    "validate_decal",
]
