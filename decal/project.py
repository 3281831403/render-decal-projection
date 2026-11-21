"""Decal projection contracts.

The module intentionally contains only stubs.  The data model is:

* A decal is ``{"matrix": list[float], "size": float}``.  ``matrix`` is a flat
  4x4 row-major world-to-decal matrix and ``size`` is the world-space side
  length of the square decal.
* A fragment / projector is ``{"pos": [x, y, z], "normal": [x, y, z]}``.
  Extra keys are preserved where the function returns fragments.
* The decal's local projection volume is the unit cube
  ``[-0.5, 0.5] ** 3``.  The world-space decal square has side ``size``.

All geometric comparisons use an absolute tolerance of ``1e-6``.
"""

import math

EPS = 1e-6


class DecalError(Exception):
    """Raised when a decal, fragment, matrix, camera, or batch is malformed."""


def build_matrix(
    origin: list[float],
    normal: list[float],
    tangent: list[float],
    size: float,
) -> list[float]:
    """Build a 4x4 row-major world-to-decal projection matrix.

    The local projection volume is the unit cube ``[-0.5, 0.5] ** 3``.  Let
    ``w = normalize(normal)`` and
    ``u = normalize(tangent - dot(tangent, w) * w)``.  The tangent is therefore
    Gram-Schmidt orthogonalized against the normal instead of being used
    raw; a tangent parallel to the normal raises ``DecalError``.  The third
    basis vector is ``v = cross(w, u)``.

    The returned matrix has rows ``u / size``, ``v / size`` and ``w / size``,
    translation column
    ``(-dot(u, origin) / size, -dot(v, origin) / size,
    -dot(w, origin) / size)``, and last row ``[0, 0, 0, 1]``.  With column
    vectors, ``q = M * [p, 1]``.  Consequently ``M * [origin, 1]`` is the
    local origin and points ``size / 2`` from ``origin`` along the basis map
    to ``+/-0.5`` on the corresponding cube face.

    ``size`` must be finite and positive.  A zero-length normal, a tangent
    that cannot be orthogonalized, or non-finite input raises ``DecalError``.
    Tolerance: ``1e-6``.
    """

    raise NotImplementedError("build_matrix is not implemented")


def clip_to_box(fragments: list[dict], matrix: list[float]) -> list[dict]:
    """Project point fragments into the unit decal cube and cull outside ones.

    Each fragment is a dictionary containing at least ``"pos"`` and
    ``"normal"`` 3-vectors; additional keys are allowed.  For every fragment
    compute ``q = matrix * [pos, 1]``.  Keep the fragment only when all three
    components of ``q`` lie in ``[-0.5 - 1e-6, 0.5 + 1e-6]``.  A point that is
    outside the cube is discarded even if it falls inside the cube's
    bounding sphere.

    Return new dictionaries in input order with ``"pos"`` replaced by the
    local ``q`` and every other key shallow-copied.  The input fragments are
    not mutated.  An empty input returns ``[]``.  A malformed fragment or a
    malformed/non-invertible matrix raises ``DecalError``.  Tolerance:
    ``1e-6``.
    """

    raise NotImplementedError("clip_to_box is not implemented")


def depth_bias(fragments: list[dict], normal: list[float], bias: float) -> list[dict]:
    """Offset every fragment along a normal to avoid z-fighting.

    Normalize ``normal`` and replace each fragment's ``"pos"`` with
    ``pos + unit_normal * bias``.  The offset is along the supplied normal,
    never along a view direction, and applies to every fragment in order.
    Return new dictionaries with the other keys shallow-copied; inputs are
    not mutated.  A zero-length normal or malformed fragment raises
    ``DecalError``.  ``bias`` may be negative.  Tolerance: ``1e-6``.
    """

    raise NotImplementedError("depth_bias is not implemented")


def fade_falloff(fragment: dict, edge_fade: float) -> float:
    """Return the edge-fade alpha in ``[0, 1]`` for a local fragment.

    ``edge_fade`` is a fraction of the half-width and is clamped to
    ``[0, 1]``.  Let ``f`` be that clamped value.  When ``f <= 1e-6`` the
    fragment is fully opaque regardless of position and the result is
    ``1.0``.  Otherwise let
    ``d = min(0.5 - abs(fragment["pos"][0]), 0.5 - abs(fragment["pos"][1]))``
    and return ``clamp(d / (0.5 * f), 0.0, 1.0)``.  The center is ``1.0``, the
    cube edge is ``0.0``, and points beyond the edge stay at ``0.0`` rather
    than becoming negative and wrapping/reversing the alpha.  Only the local
    x/y position matters.  Tolerance: ``1e-6``.
    """

    raise NotImplementedError("fade_falloff is not implemented")


def blend_decal(
    base: tuple[float, float, float, float],
    decal: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    """Blend a premultiplied decal over a premultiplied base.

    Both ``base`` and ``decal`` are RGBA tuples whose RGB components are
    already premultiplied by their own alpha.  Clamp the coverage ``alpha`` to
    ``[0, 1]`` and call it ``a``.  The source-over result is::

        out_rgb = base_rgb * (1 - a) + decal_rgb * a
        out_a   = base_a   * (1 - a) + decal_a   * a

    The decal RGB is already premultiplied, so its own alpha must not be
    applied to it a second time; only the coverage weight ``a`` is applied
    once.  ``alpha=0`` returns ``base`` exactly.  Return a new 4-tuple and do
    not mutate the inputs.  Tolerance: ``1e-6``.
    """

    raise NotImplementedError("blend_decal is not implemented")


def sort_decals(decals: list[dict], camera: list[float]) -> list[dict]:
    """Sort decals by distance from ``camera`` and keep ties stable.

    ``camera`` is a world-space 3-vector.  A decal is
    ``{"matrix": list[float], "size": float}`` where ``matrix`` is the
    world-to-local transform from :func:`build_matrix`.  The decal center is
    the world point that the matrix maps to the local origin, i.e. the
    affine inverse image of ``[0, 0, 0, 1]``.  Sort ascending by Euclidean
    distance from ``camera`` to that center.

    Decals with equal distance retain their relative input order, so the sort
    must be stable; no tie-breaker such as a hash, set iteration order, or
    arbitrary identity is allowed.  Return a new list; input order and decal
    dictionaries are not mutated.  A malformed camera or decal matrix raises
    ``DecalError``.  Tolerance: ``1e-6`` for matrix inversion.
    """

    raise NotImplementedError("sort_decals is not implemented")


def validate_decal(decal: dict, max_size: float) -> list[str]:
    """Return the problems found in a decal, or ``[]`` when it is valid.

    ``decal`` must contain a flat 4x4 row-major ``"matrix"`` of finite
    numbers and a finite ``"size"``.  The checks, reported with these exact
    problem strings, are:

    * ``"matrix must be 16 finite numbers"``;
    * ``"size must be positive"`` for zero, negative, or non-finite size;
    * ``"size exceeds max_size"`` when ``size > max_size``;
    * ``"normal is zero"`` when the third basis row (the decal normal) has
      length at most ``1e-6``;
    * ``"basis is not orthogonal"`` when two normalized basis rows have a dot
      product whose absolute value exceeds ``1e-6``;
    * ``"basis is not uniformly scaled"`` when the three basis-row lengths
      differ by more than ``1e-6``.

    Return a list in that check order.  A malformed ``max_size`` (not finite
    or negative) raises ``DecalError``; decal problems themselves are always
    reported in the returned list and do not raise.  Tolerance: ``1e-6``.
    """

    raise NotImplementedError("validate_decal is not implemented")


def project_batch(decals: list[dict], fragments: list[dict], config: dict) -> dict:
    """Project a batch of decals atomically and without cross-contamination.

    ``decals`` is a list of ``{"matrix": ..., "size": ...}`` dictionaries and
    ``fragments`` is a shared list of ``{"pos": ..., "normal": ...}`` point
    fragments.  ``config`` contains ``"max_size"`` and may contain
    ``"edge_fade"`` (default ``0.0``) and ``"bias"`` (default ``0.0``).

    Validate every decal before producing any output.  If any decal has a
    problem, return
    ``{"ok": False, "decals": [], "fragment_count": 0,
    "errors": {index: issues}}`` with no partial work; this is the all-or-
    nothing rollback.  Otherwise process decals in input order, and each
    decal must use its **own** matrix and size.  For each decal clip the
    shared fragments into its local cube, offset them along the local decal
    normal ``(0, 0, 1)`` by ``config["bias"]``, and attach
    ``"alpha" = fade_falloff(fragment, edge_fade)``.  Fragment input order is
    preserved.

    On success return::

        {"ok": True,
         "decals": [{"matrix": ..., "size": ...,
                     "fragments": [{"pos": ..., "normal": ..., "alpha": ...}]}],
         "fragment_count": <total surviving fragments>}

    The result preserves decal input order.  Empty input succeeds with no
    decals and a zero count.  Malformed config or batch structure raises
    ``DecalError``.  Tolerance: ``1e-6``.
    """

    raise NotImplementedError("project_batch is not implemented")
