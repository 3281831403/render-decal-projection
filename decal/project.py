"""Decal projection contracts.

The data model is:

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

_SINGULAR_TOL = 1e-12


class DecalError(Exception):
    """Raised when a decal, fragment, matrix, camera, or batch is malformed."""


# ---------------------------------------------------------------------------
# Validation / math helpers
# ---------------------------------------------------------------------------

def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_vec3(value, name):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or not all(_is_finite_number(component) for component in value)
    ):
        raise DecalError(f"{name} must be a 3-vector of finite numbers")
    return [float(component) for component in value]


def _require_matrix(matrix):
    if (
        not isinstance(matrix, (list, tuple))
        or len(matrix) != 16
        or not all(_is_finite_number(component) for component in matrix)
    ):
        raise DecalError("matrix must be 16 finite numbers")
    return [float(component) for component in matrix]


def _require_fragment(fragment):
    if not isinstance(fragment, dict):
        raise DecalError("fragment must be a dictionary")
    if "pos" not in fragment or "normal" not in fragment:
        raise DecalError("fragment must contain 'pos' and 'normal'")
    pos = _require_vec3(fragment["pos"], "fragment pos")
    _require_vec3(fragment["normal"], "fragment normal")
    return pos


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(_dot(a, a))


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _apply(matrix, point):
    """Apply a flat 4x4 row-major matrix to a 3D point (w = 1)."""
    x, y, z = point
    return [
        matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3],
        matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7],
        matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11],
    ]


def _invert_4x4(matrix):
    """Invert a flat 4x4 row-major matrix via Gauss-Jordan elimination.

    Raises ``DecalError`` when the matrix is singular.
    """
    rows = [
        [matrix[r * 4 + c] for c in range(4)]
        + [1.0 if r == c else 0.0 for c in range(4)]
        for r in range(4)
    ]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) <= _SINGULAR_TOL:
            raise DecalError("matrix is not invertible")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        pivot_value = rows[col][col]
        rows[col] = [value / pivot_value for value in rows[col]]
        for r in range(4):
            if r != col and rows[r][col] != 0.0:
                factor = rows[r][col]
                rows[r] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(rows[r], rows[col])
                ]
    return [rows[r][4 + c] for r in range(4) for c in range(4)]


def _clamp(value, low, high):
    return min(max(value, low), high)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_matrix(
    origin: list[float],
    normal: list[float],
    tangent: list[float],
    size: float,
) -> list[float]:
    """Build a 4x4 row-major world-to-decal projection matrix.

    The tangent is Gram-Schmidt orthogonalized against the normalized normal,
    so the basis is orthonormal and the decal is never sheared.  A zero
    normal, a tangent parallel to the normal, a non-positive or non-finite
    ``size``, or any non-finite input raises ``DecalError``.
    """
    origin = _require_vec3(origin, "origin")
    normal = _require_vec3(normal, "normal")
    tangent = _require_vec3(tangent, "tangent")
    if not _is_finite_number(size) or size <= 0.0:
        raise DecalError("size must be finite and positive")

    normal_length = _norm(normal)
    if normal_length <= EPS:
        raise DecalError("normal is zero")
    w = [component / normal_length for component in normal]

    projection = _dot(tangent, w)
    orthogonalized = [tangent[i] - projection * w[i] for i in range(3)]
    tangent_length = _norm(orthogonalized)
    if tangent_length <= EPS:
        raise DecalError("tangent is parallel to normal")
    u = [component / tangent_length for component in orthogonalized]
    v = _cross(w, u)

    return [
        u[0] / size, u[1] / size, u[2] / size, -_dot(u, origin) / size,
        v[0] / size, v[1] / size, v[2] / size, -_dot(v, origin) / size,
        w[0] / size, w[1] / size, w[2] / size, -_dot(w, origin) / size,
        0.0, 0.0, 0.0, 1.0,
    ]


def clip_to_box(fragments: list[dict], matrix: list[float]) -> list[dict]:
    """Project point fragments into the unit decal cube and cull outside ones.

    The test is against the cube ``[-0.5, 0.5] ** 3`` itself (with ``1e-6``
    tolerance), not its bounding sphere, so points outside the box are always
    discarded.  Returns new dictionaries; inputs are not mutated.
    """
    matrix = _require_matrix(matrix)
    _invert_4x4(matrix)  # a non-invertible projection matrix is malformed
    if not isinstance(fragments, list):
        raise DecalError("fragments must be a list")

    kept = []
    for fragment in fragments:
        pos = _require_fragment(fragment)
        q = _apply(matrix, pos)
        if all(-0.5 - EPS <= component <= 0.5 + EPS for component in q):
            projected = dict(fragment)
            projected["pos"] = q
            kept.append(projected)
    return kept


def depth_bias(fragments: list[dict], normal: list[float], bias: float) -> list[dict]:
    """Offset every fragment along the supplied normal to avoid z-fighting.

    The offset follows the (normalized) normal, never a view direction.
    Returns new dictionaries; inputs are not mutated.  ``bias`` may be
    negative.  A zero-length normal raises ``DecalError``.
    """
    normal = _require_vec3(normal, "normal")
    if not _is_finite_number(bias):
        raise DecalError("bias must be a finite number")
    if not isinstance(fragments, list):
        raise DecalError("fragments must be a list")

    length = _norm(normal)
    if length <= EPS:
        raise DecalError("normal is zero")
    unit = [component / length for component in normal]

    offset = []
    for fragment in fragments:
        pos = _require_fragment(fragment)
        moved = dict(fragment)
        moved["pos"] = [pos[i] + unit[i] * bias for i in range(3)]
        offset.append(moved)
    return offset


def fade_falloff(fragment: dict, edge_fade: float) -> float:
    """Return the edge-fade alpha in ``[0, 1]`` for a local fragment.

    The result is clamped to ``[0, 1]``: past the cube edge the alpha stays
    ``0.0`` instead of going negative and reversing the ramp.
    """
    pos = _require_fragment(fragment)
    if not _is_finite_number(edge_fade):
        raise DecalError("edge_fade must be a finite number")

    f = _clamp(edge_fade, 0.0, 1.0)
    if f <= EPS:
        return 1.0
    d = min(0.5 - abs(pos[0]), 0.5 - abs(pos[1]))
    return _clamp(d / (0.5 * f), 0.0, 1.0)


def blend_decal(
    base: tuple[float, float, float, float],
    decal: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    """Blend a premultiplied decal over a premultiplied base.

    The decal RGB is already premultiplied, so the coverage weight ``a`` (the
    clamped ``alpha``) is applied exactly once; the decal's own alpha is not
    multiplied in again.
    """
    for name, color in (("base", base), ("decal", decal)):
        if (
            not isinstance(color, (list, tuple))
            or len(color) != 4
            or not all(_is_finite_number(component) for component in color)
        ):
            raise DecalError(f"{name} must be an RGBA tuple of finite numbers")
    if not _is_finite_number(alpha):
        raise DecalError("alpha must be a finite number")

    a = _clamp(alpha, 0.0, 1.0)
    return tuple(
        base[i] * (1.0 - a) + decal[i] * a for i in range(4)
    )


def sort_decals(decals: list[dict], camera: list[float]) -> list[dict]:
    """Sort decals by distance from ``camera``, keeping ties stable.

    The decal center is the affine inverse image of the local origin under
    the decal's world-to-local matrix.  Python's ``sorted`` is stable, so
    equal-distance decals retain their relative input order.  Returns a new
    list; the input is not mutated.
    """
    camera = _require_vec3(camera, "camera")
    if not isinstance(decals, list):
        raise DecalError("decals must be a list")

    def distance(decal):
        if not isinstance(decal, dict):
            raise DecalError("decal must be a dictionary")
        matrix = _require_matrix(decal.get("matrix"))
        center = _apply(_invert_4x4(matrix), (0.0, 0.0, 0.0))
        return _norm([center[i] - camera[i] for i in range(3)])

    return sorted(decals, key=distance)


def validate_decal(decal: dict, max_size: float) -> list[str]:
    """Return the problems found in a decal, or ``[]`` when it is valid.

    Problems are reported in a fixed check order and never raise; only a
    malformed ``max_size`` raises ``DecalError``.
    """
    if not _is_finite_number(max_size) or max_size < 0.0:
        raise DecalError("max_size must be finite and non-negative")
    if not isinstance(decal, dict):
        raise DecalError("decal must be a dictionary")

    issues = []

    matrix = decal.get("matrix")
    matrix_ok = (
        isinstance(matrix, (list, tuple))
        and len(matrix) == 16
        and all(_is_finite_number(component) for component in matrix)
    )
    if not matrix_ok:
        issues.append("matrix must be 16 finite numbers")

    size = decal.get("size")
    if not _is_finite_number(size) or size <= 0.0:
        issues.append("size must be positive")
    elif size > max_size:
        issues.append("size exceeds max_size")

    if matrix_ok:
        rows = [matrix[0:3], matrix[4:7], matrix[8:11]]
        lengths = [_norm(row) for row in rows]

        if lengths[2] <= EPS:
            issues.append("normal is zero")

        normalized = [
            [component / length for component in row] if length > EPS else None
            for row, length in zip(rows, lengths)
        ]
        orthogonal = all(
            normalized[i] is None
            or normalized[j] is None
            or abs(_dot(normalized[i], normalized[j])) <= EPS
            for i in range(3)
            for j in range(i + 1, 3)
        )
        if not orthogonal:
            issues.append("basis is not orthogonal")

        if max(lengths) - min(lengths) > EPS:
            issues.append("basis is not uniformly scaled")

    return issues


def project_batch(decals: list[dict], fragments: list[dict], config: dict) -> dict:
    """Project a batch of decals atomically and without cross-contamination.

    Every decal is validated first; any problem rolls the whole batch back to
    ``{"ok": False, ...}`` with per-decal errors and no partial output.  On
    success each decal is clipped with its **own** matrix, offset along the
    local decal normal ``(0, 0, 1)``, and given a fade alpha.
    """
    if not isinstance(decals, list):
        raise DecalError("decals must be a list")
    if not isinstance(fragments, list):
        raise DecalError("fragments must be a list")
    if not isinstance(config, dict):
        raise DecalError("config must be a dictionary")

    max_size = config.get("max_size")
    if not _is_finite_number(max_size) or max_size < 0.0:
        raise DecalError("config max_size must be finite and non-negative")
    edge_fade = config.get("edge_fade", 0.0)
    bias = config.get("bias", 0.0)
    if not _is_finite_number(edge_fade):
        raise DecalError("config edge_fade must be a finite number")
    if not _is_finite_number(bias):
        raise DecalError("config bias must be a finite number")

    errors = {}
    for index, decal in enumerate(decals):
        if not isinstance(decal, dict):
            raise DecalError("each decal must be a dictionary")
        issues = validate_decal(decal, max_size)
        if issues:
            errors[index] = issues
    if errors:
        return {"ok": False, "decals": [], "fragment_count": 0, "errors": errors}

    projected_decals = []
    fragment_count = 0
    for decal in decals:
        clipped = clip_to_box(fragments, decal["matrix"])
        biased = depth_bias(clipped, (0.0, 0.0, 1.0), bias)
        projected_fragments = []
        for fragment in biased:
            shaded = dict(fragment)
            shaded["alpha"] = fade_falloff(fragment, edge_fade)
            projected_fragments.append(shaded)
        fragment_count += len(projected_fragments)
        projected_decals.append(
            {
                "matrix": list(decal["matrix"]),
                "size": decal["size"],
                "fragments": projected_fragments,
            }
        )

    return {"ok": True, "decals": projected_decals, "fragment_count": fragment_count}
