"""Red tests for the decal projector.

Every test intentionally calls one of the unimplemented stubs in
``decal.project``.  The initial suite must therefore report exactly
``15 failed`` with no collection or import errors.
"""

import math

import pytest

from decal.project import (
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


def _identity():
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(_dot(a, a))


# Defect 1: the tangent is used raw, so the decal basis is not orthogonal and
# the projected texture is sheared.
def test_defect_01_build_matrix_orthonormalizes_basis():
    matrix = build_matrix(
        (2.0, 3.0, 4.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0),
        2.0,
    )

    assert len(matrix) == 16
    u = matrix[0:3]
    v = matrix[4:7]
    w = matrix[8:11]

    for first, second in ((u, v), (u, w), (v, w)):
        assert abs(_dot(first, second)) <= 1e-6
    for axis in (u, v, w):
        assert abs(_norm(axis) - 0.5) <= 1e-6

    # The origin must project to the local origin.
    for row, translation in ((u, matrix[3]), (v, matrix[7]), (w, matrix[11])):
        assert abs(_dot(row, (2.0, 3.0, 4.0)) + translation) <= 1e-6


# Defect 2: a zero normal is accepted, producing a degenerate matrix.
def test_defect_02_build_matrix_rejects_zero_normal():
    with pytest.raises(DecalError):
        build_matrix((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0)


# Defect 3: clipping tests the bounding sphere instead of the cube, so points
# just outside the box survive.
def test_defect_03_clip_to_box_rejects_points_outside_cube_but_inside_bounding_sphere():
    outside = (0.6, 0.6, 0.0)
    # This point is outside the cube but well inside its bounding sphere.
    assert math.sqrt(0.6 ** 2 + 0.6 ** 2) < math.sqrt(3.0) / 2.0

    fragments = [
        {"pos": outside, "normal": (0.0, 0.0, 1.0)},
        {"pos": (0.2, -0.2, 0.0), "normal": (0.0, 0.0, 1.0)},
    ]

    result = clip_to_box(fragments, _identity())

    assert len(result) == 1
    assert result[0]["pos"] == pytest.approx((0.2, -0.2, 0.0), abs=1e-6)


# Defect 4: the depth offset follows the view direction instead of the decal
# normal, causing color bleeding on slopes.
def test_defect_04_depth_bias_uses_supplied_normal():
    fragments = [{"pos": (1.0, 2.0, 3.0), "normal": (0.0, 0.0, 1.0)}]

    result = depth_bias(fragments, (0.0, 2.0, 0.0), 0.25)

    assert result[0]["pos"] == pytest.approx((1.0, 2.25, 3.0), abs=1e-6)
    assert fragments[0]["pos"] == (1.0, 2.0, 3.0)


# Defect 5: edge fade is not clamped, so it becomes negative past the edge and
# the alpha ramp reverses.
def test_defect_05_fade_falloff_clamps_alpha():
    center = {"pos": (0.0, 0.0, 0.0), "normal": (0.0, 0.0, 1.0)}
    edge = {"pos": (0.5, 0.0, 0.0), "normal": (0.0, 0.0, 1.0)}
    outside = {"pos": (10.0, 0.0, 0.0), "normal": (0.0, 0.0, 1.0)}

    assert fade_falloff(center, 1.0) == pytest.approx(1.0, abs=1e-6)
    assert fade_falloff(edge, 1.0) == pytest.approx(0.0, abs=1e-6)
    assert fade_falloff(outside, 1.0) == pytest.approx(0.0, abs=1e-6)


# Defect 6: alpha is applied twice, darkening the decal's soft edges.
def test_defect_06_blend_decal_does_not_double_apply_alpha():
    base = (0.0, 0.0, 0.0, 1.0)
    decal = (1.0, 0.0, 0.0, 1.0)

    result = blend_decal(base, decal, 0.5)

    assert result == pytest.approx((0.5, 0.0, 0.0, 1.0), abs=1e-6)


# Defect 7: an unstable sort reorders equal-distance decals, making them
# flicker frame to frame.
def test_defect_07_sort_decals_is_stable_for_equal_distance():
    decals = [
        {"matrix": _identity(), "size": 3.0},
        {"matrix": _identity(), "size": 1.0},
        {"matrix": _identity(), "size": 2.0},
    ]

    result = sort_decals(decals, (0.0, 0.0, 5.0))

    assert [decal["size"] for decal in result] == [3.0, 1.0, 2.0]


# Defect 8: an oversized decal is accepted and its texture is stretched.
def test_defect_08_validate_decal_reports_oversized_decal():
    scaled = [
        0.25, 0.0, 0.0, 0.0,
        0.0, 0.25, 0.0, 0.0,
        0.0, 0.0, 0.25, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]

    issues = validate_decal({"matrix": scaled, "size": 4.0}, 2.0)

    assert any("size" in issue.lower() for issue in issues)


# Defect 9: the batch reuses one matrix for every decal, so projections leak
# from one decal into the next.
def test_defect_09_project_batch_uses_each_decals_own_matrix():
    translated = [
        1.0, 0.0, 0.0, 0.25,
        0.0, 1.0, 0.0, 0.25,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
    decals = [
        {"matrix": _identity(), "size": 1.0},
        {"matrix": translated, "size": 1.0},
    ]
    fragments = [{"pos": (0.0, 0.0, 0.0), "normal": (0.0, 0.0, 1.0)}]

    result = project_batch(decals, fragments, {"max_size": 2.0})

    assert result["ok"] is True
    first = result["decals"][0]["fragments"][0]["pos"]
    second = result["decals"][1]["fragments"][0]["pos"]
    assert first == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert second == pytest.approx((0.25, 0.25, 0.0), abs=1e-6)


# Defect 10: size=0 is not validated, later causing a divide-by-zero.
def test_defect_10_validate_decal_rejects_zero_size():
    issues = validate_decal({"matrix": _identity(), "size": 0.0}, 1.0)

    assert any("size" in issue.lower() for issue in issues)


# Combination: with size 1 and a canonical frame the projection matrix is the
# identity, so local fragments line up with the projector.
def test_combo_identity_matrix_aligns_decal_with_fragment():
    matrix = build_matrix((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 1.0)
    fragment = {"pos": (0.25, -0.25, 0.0), "normal": (0.0, 0.0, 1.0)}

    assert matrix == pytest.approx(_identity(), abs=1e-6)
    result = clip_to_box([fragment], matrix)
    assert len(result) == 1
    assert result[0]["pos"] == pytest.approx(fragment["pos"], abs=1e-6)


# Combination: zero coverage leaves the base color untouched.
def test_combo_alpha_zero_returns_base_color():
    base = (0.2, 0.4, 0.6, 1.0)
    decal = (1.0, 1.0, 1.0, 1.0)

    assert blend_decal(base, decal, 0.0) == pytest.approx(base, abs=1e-6)


# Combination: every fragment outside the cube produces an empty result.
def test_combo_all_fragments_outside_box_returns_empty():
    fragments = [
        {"pos": (2.0, 0.0, 0.0), "normal": (0.0, 0.0, 1.0)},
        {"pos": (0.0, -1.0, 1.0), "normal": (0.0, 0.0, 1.0)},
    ]

    assert clip_to_box(fragments, _identity()) == []


# Combination: a zero size is rejected before it can divide by zero.
def test_combo_zero_size_raises_decal_error():
    with pytest.raises(DecalError):
        build_matrix((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 0.0)


# Combination: a fully valid batch keeps the caller's decal order.
def test_combo_project_batch_preserves_order_on_success():
    decals = [
        {"matrix": _identity(), "size": 1.0},
        {"matrix": _identity(), "size": 2.0},
        {"matrix": _identity(), "size": 3.0},
    ]
    fragments = [{"pos": (0.0, 0.0, 0.0), "normal": (0.0, 0.0, 1.0)}]

    result = project_batch(decals, fragments, {"max_size": 5.0})

    assert result["ok"] is True
    assert [decal["size"] for decal in result["decals"]] == [1.0, 2.0, 3.0]
    assert result["fragment_count"] == 3
