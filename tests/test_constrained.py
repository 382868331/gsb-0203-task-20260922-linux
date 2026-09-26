"""受约束包围球（球心限定在直线/平面）的必要测试。

参考实现独立编写：直接在三维中把中心参数化为 c = o + B u，用另一套
Cramer 行列式求解等值方程，不 import、不调用 miniball3d 的任何求解
函数；只在测试里逐字段对照。另对每个结果逐点检查覆盖、并核验活动点
梯度沿子空间方向的凸组合为零（精确 Fraction）。
"""

from fractions import Fraction
from itertools import combinations
import random
import unittest

import miniball3d
from miniball3d import (
    NoFeasibleCandidateError,
    SearchBudgetExhaustedError,
    minimum_enclosing_sphere,
    minimum_enclosing_sphere_constrained,
)

F = Fraction


# ---------------------------------------------------------------------------
# 独立参考实现：禁止调用 miniball3d 中的任何求解函数
# ---------------------------------------------------------------------------

def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _det2(m):
    return m[0][0] * m[1][1] - m[0][1] * m[1][0]


def _solve2(a, b):
    d = _det2(a)
    if d == 0:
        return None
    return (
        (b[0] * a[1][1] - b[1] * a[0][1]) / d,
        (a[0][0] * b[1] - a[1][0] * b[0]) / d,
    )


def _reference_constrained(points, origin, directions):
    """独立枚举参考：返回 (center, r2, active_indices) 或 None。

    对每个活动集 S（1..dim+1 个），在三维中直接列等值方程
    |c - p_i|^2 = |c - p_0|^2（i in S），其中 c = o + B u，u 为低维坐标；
    再要求 u 是 S 投影的凸组合（重心系数非负）。投影用解正规方程得到，
    与被测实现的度量 Gram 路径不同。
    """
    dim = len(directions)
    B = directions

    def project(p):
        # 解正规方程 G u = B^T (p - o)，返回 (u, h)
        g = [[_dot(B[a], B[b]) for b in range(dim)] for a in range(dim)]
        rhs = [_dot(B[a], [p[d] - origin[d] for d in range(3)]) for a in range(dim)]
        if dim == 1:
            u = (rhs[0] / g[0][0],)
        else:
            u = _solve2(g, rhs)
        pu = [origin[d] + sum(u[j] * B[j][d] for j in range(dim)) for d in range(3)]
        h = sum((p[d] - pu[d]) ** 2 for d in range(3))
        return u, h

    proj = [project(p) for p in points]
    m = len(points)
    best = None
    for k in range(1, min(dim + 1, m) + 1):
        for combo in combinations(range(m), k):
            u0, h0 = proj[combo[0]]
            if k == 1:
                u, weights = u0, [F(1)]
            else:
                # 等值方程：2 (u_i - u_0)^T G u = |u_i|_G^2 - |u_0|_G^2 + h_i - h_0
                # 以 u = u0 + sum alpha_j (u_j - u0) 代入，用 Cramer 求解。
                vs = [tuple(proj[c][0][d] - u0[d] for d in range(dim))
                      for c in combo[1:]]
                hs = [proj[c][1] for c in combo[1:]]

                def mg(x, y):
                    return sum(x[a] * _dot(B[a], B[b]) * y[b]
                               for a in range(dim) for b in range(dim))

                amat = [[mg(vs[a], vs[b]) for b in range(k - 1)]
                        for a in range(k - 1)]
                brhs = [(mg(vs[a], vs[a]) + hs[a] - h0) / 2
                        for a in range(k - 1)]
                if k - 1 == 1:
                    if amat[0][0] == 0:
                        continue
                    alpha = [brhs[0] / amat[0][0]]
                else:
                    alpha = _solve2(amat, brhs)
                    if alpha is None:
                        continue
                u = tuple(u0[d] + sum(alpha[j] * vs[j][d] for j in range(k - 1))
                          for d in range(dim))
                weights = [F(1) - sum(alpha)] + list(alpha)
            if any(w < 0 for w in weights):
                continue
            t = sum((u[d] - u0[d]) ** 2 for d in range(dim))  # placeholder
            # 真实目标值：|c - p_0|^2（三维直接算）
            c = [origin[d] + sum(u[j] * B[j][d] for j in range(dim))
                 for d in range(3)]
            t = sum((c[d] - points[combo[0]][d]) ** 2 for d in range(3))
            if all(sum((c[d] - p[d]) ** 2 for d in range(3)) <= t
                   for p in points):
                if best is None or t < best[0]:
                    best = (t, tuple(F(v) for v in c), combo)
    if best is None:
        return None
    return best[1], best[0], best[2]


def _check_certificate(testcase, result, points):
    """逐点覆盖 + 活动点梯度凸组合为零 + 对偶恒等式（全部精确）。"""
    for p in points:
        testcase.assertTrue(result.contains(p), f"point not covered: {p}")
    for s in result.support:
        d2 = sum((result.center[i] - s.point[i]) ** 2 for i in range(3))
        testcase.assertEqual(d2, result.radius_squared)
    testcase.assertEqual(sum(result.weights), F(1))
    testcase.assertTrue(all(w >= 0 for w in result.weights))
    testcase.assertEqual(
        result.certificate_residual(),
        tuple(F(0) for _ in range(result.subspace_dimension)),
    )
    testcase.assertEqual(
        result.dual_identity_radius_squared(), result.radius_squared
    )
    # 球心确实落在约束子空间上：法向距离为 0
    testcase.assertEqual(result.normal_distance_squared(result.center), F(0))


PLANE = dict(subspace_origin=(0, 0, 0),
             subspace_directions=[(1, 0, 0), (0, 1, 0)])
LINE = dict(subspace_origin=(0, 0, 0), subspace_directions=[(1, 0, 0)])


class PlaneConstraintTests(unittest.TestCase):
    def test_two_points_opposite_sides(self):
        # 两个点位于平面两侧：投影重合，最优中心即投影点，半径=法向距离
        pts = [(0, 0, 3), (0, 0, -3)]
        r = minimum_enclosing_sphere_constrained(pts, **PLANE)
        self.assertEqual(r.center, (F(0), F(0), F(0)))
        self.assertEqual(r.radius_squared, F(9))
        # 两点投影与 h 完全相同，合并为一个活动点但保留全部 ID
        self.assertEqual(len(r.support), 1)
        self.assertEqual(sorted(r.support[0].ids), [0, 1])
        _check_certificate(self, r, pts)

    def test_active_points_not_all_in_constraint_plane(self):
        # 活动点不全部落在约束平面内（含平面外点，法向常数项起作用）
        pts = [(0, 0, 1), (2, 0, 0), (0, 2, 0)]
        r = minimum_enclosing_sphere_constrained(pts, **PLANE)
        self.assertEqual(len(r.support), 3)
        self.assertEqual(r.center, (F(3, 4), F(3, 4), F(0)))
        self.assertEqual(r.radius_squared, F(17, 8))
        self.assertEqual(r.weights, (F(1, 4), F(3, 8), F(3, 8)))
        self.assertTrue(any(
            r.normal_distance_squared(s.point) > 0 for s in r.support
        ))
        _check_certificate(self, r, pts)

    def test_not_projection_of_free_center(self):
        # 不能先求自由球心再投影：约束最优一般不同于自由中心的投影
        pts = [(0, 0, 1), (2, 0, 0), (0, 2, 0), (1, 1, 1)]
        free = minimum_enclosing_sphere(pts)
        con = minimum_enclosing_sphere_constrained(pts, **PLANE)
        projected = tuple(free.center[:2]) + (F(0),)
        # 自由球心投影要么不满足约束最优，要么半径更大；此处二者皆成立
        self.assertNotEqual(con.center, projected)
        self.assertGreaterEqual(con.radius_squared, free.radius_squared)
        _check_certificate(self, con, pts)

    def test_oblique_plane(self):
        # 非坐标平面：斜平面上的约束
        pts = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 2, 2)]
        r = minimum_enclosing_sphere_constrained(
            pts, subspace_origin=(0, 0, 0),
            subspace_directions=[(1, 1, 0), (0, 1, 1)],
        )
        _check_certificate(self, r, pts)
        ref = _reference_constrained(pts, (F(0),) * 3,
                                     [(F(1), F(1), F(0)), (F(0), F(1), F(1))])
        self.assertEqual(r.center, ref[0])
        self.assertEqual(r.radius_squared, ref[1])

    def test_degenerate_collinear_projections_flagged(self):
        # 投影共线：三点活动集 Gram 奇异被跳过，必须明确标注
        pts = [(0, 0, 1), (1, 0, -1), (2, 0, 2)]
        r = minimum_enclosing_sphere_constrained(pts, **PLANE)
        self.assertGreater(r.degenerate_combinations, 0)
        _check_certificate(self, r, pts)


class LineConstraintTests(unittest.TestCase):
    def test_optimal_center_on_line(self):
        # 最优中心在直线上：x 轴约束，两点投影中点
        pts = [(0, 1, 0), (2, -1, 0)]
        r = minimum_enclosing_sphere_constrained(pts, **LINE)
        self.assertEqual(r.center, (F(1), F(0), F(0)))
        self.assertEqual(r.radius_squared, F(2))
        self.assertEqual(r.weights, (F(1, 2), F(1, 2)))
        _check_certificate(self, r, pts)

    def test_line_single_active_point(self):
        # 单活动点：一侧法向距离主导
        pts = [(0, 5, 0), (1, 0, 0)]
        r = minimum_enclosing_sphere_constrained(pts, **LINE)
        self.assertEqual(r.center, (F(0), F(0), F(0)))
        self.assertEqual(r.radius_squared, F(25))
        _check_certificate(self, r, pts)

    def test_shifted_line(self):
        # 不过原点的直线
        pts = [(0, 0, 0), (4, 2, 0), (2, 0, 3)]
        r = minimum_enclosing_sphere_constrained(
            pts, subspace_origin=(0, 1, 0), subspace_directions=[(0, 0, 2)]
        )
        _check_certificate(self, r, pts)
        ref = _reference_constrained(pts, (F(0), F(1), F(0)), [(F(0), F(0), F(2))])
        self.assertEqual(r.center, ref[0])
        self.assertEqual(r.radius_squared, ref[1])


class ReferenceComparisonTests(unittest.TestCase):
    def test_random_small_sets_against_reference(self):
        rng = random.Random(20260926)
        for trial in range(40):
            n = rng.randint(1, 7)
            pts = [tuple(F(rng.randint(-4, 4)) for _ in range(3))
                   for _ in range(n)]
            if trial % 2 == 0:
                kw = PLANE
                dirs = [(F(1), F(0), F(0)), (F(0), F(1), F(0))]
            else:
                kw = LINE
                dirs = [(F(1), F(0), F(0))]
            r = minimum_enclosing_sphere_constrained(pts, **kw)
            ref = _reference_constrained(pts, (F(0),) * 3, dirs)
            self.assertEqual(r.radius_squared, ref[1], f"trial {trial}: {pts}")
            self.assertEqual(r.center, ref[0], f"trial {trial}: {pts}")
            _check_certificate(self, r, pts)


class ConstrainedValidationTests(unittest.TestCase):
    def test_dependent_directions_rejected(self):
        with self.assertRaises(ValueError):
            minimum_enclosing_sphere_constrained(
                [(0, 0, 0)], subspace_origin=(0, 0, 0),
                subspace_directions=[(1, 0, 0), (2, 0, 0)])

    def test_duplicate_directions_rejected(self):
        with self.assertRaises(ValueError):
            minimum_enclosing_sphere_constrained(
                [(0, 0, 0)], subspace_origin=(0, 0, 0),
                subspace_directions=[(1, 2, 3), (1, 2, 3)])

    def test_zero_direction_rejected(self):
        with self.assertRaises(ValueError):
            minimum_enclosing_sphere_constrained(
                [(0, 0, 0)], subspace_origin=(0, 0, 0),
                subspace_directions=[(0, 0, 0)])

    def test_wrong_direction_count(self):
        for dirs in ([], [(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
            with self.assertRaises(ValueError):
                minimum_enclosing_sphere_constrained(
                    [(0, 0, 0)], subspace_origin=(0, 0, 0),
                    subspace_directions=dirs)

    def test_bool_and_nan_rejected_in_subspace(self):
        with self.assertRaises(TypeError):
            minimum_enclosing_sphere_constrained(
                [(0, 0, 0)], subspace_origin=(0, 0, 0),
                subspace_directions=[(True, 0, 0)])
        with self.assertRaises(ValueError):
            minimum_enclosing_sphere_constrained(
                [(0, 0, 0)], subspace_origin=(float("nan"), 0, 0),
                subspace_directions=[(1, 0, 0)])

    def test_budget_exhausted(self):
        pts = [(i, 0, 1) for i in range(6)]
        with self.assertRaises(SearchBudgetExhaustedError):
            minimum_enclosing_sphere_constrained(pts, budget=1, **PLANE)

    def test_result_immutable(self):
        r = minimum_enclosing_sphere_constrained(
            [(0, 0, 1), (1, 0, 0)], **PLANE)
        with self.assertRaises(Exception):
            r.center = (F(0),) * 3


if __name__ == "__main__":
    unittest.main()
