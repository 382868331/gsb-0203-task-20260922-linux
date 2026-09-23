"""miniball3d 的必要测试。

参考实现独立编写（叉积 + 行列式 Cramer 法则、按 R^3 直接求中心），
不 import、不调用被测核心的任何函数；被测核心用的是仿射包内的
Gram 方程 + 高斯消元，二者代码路径相互独立。两套实现只在测试里
逐字段对照。对偶恒等式与证书条件做精确（Fraction）核验。
"""

from fractions import Fraction
from itertools import combinations
import math
import random
import unittest

import miniball3d
from miniball3d import (
    MinimumEnclosingSphere,
    NoFeasibleCandidateError,
    SearchBudgetExhaustedError,
    minimum_enclosing_sphere,
)

F = Fraction


# ---------------------------------------------------------------------------
# 独立参考实现：禁止调用 miniball3d 中的任何求解函数
# ---------------------------------------------------------------------------

def _sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def _add(a, b):
    return tuple(a[i] + b[i] for i in range(3))


def _scale(t, a):
    return tuple(t * x for x in a)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _det3(a, b, c):
    return _dot(a, _cross(b, c))


def _det(m):
    n = len(m)
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    out = F(0)
    for j in range(n):
        minor = [row[:j] + row[j + 1:] for row in m[1:]]
        sign = F(1) if j % 2 == 0 else F(-1)
        out += sign * m[0][j] * _det(minor)
    return out


def _cramer(A, b):
    """独立于核心高斯消元的 Cramer 法则；奇异返回 None。"""
    n = len(A)
    d = _det(A)
    if d == 0:
        return None
    xs = []
    for j in range(n):
        mat = [list(A[r][:j]) + [b[r]] + list(A[r][j + 1:]) for r in range(n)]
        xs.append(_det(mat) / d)
    return xs


def _reference_circumsphere(pts):
    """独立求 pts(1..4 个仿射独立点) 的等距中心、r^2 与重心系数。

    与核心不同的路径：
      k=1 直接返回；
      k=2 取中点；
      k=3 用叉积构造平面有理正交基，解 2x2；
      k=4 直接在 R^3 解线性化等距方程 2c.(pi-p0)=|pi|^2-|p0|^2（3x3）。
    仿射相关返回 None。
    """
    k = len(pts)
    p0 = pts[0]
    if k == 1:
        return p0, F(0), (F(1),)
    if k == 2:
        c = _scale(F(1, 2), _add(pts[0], pts[1]))
        return c, _dot(_sub(c, p0), _sub(c, p0)), (F(1, 2), F(1, 2))

    if k == 3:
        v1 = _sub(pts[1], p0)
        v2 = _sub(pts[2], p0)
        normal = _cross(v1, v2)
        if normal == (0, 0, 0):
            return None
        b1 = v1
        b2 = _cross(normal, v1)  # 在平面内且与 b1 正交
        # c = p0 + x b1 + y b2；等距方程 2c.v = |v|^2
        A = [
            [2 * _dot(b1, v1), 2 * _dot(b2, v1)],
            [2 * _dot(b1, v2), 2 * _dot(b2, v2)],
        ]
        rhs = [_dot(v1, v1), _dot(v2, v2)]
        sol = _cramer(A, rhs)
        if sol is None:
            return None
        x, y = sol
        c = _add(p0, _add(_scale(x, b1), _scale(y, b2)))
        # 重心系数：p0 + x b1 + y b2 = p0 + a v1 + beta v2，其中
        # b2 = n×v1，将 y b2 按 {v1,v2} 分解：解 2x2 坐标方程。
        coord = _cramer(
            [
                [_dot(b1, v1), _dot(b1, v2)],
                [_dot(b2, v1), _dot(b2, v2)],
            ],
            [x * _dot(b1, b1), y * _dot(b2, b2)],
        )
        alpha, beta = coord
        w = (F(1) - alpha - beta, alpha, beta)
        return c, _dot(_sub(c, p0), _sub(c, p0)), w

    # k == 4：直接解中心 c（3x3），与核心的 Gram 参数化不同
    rows = []
    rhs = []
    for p in pts[1:]:
        v = _sub(p, p0)
        rows.append(tuple(2 * t for t in v))
        rhs.append(_dot(p, p) - _dot(p0, p0))
    c = _cramer(rows, rhs)
    if c is None:
        return None
    c = tuple(c)
    vs = [_sub(p, p0) for p in pts[1:]]
    # c-p0 = sum_j alpha_j v_j：矩阵以 v_j 为 *列*（A[d][j] = v_j[d]）
    alpha = _cramer(
        [[vs[c][d] for c in range(3)] for d in range(3)],
        _sub(c, p0),
    )
    if alpha is None:
        return None
    w = (F(1) - sum(alpha, F(0)), *alpha)
    return c, _dot(_sub(c, p0), _sub(c, p0)), w


def reference_miniball(points):
    """独立暴力参考：枚举 1..4 支撑，同样的可行性条件，选 r^2 最小。"""
    uniq = list(dict.fromkeys(tuple(F(x) for x in p) for p in points))
    best = None
    for k in range(1, min(4, len(uniq)) + 1):
        for combo in combinations(range(len(uniq)), k):
            got = _reference_circumsphere([uniq[i] for i in combo])
            if got is None:
                continue
            c, r2, w = got
            if any(x < 0 for x in w):
                continue
            if any(_dot(_sub(q, c), _sub(q, c)) > r2 for q in uniq):
                continue
            key = (r2, k, combo)
            if best is None or key < best[0]:
                best = (key, c, r2, w, combo)
    assert best is not None
    _, c, r2, w, combo = best
    return c, r2, w, tuple(uniq[i] for i in combo), uniq


# ---------------------------------------------------------------------------
# 证书核验工具（只用结果自身的公开字段做恒等式核验）
# ---------------------------------------------------------------------------

def assert_valid_certificate(test, result, raw_points):
    for s in result.support:
        test.assertIsInstance(s.weight, Fraction)
        test.assertGreaterEqual(s.weight, 0)
    test.assertEqual(sum(result.weights), 1)
    # 残差 sum lambda_i (c - p_i) = 0
    for x in result.certificate_residual():
        test.assertEqual(x, 0)
    # 支撑点都在球面上
    for s in result.support:
        d = sum(
            (s.point[i] - result.center[i]) ** 2 for i in range(3)
        )
        test.assertEqual(d, result.radius_squared)
    # 全部原始点被包含
    for p in raw_points:
        pf = tuple(F(x) for x in p)
        d = sum((pf[i] - result.center[i]) ** 2 for i in range(3))
        test.assertLessEqual(d, result.radius_squared)
    # 对偶恒等式：r^2 = sum lambda_i |p_i|^2 - |sum lambda_i p_i|^2
    test.assertEqual(
        result.dual_identity_radius_squared(), result.radius_squared
    )
    # 权重-支撑一一对应，长度一致
    test.assertEqual(len(result.support), len(result.weights))


class TestCertificateAndKnownCases(unittest.TestCase):
    def test_single_point(self):
        r = minimum_enclosing_sphere([(3, -1, 2)])
        self.assertEqual(r.center, (F(3), F(-1), F(2)))
        self.assertEqual(r.radius_squared, 0)
        self.assertEqual(len(r.support), 1)
        self.assertEqual(r.support[0].weight, 1)
        assert_valid_certificate(self, r, [(3, -1, 2)])

    def test_two_points(self):
        r = minimum_enclosing_sphere([(0, 0, 0), (4, 0, 0)])
        self.assertEqual(r.center, (F(2), F(0), F(0)))
        self.assertEqual(r.radius_squared, 4)
        self.assertEqual(r.weights, (F(1, 2), F(1, 2)))

    def test_obtuse_triangle_support_is_longest_edge(self):
        pts = [(0, 0, 0), (4, 0, 0), (1, 1, 0)]
        r = minimum_enclosing_sphere(pts)
        # 钝角三角形：最小包围球以最长边为直径，第三点在球内
        self.assertEqual(len(r.support), 2)
        self.assertEqual(r.center, (F(2), 0, 0))
        self.assertEqual(r.radius_squared, 4)
        self.assertEqual(
            tuple(s.point for s in r.support),
            ((F(0), 0, 0), (F(4), 0, 0)),
        )
        self.assertEqual(r.weights, (F(1, 2), F(1, 2)))
        assert_valid_certificate(self, r, pts)

    def test_acute_triangle_three_supports(self):
        pts = [(0, 0, 0), (2, 0, 0), (1, 3, 0)]
        r = minimum_enclosing_sphere(pts)
        self.assertEqual(len(r.support), 3)
        # 对称等腰三角形，外心在 (1, 4/3, 0)
        self.assertEqual(r.center, (F(1), F(4, 3), 0))
        self.assertEqual(r.radius_squared, F(25, 9))
        for w in r.weights:
            self.assertGreater(w, 0)
        assert_valid_certificate(self, r, pts)

    def test_collinear_points(self):
        pts = [(-5, 1, 1), (0, 1, 1), (2, 1, 1), (7, 1, 1)]
        r = minimum_enclosing_sphere(pts)
        self.assertEqual(len(r.support), 2)
        self.assertEqual(r.center, (F(1), 1, 1))
        self.assertEqual(r.radius_squared, 36)
        assert_valid_certificate(self, r, pts)

    def test_collinear_non_axis_direction(self):
        # 沿 (1,2,3) 方向的三个共线点 t=-2,0,1
        pts = [(-2, -4, -6), (0, 0, 0), (1, 2, 3)]
        r = minimum_enclosing_sphere(pts)
        self.assertEqual(len(r.support), 2)
        self.assertEqual(r.center, (F(-1, 2), -1, F(-3, 2)))
        # |(3/2)(1,2,3)|^2 = 9/4 * 14 = 63/2
        self.assertEqual(r.radius_squared, F(63, 2))
        assert_valid_certificate(self, r, pts)

    def test_coplanar_tilted_equilateral_with_inner_point(self):
        # 点都在平面 z = x + y 上：正三角形（边长平方均为 8）
        # 加一个平面内（实为外心）和一个离面但在球内的点
        pts = [(0, 0, 0), (2, 0, 2), (0, 2, 2), (1, 1, 1)]
        r = minimum_enclosing_sphere(pts)
        self.assertEqual(len(r.support), 3)
        self.assertEqual(r.center, (F(2, 3), F(2, 3), F(4, 3)))
        self.assertEqual(r.radius_squared, F(8, 3))
        self.assertTrue(all(w == F(1, 3) for w in r.weights))
        assert_valid_certificate(self, r, pts)

    def test_coplanar_square_diagonal(self):
        pts = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
        r = minimum_enclosing_sphere(pts)
        self.assertEqual(r.center, (1, 1, 0))
        self.assertEqual(r.radius_squared, 2)
        # 两条对角线同为最优；按平局规则取支撑更少（2 点）且下标字典序
        self.assertEqual(len(r.support), 2)
        self.assertEqual(r.support[0].ids, (0,))
        self.assertEqual(r.support[1].ids, (2,))
        assert_valid_certificate(self, r, pts)

    def test_regular_tetrahedron(self):
        pts = [
            (1, 1, 1),
            (1, -1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
        ]
        r = minimum_enclosing_sphere(pts)
        self.assertEqual(r.center, (0, 0, 0))
        self.assertEqual(r.radius_squared, 3)
        self.assertEqual(len(r.support), 4)
        self.assertTrue(all(w == F(1, 4) for w in r.weights))
        assert_valid_certificate(self, r, pts)

    def test_duplicate_coordinates_preserve_ids(self):
        pts = [(0, 0, 0), (2, 0, 0), (0, 0, 0), (2, 0, 0), (1, 1, 0)]
        ids = ["a", "b", "c", "d", "e"]
        r = minimum_enclosing_sphere(pts, ids=ids)
        self.assertEqual(r.num_input_points, 5)
        self.assertEqual(r.num_unique_points, 3)
        id_sets = {s.point: set(s.ids) for s in r.support}
        self.assertEqual(id_sets[(F(0), 0, 0)], {"a", "c"})
        self.assertEqual(id_sets[(F(2), 0, 0)], {"b", "d"})
        # 重复坐标不改变几何结果
        r2 = minimum_enclosing_sphere([(0, 0, 0), (2, 0, 0), (1, 1, 0)])
        self.assertEqual(r.center, r2.center)
        self.assertEqual(r.radius_squared, r2.radius_squared)
        assert_valid_certificate(self, r, pts)

    def test_all_points_coincide(self):
        pts = [(5, 5, 5), (5, 5, 5), (5, 5, 5)]
        r = minimum_enclosing_sphere(pts)
        self.assertEqual(r.radius_squared, 0)
        self.assertEqual(len(r.support), 1)
        self.assertEqual(r.support[0].ids, (0, 1, 2))
        self.assertEqual(r.support[0].weight, 1)


class TestCrossCheckIndependentReference(unittest.TestCase):
    def _cross_check(self, pts):
        r = minimum_enclosing_sphere(pts)
        c, r2, w, sup_pts, _ = reference_miniball(pts)
        self.assertEqual(r.center, c)
        self.assertEqual(r.radius_squared, r2)
        self.assertEqual(tuple(s.point for s in r.support), sup_pts)
        self.assertEqual(r.weights, w)
        assert_valid_certificate(self, r, pts)

    def test_fixed_sets(self):
        sets = [
            [(0, 0, 0), (1, 2, 3)],
            [(0, 0, 0), (6, 0, 0), (3, 1, 0)],       # 钝三角形
            [(0, 0, 0), (4, 0, 0), (2, 3, 0), (2, 1, 1)],
            [(1, 1, 1), (-2, 0, 1), (3, -4, 2), (0, 0, 0), (2, 2, -1)],
            [
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
            ],
            [(-3, 7, 2), (4, -1, 5), (4, -1, 5), (2, 2, 2)],  # 含重复
        ]
        for pts in sets:
            with self.subTest(pts=pts):
                self._cross_check(pts)

    def test_deterministic_random_sets_up_to_8_points(self):
        # 确定性数据生成（固定种子），不使用任何在线语料
        rng = random.Random(20260922)
        for trial in range(60):
            n = rng.randint(1, 8)
            pts = [
                (rng.randint(-6, 6), rng.randint(-6, 6), rng.randint(-6, 6))
                for _ in range(n)
            ]
            with self.subTest(trial=trial, n=n):
                self._cross_check(pts)

    def test_fraction_inputs_exact(self):
        pts = [
            (F(1, 3), 0, 0),
            (F(1, 2), F(1, 7), 0),
            (F(-2, 5), F(3, 4), F(1, 9)),
        ]
        self._cross_check(pts)


class TestInputValidation(unittest.TestCase):
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            minimum_enclosing_sphere([])

    def test_too_many_points_rejected(self):
        pts = [(i, 0, 0) for i in range(31)]
        with self.assertRaises(ValueError):
            minimum_enclosing_sphere(pts)

    def test_boundary_30_accepted(self):
        # 沿一条线放 30 点（接口上限）：坐标 -15..14
        pts = [(i - 15, 0, 0) for i in range(30)]
        r = minimum_enclosing_sphere(pts)
        self.assertEqual(r.num_input_points, 30)
        # 中点 -1/2，最远点距离 29/2，r^2 = 841/4
        self.assertEqual(r.center, (F(-1, 2), 0, 0))
        self.assertEqual(r.radius_squared, F(841, 4))

    def test_wrong_dimension_locatable(self):
        with self.assertRaises(ValueError) as cm:
            minimum_enclosing_sphere([(0, 0, 0), (1, 2)])
        self.assertIn("points[1]", str(cm.exception))

    def test_nan_infinity_rejected_locatable(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as cm:
                    minimum_enclosing_sphere([(0, 0, 0), (1, bad, 0)])
                self.assertIn("points[1][1]", str(cm.exception))

    def test_bool_coordinate_rejected(self):
        with self.assertRaises(TypeError) as cm:
            minimum_enclosing_sphere([(True, 0, 0)])
        self.assertIn("bool", str(cm.exception))

    def test_bool_integer_argument_rejected(self):
        with self.assertRaises(TypeError):
            minimum_enclosing_sphere([(0, 0, 0)], budget=True)

    def test_bad_type_message(self):
        with self.assertRaises(TypeError):
            minimum_enclosing_sphere([("x", 0, 0)])
        with self.assertRaises(TypeError):
            minimum_enclosing_sphere("abc")

    def test_ids_mismatch_and_duplicate(self):
        with self.assertRaises(ValueError):
            minimum_enclosing_sphere([(0, 0, 0), (1, 0, 0)], ids=[0])
        with self.assertRaises(ValueError) as cm:
            minimum_enclosing_sphere(
                [(0, 0, 0), (1, 0, 0)], ids=["x", "x"]
            )
        self.assertIn("ids[1]", str(cm.exception))
        with self.assertRaises(TypeError):
            minimum_enclosing_sphere(
                [(0, 0, 0), (1, 0, 0)], ids=([0], [1])
            )

    def test_finite_float_accepted_exactly(self):
        r = minimum_enclosing_sphere([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)])
        self.assertEqual(r.center, (F(1, 4), 0, 0))
        self.assertEqual(r.radius_squared, F(1, 16))

    def test_input_not_mutated(self):
        pts = [(0, 0, 0), (2, 0, 0)]
        ids = ["a", "b"]
        minimum_enclosing_sphere(pts, ids=ids)
        self.assertEqual(pts, [(0, 0, 0), (2, 0, 0)])
        self.assertEqual(ids, ["a", "b"])


class TestBudgetSemantics(unittest.TestCase):
    def test_budget_sufficient(self):
        pts = [(0, 0, 0), (2, 0, 0)]
        r = minimum_enclosing_sphere(pts, budget=100)
        self.assertEqual(r.radius_squared, 1)

    def test_budget_exhausted_is_distinct_from_infeasible(self):
        pts = [(0, 0, 0), (2, 0, 0), (0, 2, 0)]
        # 预算 1：只检查第一个单点候选（覆盖不了全部点），随即耗尽
        with self.assertRaises(SearchBudgetExhaustedError):
            minimum_enclosing_sphere(pts, budget=1)
        # 耗尽 ≠ 无解：放大预算后正常求解
        r = minimum_enclosing_sphere(pts, budget=1000)
        self.assertIsInstance(r, MinimumEnclosingSphere)
        # NoFeasibleCandidateError 是不同类型
        self.assertNotIn(
            NoFeasibleCandidateError,
            SearchBudgetExhaustedError.__mro__,
        )

    def test_budget_invalid(self):
        with self.assertRaises(ValueError):
            minimum_enclosing_sphere([(0, 0, 0)], budget=0)
        with self.assertRaises(TypeError):
            minimum_enclosing_sphere([(0, 0, 0)], budget=1.5)


class TestResultDataclass(unittest.TestCase):
    def test_frozen_and_contains(self):
        r = minimum_enclosing_sphere([(0, 0, 0), (2, 0, 0)])
        with self.assertRaises(Exception):
            r.radius_squared = 9
        self.assertTrue(r.contains((1, 0, 0)))
        self.assertTrue(r.contains((0, 0, 0)))
        self.assertFalse(r.contains((0, 2, 0)))

    def test_float_views_finite(self):
        r = minimum_enclosing_sphere(
            [(F(1, 3), 0, 0), (F(1, 2), 0, 0)]
        )
        cx, cy, cz = r.center_float
        self.assertTrue(math.isfinite(cx))
        self.assertTrue(math.isfinite(r.radius_float))
        # 浮点视图不改变精确字段
        self.assertIsInstance(r.radius_squared, Fraction)


if __name__ == "__main__":
    unittest.main()
