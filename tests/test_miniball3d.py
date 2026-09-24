"""miniball3d 的必要测试。

参考实现独立编写（叉积 + 行列式 Cramer 法则、按 R^3 直接求中心），
不 import、不调用被测核心的任何函数；被测核心用的是仿射包内的
Gram 方程 + 高斯消元，二者代码路径相互独立。两套实现只在测试里
逐字段对照。对偶恒等式与证书条件做精确（Fraction）核验。

输入球（minimum_enclosing_ball）部分的期望值全部独立得出：
一维用手写的最左/最右触及区间公式，二维用手算解析例子与正三角形
几何关系，并独立重算 KKT 证书条件（凸性 ⇒ 充分必要）；零半径退化
情形与三维精确 Fraction 求解器逐字段对照，而不是调用被测数值入口
生成期望值。
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
    minimum_enclosing_ball,
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


# ---------------------------------------------------------------------------
# 输入球接口（minimum_enclosing_ball）
# ---------------------------------------------------------------------------

def _indep_dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def assert_valid_ball_certificate(test, res, raw_balls, tol_scale=1.0):
    """独立核验：逐球约束、支撑紧贴、权重条件、法向平衡、最优性下界。

    所有量在测试内重算，不调用求解器产生期望值。
    """
    c, R = res.center, res.radius
    eps = res.tol * 100 * max(1.0, R, tol_scale)

    # 1) 逐球独立约束：dist(c,p_i) + a_i <= R（每个输入球单独一个谓词）
    for (p, a) in raw_balls:
        test.assertLessEqual(_indep_dist(c, p) + a, R + eps)
    test.assertLessEqual(res.max_violation(), eps)
    # constraint_values 与独立重算一致
    for value, (p, a) in zip(res.constraint_values, raw_balls):
        test.assertAlmostEqual(value, _indep_dist(c, p) + a, places=12)
    # slack + value == R
    for slack, value in zip(res.slacks, res.constraint_values):
        test.assertAlmostEqual(slack + value, R, places=12)

    # 2) 每个支撑球紧贴：dist + a == R
    support_keys = set()
    for s in res.support:
        test.assertGreaterEqual(s.weight, -eps)
        test.assertAlmostEqual(
            _indep_dist(c, s.center) + s.radius, R, places=9
        )
        support_keys.add((tuple(s.center), s.radius, s.ids))

    # 3) 权重：凸组合（和为 1、非负）
    weights = res.weights
    test.assertEqual(len(weights), len(res.support))
    for w in weights:
        test.assertGreaterEqual(w, -eps)
    test.assertAlmostEqual(sum(weights), 1.0, places=10)

    # 4) 法向平衡 Σ λ n = 0（dist=0 的退化单支撑除外，已在 API 侧处理）
    for component in res.balance_residual():
        test.assertLess(abs(component), math.sqrt(res.tol) * 10)

    # 5) 最优性充分条件（凸性 ⇒ 必要）：沿任意方向小位移都会让某支撑越界。
    #    用数值方向探针独立验证：R 不超过 c 邻域内能达到的最小上界，
    #    即对 24 个试探方向 h，max_i(dist(c+δh,p_i)+a_i) >= R - eps。
    dim = len(c)
    directions = [(1.0,), (-1.0,)] if dim == 1 else [
        (math.cos(2 * math.pi * j / 24), math.sin(2 * math.pi * j / 24))
        for j in range(24)
    ]
    delta = 1e-5 * max(1.0, R)
    for h in directions:
        probe = tuple(c[d] + delta * h[d] for d in range(dim))
        need = max(_indep_dist(probe, p) + a for p, a in raw_balls)
        test.assertGreaterEqual(need, R - eps * 10)


class TestEnclosingBallsKnownCases(unittest.TestCase):
    def test_single_ball(self):
        r = minimum_enclosing_ball([((2.0, -1.0), 0.75)])
        self.assertEqual(r.center, (2.0, -1.0))
        self.assertEqual(r.radius, 0.75)
        self.assertEqual(len(r.support), 1)
        self.assertEqual(r.weights, (1.0,))
        self.assertEqual(r.num_input_balls, 1)

    def test_single_point_zero_radius(self):
        r = minimum_enclosing_ball([((3.0, 1.0), 0.0)])
        self.assertEqual(r.radius, 0.0)
        self.assertEqual(r.center, (3.0, 1.0))

    def test_two_points_midpoint(self):
        r = minimum_enclosing_ball([((0.0, 0.0), 0.0), ((4.0, 0.0), 0.0)])
        self.assertEqual(r.center, (2.0, 0.0))
        self.assertEqual(r.radius, 2.0)
        self.assertEqual(r.weights, (0.5, 0.5))
        assert_valid_ball_certificate(self, r, [((0.0, 0.0), 0.0), ((4.0, 0.0), 0.0)])

    def test_two_unequal_balls_analytic(self):
        # 心 0 与 10、半径 1 与 3：R=(d+a+b)/2=7，c=p0+(R-a0)=6
        r = minimum_enclosing_ball(
            [((0.0,), 1.0), ((10.0,), 3.0)], dim=1
        )
        self.assertAlmostEqual(r.radius, 7.0, places=12)
        self.assertAlmostEqual(r.center[0], 6.0, places=12)
        assert_valid_ball_certificate(
            self, r, [((0.0,), 1.0), ((10.0,), 3.0)]
        )

    def test_two_balls_2d_analytic(self):
        # 心 (0,0)、(6,8)（距离 10），半径 1、3：R=7，c=p0+(R-1)u=(3.6,4.8)
        r = minimum_enclosing_ball(
            [((0.0, 0.0), 1.0), ((6.0, 8.0), 3.0)]
        )
        self.assertAlmostEqual(r.radius, 7.0, places=12)
        self.assertAlmostEqual(r.center[0], 3.6, places=12)
        self.assertAlmostEqual(r.center[1], 4.8, places=12)
        self.assertEqual(len(r.support), 2)

    def test_nested_ball_single_support(self):
        # 大球完全覆盖小球：d+a <= R0 ⇔ 3+1 <= 5
        r = minimum_enclosing_ball([((0.0,), 5.0), ((3.0,), 1.0)], dim=1)
        self.assertEqual(r.center, (0.0,))
        self.assertEqual(r.radius, 5.0)
        self.assertEqual(len(r.support), 1)
        self.assertEqual(r.support[0].radius, 5.0)
        # 被包含球的 slack：R - (d + a) = 5 - 4 = 1 ≥ 0
        self.assertGreaterEqual(r.slacks[1], 1.0 - 1e-12)
        self.assertLessEqual(r.slacks[1], 1.0 + 1e-12)

    def test_concentric_balls(self):
        r = minimum_enclosing_ball(
            [((1.0, 1.0), 3.0), ((1.0, 1.0), 1.0)], ids=("big", "small")
        )
        self.assertEqual(r.radius, 3.0)
        self.assertEqual(r.support[0].ids, ("big",))
        self.assertAlmostEqual(r.slacks[1], 2.0, places=12)

    def test_equilateral_points_three_support(self):
        # 边长 4 的正三角形顶点：外心 (2, 2/√3)，R = 4/√3
        h = 2.0 * math.sqrt(3.0)
        pts = [((0.0, 0.0), 0.0), ((4.0, 0.0), 0.0), ((2.0, h), 0.0)]
        r = minimum_enclosing_ball(pts)
        self.assertAlmostEqual(r.center[0], 2.0, places=12)
        self.assertAlmostEqual(r.center[1], 2.0 / math.sqrt(3.0), places=12)
        self.assertAlmostEqual(r.radius, 4.0 / math.sqrt(3.0), places=12)
        self.assertEqual(len(r.support), 3)
        for w in r.weights:
            self.assertAlmostEqual(w, 1.0 / 3.0, places=10)
        assert_valid_ball_certificate(self, r, pts)

    def test_three_equal_balls(self):
        # 三个半径 1 的球心在正三角形顶点：R = 4/√3 + 1
        h = 2.0 * math.sqrt(3.0)
        balls = [((0.0, 0.0), 1.0), ((4.0, 0.0), 1.0), ((2.0, h), 1.0)]
        r = minimum_enclosing_ball(balls)
        self.assertAlmostEqual(r.radius, 4.0 / math.sqrt(3.0) + 1.0, places=12)
        self.assertAlmostEqual(r.center[1], 2.0 / math.sqrt(3.0), places=12)
        assert_valid_ball_certificate(self, r, balls)

    def test_three_unequal_balls_constructed_solution(self):
        # 解析构造：c*=0, R*=10，三条法向相隔 120° 时 Σn=0。
        #   n1=(1,0),      d1=8, a1=2  -> p1=(-8,0)
        #   n2=(-1/2,√3/2),d2=5, a2=5  -> p2=(5/2,-5√3/2)
        #   n3=(-1/2,-√3/2),d3=6,a3=4  -> p3=(3,3√3)
        balls = [
            ((-8.0, 0.0), 2.0),
            ((2.5, -5.0 * math.sqrt(3.0) / 2.0), 5.0),
            ((3.0, 3.0 * math.sqrt(3.0)), 4.0),
        ]
        r = minimum_enclosing_ball(balls)
        self.assertAlmostEqual(r.radius, 10.0, places=9)
        self.assertAlmostEqual(r.center[0], 0.0, places=8)
        self.assertAlmostEqual(r.center[1], 0.0, places=8)
        self.assertEqual(len(r.support), 3)
        for w in r.weights:
            self.assertAlmostEqual(w, 1.0 / 3.0, places=8)
        assert_valid_ball_certificate(self, r, balls, tol_scale=10.0)

    def test_obtuse_triangle_degrades_to_two_support(self):
        balls = [((0.0, 0.0), 0.0), ((4.0, 0.0), 0.0), ((1.0, 1.0), 0.0)]
        r = minimum_enclosing_ball(balls)
        self.assertEqual(len(r.support), 2)
        self.assertEqual(r.center, (2.0, 0.0))
        self.assertEqual(r.radius, 2.0)
        # 第三点严格在内部：dist((2,0),(1,1))=√2 < 2
        self.assertGreater(r.slacks[2], 0.5)

    def test_collinear_2d_uses_two_support(self):
        balls = [
            ((0.0, 0.0), 0.0), ((2.0, 0.0), 0.0), ((5.0, 0.0), 0.0)
        ]
        r = minimum_enclosing_ball(balls)
        self.assertEqual(len(r.support), 2)
        self.assertAlmostEqual(r.center[0], 2.5, places=12)
        self.assertAlmostEqual(r.radius, 2.5, places=12)

    def test_1d_extremes(self):
        # 独立期望：触及区间 [p-a, p+a] 的并集，c=(L+U)/2, R=(U-L)/2
        raw = [(-3.0, 0.5), (0.0, 1.0), (5.0, 2.0)]
        left = min(p - a for p, a in raw)   # -3.5
        right = max(p + a for p, a in raw)  # 7.0
        balls = [(((p,), a)) for p, a in raw]
        r = minimum_enclosing_ball(balls, dim=1)
        self.assertAlmostEqual(r.center[0], (left + right) / 2, places=12)
        self.assertAlmostEqual(r.radius, (right - left) / 2, places=12)
        assert_valid_ball_certificate(self, r, balls)

    def test_1d_integer_points(self):
        balls = [((float(i),), 0.0) for i in range(6)]
        r = minimum_enclosing_ball(balls, dim=1)
        self.assertAlmostEqual(r.center[0], 2.5, places=12)
        self.assertAlmostEqual(r.radius, 2.5, places=12)
        self.assertEqual(len(r.support), 2)

    def test_six_balls_boundary_count(self):
        # 6 个输入球（接口上限）：正六边形顶点，半径各异
        balls = []
        for j in range(6):
            ang = 2 * math.pi * j / 6
            balls.append(((math.cos(ang), math.sin(ang)), 0.1 * (j + 1)))
        r = minimum_enclosing_ball(balls)
        self.assertEqual(r.num_input_balls, 6)
        assert_valid_ball_certificate(self, r, balls)

    def test_duplicate_balls_merge_ids(self):
        r = minimum_enclosing_ball(
            [((0.0, 0.0), 1.0), ((0.0, 0.0), 1.0), ((3.0, 0.0), 0.0)],
            ids=("a", "b", "c"),
        )
        self.assertEqual(r.num_input_balls, 3)
        sup = {s.ids: s for s in r.support}
        self.assertIn(("a", "b"), sup)

    def test_integer_and_fraction_inputs(self):
        # int 与 Fraction 都按精确值转 float：球 (1/3,0) r=1/2 与 (1,0) 点
        r = minimum_enclosing_ball(
            [((F(1, 3), 0), F(1, 2)), ((1, 0), 0)]
        )
        # 两心距 2/3；R=(d+a+b)/2=(2/3+1/2)/2=7/12
        # c = p0 + (R-a)·u，u 为 p0→p1 单位方向 (1,0)，故 c_x=1/3+1/12=5/12
        self.assertAlmostEqual(r.radius, F(7, 12), places=12)
        expected_c = F(1, 3) + (F(7, 12) - F(1, 2))
        self.assertAlmostEqual(r.center[0], float(expected_c), places=12)


class TestEnclosingBallsDegenerateToPoints(unittest.TestCase):
    """零半径输入必须与点集最小包围球一致（与精确 Fraction 求解器对照）。"""

    def _cross_check_3d_embedding(self, points3):
        # 三维精确结果投影到二维（点集都在 z=0 平面）
        exact = minimum_enclosing_sphere(points3)
        balls = [((float(p[0]), float(p[1])), 0.0) for p in points3]
        r = minimum_enclosing_ball(balls)
        self.assertAlmostEqual(
            r.radius, math.sqrt(float(exact.radius_squared)), places=8
        )
        self.assertAlmostEqual(r.center[0], float(exact.center[0]), places=8)
        self.assertAlmostEqual(r.center[1], float(exact.center[1]), places=8)
        # 支撑数一致
        self.assertEqual(len(r.support), len(exact.support))
        assert_valid_ball_certificate(self, r, balls)

    def test_square_matches_exact_solver(self):
        pts = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
        self._cross_check_3d_embedding(pts)

    def test_acute_triangle_matches_exact_solver(self):
        pts = [(0, 0, 0), (2, 0, 0), (1, 3, 0)]
        self._cross_check_3d_embedding(pts)

    def test_random_sets_match_exact_solver(self):
        rng = random.Random(20260924)
        for trial in range(40):
            n = rng.randint(1, 6)
            pts = [
                (rng.randint(-5, 5), rng.randint(-5, 5), 0)
                for _ in range(n)
            ]
            with self.subTest(trial=trial, n=n):
                self._cross_check_3d_embedding(pts)

    def test_1d_matches_exact_solver(self):
        # 一维沿 x 轴的点与三维精确结果对照
        pts3 = [(-4, 0, 0), (1, 0, 0), (3, 0, 0)]
        exact = minimum_enclosing_sphere(pts3)
        r = minimum_enclosing_ball(
            [((float(p[0]),), 0.0) for p in pts3], dim=1
        )
        self.assertAlmostEqual(r.radius, math.sqrt(float(exact.radius_squared)), places=9)
        self.assertAlmostEqual(r.center[0], float(exact.center[0]), places=9)


class TestEnclosingBallsIndependentSweep(unittest.TestCase):
    def test_1d_exhaustive_reference(self):
        # 一维独立参考：触及区间并集（完全独立的代码路径，不调用求解器）
        rng = random.Random(424242)
        for trial in range(100):
            n = rng.randint(1, 6)
            balls = [
                ((rng.uniform(-5, 5),), rng.uniform(0, 2))
                for _ in range(n)
            ]
            left = min(p[0] - a for (p, a) in balls)
            right = max(p[0] + a for (p, a) in balls)
            r = minimum_enclosing_ball(balls, dim=1)
            self.assertAlmostEqual(r.center[0], (left + right) / 2, places=8)
            self.assertAlmostEqual(r.radius, (right - left) / 2, places=8)
            assert_valid_ball_certificate(self, r, balls)

    def test_2d_grid_enumeration(self):
        # 小规模枚举：球心取自整数网格、半径取离散集合，独立探针验证最优性
        grid = [(x, y) for x in (-2, 0, 2) for y in (-2, 0, 2)]
        radii = (0.0, 0.5, 1.0)
        rng = random.Random(99)
        for trial in range(60):
            n = rng.randint(2, 5)
            idxs = rng.sample(range(len(grid)), n)
            balls = [
                ((float(grid[i][0]), float(grid[i][1])), rng.choice(radii))
                for i in idxs
            ]
            r = minimum_enclosing_ball(balls)
            assert_valid_ball_certificate(self, r, balls)
            # 独立全局网格下界：在细网格上枚举中心，R* 不得明显大于
            # 网格找到的最小上界（网格分辨率 0.1，允许 0.05 误差）
            best_grid = float("inf")
            gx = [x / 10 for x in range(-30, 31)]
            gy = [y / 10 for y in range(-30, 31)]
            for x in gx:
                for y in gy:
                    need = max(
                        math.hypot(x - p[0], y - p[1]) + a for p, a in balls
                    )
                    if need < best_grid:
                        best_grid = need
            self.assertLessEqual(r.radius, best_grid + 0.06)


class TestEnclosingBallsValidation(unittest.TestCase):
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            minimum_enclosing_ball([])

    def test_too_many_rejected(self):
        balls = [((0.0, 0.0), 0.0)] * 7
        with self.assertRaises(ValueError) as cm:
            minimum_enclosing_ball(balls)
        self.assertIn("at most 6", str(cm.exception))

    def test_six_accepted(self):
        balls = [((float(i), 0.0), 0.0) for i in range(6)]
        r = minimum_enclosing_ball(balls)
        self.assertEqual(r.num_input_balls, 6)

    def test_negative_radius_locatable(self):
        with self.assertRaises(ValueError) as cm:
            minimum_enclosing_ball([((0.0, 0.0), -0.1)])
        self.assertIn("balls[0]", str(cm.exception))

    def test_nan_inf_rejected_locatable(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as cm:
                    minimum_enclosing_ball([((0.0, bad), 0.0)])
                self.assertIn("balls[0][0]", str(cm.exception))
            with self.assertRaises(ValueError):
                minimum_enclosing_ball([((0.0, 0.0), bad)])

    def test_wrong_dimension(self):
        with self.assertRaises(ValueError) as cm:
            minimum_enclosing_ball([((1.0,), 0.0)], dim=2)
        self.assertIn("balls[0]", str(cm.exception))
        with self.assertRaises(ValueError):
            minimum_enclosing_ball([((1.0, 2.0, 3.0), 0.0)], dim=2)
        with self.assertRaises(ValueError) as cm:
            minimum_enclosing_ball([((1.0, 2.0), 0.0)], dim=1)
        self.assertIn("balls[0]", str(cm.exception))

    def test_bad_pair_shape(self):
        with self.assertRaises(ValueError):
            minimum_enclosing_ball([((0.0, 0.0),)])  # 只有中心没有半径
        with self.assertRaises(TypeError):
            minimum_enclosing_ball([("x", 0.0)])
        with self.assertRaises(TypeError):
            minimum_enclosing_ball("ab")

    def test_bool_rejected(self):
        with self.assertRaises(TypeError):
            minimum_enclosing_ball([((True, 0.0), 0.0)])
        with self.assertRaises(TypeError):
            minimum_enclosing_ball([((0.0, 0.0), True)])
        with self.assertRaises(TypeError):
            minimum_enclosing_ball([((0.0, 0.0), 0.0)], dim=True)

    def test_dim_restrictions(self):
        with self.assertRaises(ValueError):
            minimum_enclosing_ball([((0.0, 0.0, 0.0), 0.0)], dim=3)
        with self.assertRaises(TypeError):
            minimum_enclosing_ball([((0.0,), 0.0)], dim=1.0)

    def test_tol_validation(self):
        with self.assertRaises(ValueError):
            minimum_enclosing_ball([((0.0,), 0.0)], dim=1, tol=-1e-9)
        with self.assertRaises(ValueError):
            minimum_enclosing_ball([((0.0,), 0.0)], dim=1, tol=float("nan"))
        with self.assertRaises(TypeError):
            minimum_enclosing_ball([((0.0,), 0.0)], dim=1, tol="x")
        # tol=0 合法：精确可表示的配置仍应通过
        r = minimum_enclosing_ball(
            [((0.0,), 0.0), ((2.0,), 0.0)], dim=1, tol=0.0
        )
        self.assertAlmostEqual(r.radius, 1.0)

    def test_ids_semantics(self):
        with self.assertRaises(ValueError):
            minimum_enclosing_ball(
                [((0.0,), 0.0), ((1.0,), 0.0)], dim=1, ids=[1]
            )
        with self.assertRaises(ValueError) as cm:
            minimum_enclosing_ball(
                [((0.0,), 0.0), ((1.0,), 0.0)], dim=1, ids=["x", "x"]
            )
        self.assertIn("ids[1]", str(cm.exception))

    def test_input_not_mutated(self):
        balls = [((0.0, 0.0), 1.0), ((3.0, 0.0), 0.5)]
        ids = ["a", "b"]
        minimum_enclosing_ball(balls, ids=ids)
        self.assertEqual(balls, [((0.0, 0.0), 1.0), ((3.0, 0.0), 0.5)])
        self.assertEqual(ids, ["a", "b"])

    def test_frozen_result(self):
        r = minimum_enclosing_ball([((0.0,), 1.0)], dim=1)
        with self.assertRaises(Exception):
            r.radius = 9.0

    def test_contains_helpers(self):
        r = minimum_enclosing_ball(
            [((0.0, 0.0), 1.0), ((4.0, 0.0), 1.0)]
        )
        # R = 3, c = (2,0)；球面上的点算包含
        self.assertTrue(r.contains_point((-1.0, 0.0)))
        self.assertTrue(r.contains_point((5.0, 0.0)))
        # (2,2) 距心 2 ≤ 3 在球内；(2,3.5) 距心 3.5 > 3 在球外
        self.assertTrue(r.contains_point((2.0, 2.0)))
        self.assertFalse(r.contains_point((2.0, 3.5)))
        # 被包含的球：心 (2,0) r=1
        self.assertTrue(r.contains_ball((2.0, 0.0), 1.0))
        # 越界的球：心 (2,1.5) r=2，dist+2=3.5 > 3
        self.assertFalse(r.contains_ball((2.0, 1.5), 2.0))

    def test_balls_attribute_is_independent_copy(self):
        raw = [((0.0, 0.0), 1.0)]
        r = minimum_enclosing_ball(raw)
        self.assertEqual(r.balls, (((0.0, 0.0), 1.0),))
        # 修改原输入不影响结果内的存档
        raw[0] = ((9.0, 9.0), 9.0)
        self.assertEqual(r.balls, (((0.0, 0.0), 1.0),))


if __name__ == "__main__":
    unittest.main()
