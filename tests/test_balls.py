"""minimum_enclosing_ball_of_balls（1D/2D 测量球覆盖）的必要测试。

期望值绝不取自被测实现：
  * 1D 用手算的区间并集解析值（Fraction 精确比较）；
  * 2D 用一份*独立参考实现*：在绝对坐标下把相切方程相减得到关于
    (c_x,c_y,R) 的线性方程，2x2 用 Cramer 法则解出 c(R)，代回得 R 的
    二次方程；它与被测核心（在基 {u1,u2} 内的 Gram 参数化 + 高斯消元
    组织方式）是两条独立代码路径；
  * 另外对每个结果做凸分析 KKT 证书独立核验：相切支撑球的单位向量
    （从输入球心指向结果球心）存在非负凸组合等于零向量。F(c)=
    max_i(|c-p_i|+r_i) 是凸函数，该条件是全局最优的充要条件，
    不依赖被测代码产生的任何量（除结果本身外，单位向量、系数全部
    在本文件内重算）。
随机数据由固定种子确定性生成，全部离线。
"""

from fractions import Fraction
from itertools import combinations
import math
import random
import unittest

import miniball3d
from miniball3d import (
    BallSupport,
    MinimumEnclosingBall,
    NoFeasibleCandidateError,
    minimum_enclosing_ball_of_balls,
)

F = Fraction


# ---------------------------------------------------------------------------
# 独立参考实现（2D）：绝对坐标 + Cramer 法则，不调用被测核心任何函数
# ---------------------------------------------------------------------------

def _ref_solve2(A, b):
    """2x2 Cramer 法则；奇异返回 None。"""
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if det == 0.0:
        return None
    return (
        (b[0] * A[1][1] - A[0][1] * b[1]) / det,
        (A[0][0] * b[1] - b[0] * A[1][0]) / det,
    )


def _ref_pair(p0, r0, p1, r1):
    """两球外切候选（独立写法：直接解中点偏移 s=(d+r0-r1)/2 的镜像推导）。

    令 c = midpoint + g u，midpoint=(p0+p1)/2，两约束相减得
    |c-p1| - |c-p0| = r0 - r1；c 在连线上时 |c-p1|-|c-p0| = -2g，
    故 g = (r1 - r0)/2，c = midpoint + (r1-r0)/2 u。
    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    d = math.hypot(dx, dy)
    if d == 0.0:
        return None
    ux, uy = dx / d, dy / d
    g = (r1 - r0) / 2.0
    t = d / 2.0 + g  # 距 p0 的有向距离
    if t < 0.0:
        return None
    c = (p0[0] + t * ux, p0[1] + t * uy)
    return c, r0 + t


def _ref_triple(centers, radii, idx):
    """三球候选（绝对坐标路径）：相减得到 2 个关于 (cx,cy,R) 的线性方程，
    解出 c(R)=c0+cv*R，代回 |c-p0|=(R-r0) 得二次方程，回传全部可行根。
    """
    (p0, r0), (p1, r1), (p2, r2) = (
        (centers[idx[0]], radii[idx[0]]),
        (centers[idx[1]], radii[idx[1]]),
        (centers[idx[2]], radii[idx[2]]),
    )
    u1 = (p1[0] - p0[0], p1[1] - p0[1])
    u2 = (p2[0] - p0[0], p2[1] - p0[1])
    A = [[2 * u1[0], 2 * u1[1]], [2 * u2[0], 2 * u2[1]]]
    k1 = p1[0] ** 2 + p1[1] ** 2 - (p0[0] ** 2 + p0[1] ** 2) - (r1 ** 2 - r0 ** 2)
    k2 = p2[0] ** 2 + p2[1] ** 2 - (p0[0] ** 2 + p0[1] ** 2) - (r2 ** 2 - r0 ** 2)
    h1, h2 = 2 * (r1 - r0), 2 * (r2 - r0)
    base = _ref_solve2(A, (k1, k2))
    slope = _ref_solve2(A, (h1, h2))
    if base is None or slope is None:
        return []
    # c(R) = base + R slope；|c(R) - p0|^2 = (R - r0)^2
    c0x, c0y = base[0] - p0[0], base[1] - p0[1]
    vx, vy = slope
    qa = vx * vx + vy * vy - 1.0
    qb = 2.0 * (c0x * vx + c0y * vy + r0)
    qc = c0x * c0x + c0y * c0y - r0 * r0
    roots = []
    if qa == 0.0:
        if qb != 0.0:
            roots.append(-qc / qb)
    else:
        disc = qb * qb - 4 * qa * qc
        if disc >= 0.0:
            s = math.sqrt(disc)
            roots.extend(((-qb + s) / (2 * qa), (-qb - s) / (2 * qa)))
    out = []
    for R in roots:
        cx, cy = base[0] + R * vx, base[1] + R * vy
        if not all(math.isfinite(z) for z in (cx, cy, R)):
            continue
        if min(R - r0, R - r1, R - r2) < 0.0:
            continue
        out.append(((cx, cy), R))
    return out


def reference_minimum_ball_2d(balls):
    """独立暴力参考：枚举 1/2/3 球的全部相切候选，可行者取 R 最小。"""
    pts = [(float(F(c[0])), float(F(c[1]))) for c, _ in balls]
    rads = [float(F(r)) for _, r in balls]
    n = len(pts)

    def feasible(c, R):
        return all(math.hypot(c[0] - q[0], c[1] - q[1]) + s <= R + 1e-12
                   for q, s in zip(pts, rads))

    best = None
    for i in range(n):
        cand = (pts[i], rads[i])
        if feasible(*cand) and (best is None or cand[1] < best[1]):
            best = cand
    for i, j in combinations(range(n), 2):
        got = _ref_pair(pts[i], rads[i], pts[j], rads[j])
        if got is not None and feasible(*got) and (best is None or got[1] < best[1]):
            best = got
    for tri in combinations(range(n), 3):
        for got in _ref_triple(pts, rads, tri):
            if feasible(*got) and (best is None or got[1] < best[1]):
                best = got
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# KKT 最优性证书（独立核验）
# ---------------------------------------------------------------------------

def _solve3(mat, rhs):
    """带部分选主元的 3x3 高斯消元；奇异返回 None。"""
    m = [list(mat[r]) + [rhs[r]] for r in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-15:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            m[r] = [m[r][k] - f * m[col][k] for k in range(4)]
    x = [0.0, 0.0, 0.0]
    for i in range(2, -1, -1):
        x[i] = (m[i][3] - sum(m[i][k] * x[k] for k in range(i + 1, 3))) / m[i][i]
    return x


def assert_kkt_certificate(test, res):
    """对 2D 结果独立核验凸最优性的 KKT 充要条件。

    设 F(c)=max_i (|c-p_i|+r_i)。结果球给出 F(c)=R 且对所有球
    |c-p_i|+r_i<=R+tol。对每个相切支撑 i，令 u_i=(c-p_i)/|c-p_i|
    （距离为 0 的单支撑情形直接通过）。全局最优 iff 存在 lambda_i>=0、
    sum lambda=1 使 sum lambda_i u_i = 0（0 属于活动约束次梯度凸包；
    F 凸，故 KKT 条件充分且必要）。
    """
    test.assertEqual(res.dimension, 2)
    c = res.center
    tol = float(res.tol)
    active = []
    for s in res.support:
        p = (float(s.center[0]), float(s.center[1]))
        d = math.hypot(c[0] - p[0], c[1] - p[1])
        # 支撑必须相切（在容差内）
        test.assertLessEqual(abs(d + float(s.radius) - float(res.radius)),
                             max(tol, 1e-9) * 10)
        active.append((p, d))
    # 全部输入球可行
    test.assertLessEqual(res.max_violation(), tol + 1e-9)

    if len(active) == 1:
        (_, d), = active
        test.assertLessEqual(d, 1e-9)  # 单支撑：球心与输入球心重合
        return

    units = []
    for p, d in active:
        test.assertGreater(d, 1e-9)  # 多支撑时中心不可能与某球心重合
        units.append(((c[0] - p[0]) / d, (c[1] - p[1]) / d))

    if len(active) == 2:
        # u0 与 u1 反向即存在非负权重
        test.assertAlmostEqual(units[0][0] + units[1][0], 0.0, delta=1e-8)
        test.assertAlmostEqual(units[0][1] + units[1][1], 0.0, delta=1e-8)
        return

    test.assertEqual(len(active), 3)
    # lambda0+lambda1+lambda2=1, sum lambda u = 0
    mat = [
        [1.0, 1.0, 1.0],
        [units[0][0], units[1][0], units[2][0]],
        [units[0][1], units[1][1], units[2][1]],
    ]
    lam = _solve3(mat, (1.0, 0.0, 0.0))
    test.assertIsNotNone(lam)
    for w in lam:
        test.assertGreaterEqual(w, -1e-8)
    test.assertAlmostEqual(sum(lam), 1.0, delta=1e-9)


def _feasible(test, res, balls, *, tol=None):
    tol = float(res.tol) if tol is None else tol
    for (c, r) in balls:
        p = tuple(float(v) for v in c)
        cf = tuple(float(v) for v in res.center)
        d = math.sqrt(sum((p[a] - cf[a]) ** 2 for a in range(res.dimension)))
        test.assertLessEqual(d + float(r), float(res.radius) + tol + 1e-9)


# ---------------------------------------------------------------------------
# 手算解析用例
# ---------------------------------------------------------------------------

class TestHandComputed1D(unittest.TestCase):
    def test_two_intervals(self):
        # 区间 [-1,1] 与 [3,7]，并集 [-1,7]：c=3, R=4
        res = minimum_enclosing_ball_of_balls([((0,), 1), ((5,), 2)])
        self.assertEqual(res.dimension, 1)
        self.assertEqual(res.center, (F(3),))
        self.assertEqual(res.radius, F(4))
        self.assertEqual(res.method, "1d-exact-interval")
        self.assertEqual(res.tol, F(0))
        self.assertEqual(len(res.support), 2)
        self.assertEqual(tuple(s.weight for s in res.support),
                         (F(1, 2), F(1, 2)))
        self.assertEqual(res.max_violation(), 0.0)

    def test_nested_interval_single_support(self):
        # 大球 [-10,10] 包住小球 [0,2]：答案就是大球
        res = minimum_enclosing_ball_of_balls([((0,), 10), ((1,), 1)])
        self.assertEqual(res.center, (F(0),))
        self.assertEqual(res.radius, F(10))
        self.assertEqual(len(res.support), 1)
        self.assertEqual(res.support[0].weight, F(1))

    def test_touching_intervals(self):
        # [-1,1] 与 [1,3] 恰相接：c=1, R=2
        res = minimum_enclosing_ball_of_balls([((0,), 1), ((2,), 1)])
        self.assertEqual(res.center, (F(1),))
        self.assertEqual(res.radius, F(2))

    def test_fraction_exact(self):
        # [1/6, 1/2] 与 [2/3, 4/3]
        res = minimum_enclosing_ball_of_balls(
            [((F(1, 3),), F(1, 6)), ((F(1),), F(1, 3))]
        )
        # lo=1/6, hi=4/3 -> c=(1/6+4/3)/2 = 3/4, R=(4/3-1/6)/2 = 7/12
        self.assertEqual(res.center, (F(3, 4),))
        self.assertEqual(res.radius, F(7, 12))
        for s in res.support:
            self.assertIsInstance(s.weight, Fraction)

    def test_zero_radius_degenerate_to_points(self):
        # 零半径退化为点集：点 -3, 1, 4 -> c=0.5, R=3.5
        res = minimum_enclosing_ball_of_balls(
            [((-3,), 0), ((1,), 0), ((4,), 0)]
        )
        self.assertEqual(res.center, (F(1, 2),))
        self.assertEqual(res.radius, F(7, 2))
        self.assertEqual(len(res.support), 2)

    def test_single_ball(self):
        res = minimum_enclosing_ball_of_balls([((F(2, 3),), F(5, 2))])
        self.assertEqual(res.center, (F(2, 3),))
        self.assertEqual(res.radius, F(5, 2))
        self.assertEqual(res.num_input_balls, 1)


class TestHandComputed2D(unittest.TestCase):
    def test_zero_radius_right_triangle_is_diameter(self):
        # (0,0),(4,0),(0,3)：直角三角形，直径为斜边，c=(2,1.5), R=2.5
        res = minimum_enclosing_ball_of_balls(
            [((0, 0), 0), ((4, 0), 0), ((0, 3), 0)]
        )
        self.assertEqual(res.dimension, 2)
        self.assertAlmostEqual(res.center[0], 2.0, places=10)
        self.assertAlmostEqual(res.center[1], 1.5, places=10)
        self.assertAlmostEqual(res.radius, 2.5, places=10)
        self.assertEqual(len(res.support), 2)
        assert_kkt_certificate(self, res)

    def test_zero_radius_isosceles_circumcenter(self):
        # (0,0),(4,0),(2,3)：外心 (2, 5/6)，R=13/6
        balls = [((0, 0), 0), ((4, 0), 0), ((2, 3), 0)]
        res = minimum_enclosing_ball_of_balls(balls)
        self.assertAlmostEqual(res.center[0], 2.0, places=9)
        self.assertAlmostEqual(res.center[1], 5.0 / 6.0, places=9)
        self.assertAlmostEqual(res.radius, 13.0 / 6.0, places=9)
        self.assertEqual(len(res.support), 3)
        assert_kkt_certificate(self, res)

    def test_equal_positive_radii_add_to_point_enclosing_radius(self):
        # 同上三点，各球半径 1：R = 点集最小包围球半径 + 1 = 13/6+1 = 19/6
        balls = [((0, 0), 1), ((4, 0), 1), ((2, 3), 1)]
        res = minimum_enclosing_ball_of_balls(balls)
        self.assertAlmostEqual(res.center[0], 2.0, places=9)
        self.assertAlmostEqual(res.center[1], 5.0 / 6.0, places=9)
        self.assertAlmostEqual(res.radius, 19.0 / 6.0, places=9)
        _feasible(self, res, balls)
        assert_kkt_certificate(self, res)

    def test_unequal_triple_support_symmetric(self):
        # r=1 球心 (0,0),(6,0)；r=2 球心 (3,5)。对称解 c=(3, 9/4)，R=19/4。
        balls = [((0, 0), 1), ((6, 0), 1), ((3, 5), 2)]
        res = minimum_enclosing_ball_of_balls(balls)
        self.assertAlmostEqual(res.center[0], 3.0, places=9)
        self.assertAlmostEqual(res.center[1], 2.25, places=9)
        self.assertAlmostEqual(res.radius, 4.75, places=9)
        self.assertEqual(len(res.support), 3)
        # 逐约束的手算余量
        self.assertAlmostEqual(math.hypot(3.0, 2.25) + 1, 4.75, places=9)
        self.assertAlmostEqual(abs(5.0 - 2.25) + 2, 4.75, places=9)
        assert_kkt_certificate(self, res)

    def test_one_ball_covers_all(self):
        balls = [((0, 0), 5), ((1, 0), 1), ((0, 1), 1), ((2, 2), 0)]
        res = minimum_enclosing_ball_of_balls(balls)
        self.assertEqual(res.center, (0.0, 0.0))
        self.assertEqual(res.radius, 5.0)
        self.assertEqual(len(res.support), 1)
        assert_kkt_certificate(self, res)

    def test_pair_of_equal_balls(self):
        balls = [((0, 0), 1), ((4, 0), 1)]
        res = minimum_enclosing_ball_of_balls(balls)
        self.assertEqual(res.center, (2.0, 0.0))
        self.assertEqual(res.radius, 3.0)
        self.assertEqual(len(res.support), 2)
        assert_kkt_certificate(self, res)

    def test_nested_balls_2d(self):
        balls = [((0, 0), 10), ((1, 1), 1)]
        res = minimum_enclosing_ball_of_balls(balls)
        self.assertEqual(res.center, (0.0, 0.0))
        self.assertEqual(res.radius, 10.0)
        self.assertEqual(len(res.support), 1)

    def test_collinear_centers_use_pair(self):
        # 三球心共线：不可能产生 3 支撑，解由两端球确定
        balls = [((0, 0), 1), ((2, 0), 1), ((9, 0), 2)]
        res = minimum_enclosing_ball_of_balls(balls)
        # 两端外切：d=9, t=(9+2-1)/2=5 -> c=(5,0), R=6
        self.assertAlmostEqual(res.center[0], 5.0, places=9)
        self.assertAlmostEqual(res.center[1], 0.0, places=9)
        self.assertAlmostEqual(res.radius, 6.0, places=9)
        self.assertEqual(len(res.support), 2)
        assert_kkt_certificate(self, res)

    def test_concentric_different_radii(self):
        balls = [((1, 1), 3), ((1, 1), 1)]
        res = minimum_enclosing_ball_of_balls(balls)
        self.assertEqual(res.center, (1.0, 1.0))
        self.assertEqual(res.radius, 3.0)
        self.assertEqual(res.num_unique_balls, 2)
        self.assertEqual(len(res.support), 1)

    def test_tol_zero_exact_geometry(self):
        # 整数几何下 sqrt 恰好精确，tol=0 也应逐约束通过
        balls = [((0, 0), 0), ((3, 4), 0)]  # d=5, c=(1.5,2), R=2.5
        res = minimum_enclosing_ball_of_balls(balls, tol=0.0)
        self.assertAlmostEqual(res.radius, 2.5, places=12)
        self.assertLessEqual(res.max_violation(), 1e-12)

    def test_covers_and_max_violation(self):
        res = minimum_enclosing_ball_of_balls([((0, 0), 1), ((4, 0), 1)])
        self.assertTrue(res.covers(((2, 0), 2)))   # 内切于结果球
        self.assertTrue(res.covers(((2, 3), 0)))   # 边界点 dist=3
        self.assertFalse(res.covers(((2, 3.1), 0)))
        self.assertLessEqual(res.max_violation(), float(res.tol) + 1e-12)
        # max_violation 接受外部重算参数，并做校验
        with self.assertRaises(ValueError):
            res.max_violation([((0, 0), -1)])


# ---------------------------------------------------------------------------
# 独立参考 + KKT 的随机交叉核对（确定性种子）
# ---------------------------------------------------------------------------

class TestCrossCheckIndependentReference(unittest.TestCase):
    def test_deterministic_random_2d(self):
        rng = random.Random(20260924)
        for trial in range(120):
            n = rng.randint(1, 6)
            balls = []
            for _ in range(n):
                c = (rng.randint(-7, 7), rng.randint(-7, 7))
                r = rng.choice([0, 0, 0, 1, 2, 3, F(1, 2)])
                balls.append((c, r))
            with self.subTest(trial=trial, balls=balls):
                res = minimum_enclosing_ball_of_balls(balls)
                ref_c, ref_R = reference_minimum_ball_2d(balls)
                # 两个独立枚举必须给出（容差内）相同的最优半径
                self.assertLess(abs(float(res.radius) - ref_R), 1e-7)
                # 参考解与被测解的中心也应一致
                self.assertLess(
                    math.hypot(res.center[0] - ref_c[0],
                               res.center[1] - ref_c[1]),
                    1e-7,
                )
                _feasible(self, res, balls)
                assert_kkt_certificate(self, res)

    def test_deterministic_random_1d_matches_interval_arithmetic(self):
        rng = random.Random(4242)
        for trial in range(80):
            n = rng.randint(1, 6)
            balls = [
                (F(rng.randint(-6, 6)), F(rng.randint(0, 3))) for _ in range(n)
            ]
            balls = [((c,), r) for c, r in balls]
            with self.subTest(trial=trial):
                res = minimum_enclosing_ball_of_balls(balls)
                lo = min(c - r for (c,), r in balls)
                hi = max(c + r for (c,), r in balls)
                self.assertEqual(res.center, ((lo + hi) / 2,))
                self.assertEqual(res.radius, (hi - lo) / 2)
                # 支撑球确实取得 lo / hi
                ends = {(s.center[0] - s.radius) for s in res.support} | \
                       {(s.center[0] + s.radius) for s in res.support}
                self.assertIn(lo, ends)
                self.assertIn(hi, ends)


# ---------------------------------------------------------------------------
# 去重 / ID / 不可变
# ---------------------------------------------------------------------------

class TestDedupIdsAndImmutable(unittest.TestCase):
    def test_identical_balls_share_ids(self):
        balls = [((0, 0), 1), ((0, 0), 1), ((4, 0), 1)]
        res = minimum_enclosing_ball_of_balls(balls, ids=["a", "b", "c"])
        self.assertEqual(res.num_input_balls, 3)
        self.assertEqual(res.num_unique_balls, 2)
        groups = {s.center: tuple(s.ids) for s in res.support}
        self.assertEqual(groups[(F(0), F(0))], ("a", "b"))
        self.assertEqual(groups[(F(4), F(0))], ("c",))

    def test_same_center_diff_radius_not_deduped(self):
        balls = [((1, 1), 3), ((1, 1), 1)]
        res = minimum_enclosing_ball_of_balls(balls)
        self.assertEqual(res.num_unique_balls, 2)

    def test_input_not_mutated(self):
        balls = [([0, 0], 1), ([4, 0], 1)]
        ids = ["a", "b"]
        minimum_enclosing_ball_of_balls(balls, ids=ids)
        self.assertEqual(balls, [([0, 0], 1), ([4, 0], 1)])
        self.assertEqual(ids, ["a", "b"])

    def test_frozen(self):
        res = minimum_enclosing_ball_of_balls([((0, 0), 1)])
        with self.assertRaises(Exception):
            res.radius = 9
        self.assertIsInstance(res, MinimumEnclosingBall)
        self.assertIsInstance(res.support[0], BallSupport)
        self.assertEqual(res.center_float, (0.0, 0.0))
        self.assertEqual(res.radius_float, 1.0)

    def test_default_and_explicit_tol_recorded(self):
        res = minimum_enclosing_ball_of_balls([((0, 0), 1)])
        self.assertEqual(res.tol, miniball3d.DEFAULT_BALL_TOL)
        res2 = minimum_enclosing_ball_of_balls([((0, 0), 1)], tol=F(1, 1000))
        self.assertEqual(res2.tol, 0.001)


# ---------------------------------------------------------------------------
# 输入校验：全部在求解前失败，错误可定位
# ---------------------------------------------------------------------------

class TestBallInputValidation(unittest.TestCase):
    def assertRejects(self, fn, exc, *fragments):
        with self.assertRaises(exc) as cm:
            fn()
        msg = str(cm.exception)
        for frag in fragments:
            self.assertIn(frag, msg)

    def test_empty(self):
        self.assertRejects(lambda: minimum_enclosing_ball_of_balls([]),
                           ValueError, "at least 1")

    def test_too_many(self):
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls([((0, 0), 1)] * 7),
            ValueError, "at most 6", "got 7")

    def test_boundary_six_accepted(self):
        balls = [((i, 0), 0) for i in range(6)]
        res = minimum_enclosing_ball_of_balls(balls)
        self.assertEqual(res.num_input_balls, 6)
        self.assertAlmostEqual(res.radius, 2.5)

    def test_three_dimensional_rejected(self):
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls([((0, 0, 0), 1)]),
            ValueError, "balls[0][0]", "1D/2D")

    def test_dimension_mismatch_locatable(self):
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls(
                [((0, 0), 1), ((0,), 1)]),
            ValueError, "balls[1][0]", "expected 2 coordinate")

    def test_explicit_dimension(self):
        res = minimum_enclosing_ball_of_balls([((0, 0), 1)], dimension=2)
        self.assertEqual(res.dimension, 2)
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls(
                [((0, 0), 1)], dimension=3),
            ValueError, "dimension", "1 or 2")
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls(
                [((0, 0), 1)], dimension=True),
            TypeError, "bool")

    def test_negative_radius_locatable(self):
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls(
                [((0, 0), 1), ((1, 1), -2)]),
            ValueError, "balls[1][1]", "non-negative")

    def test_nan_inf_coordinates_and_radius(self):
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls(
                [((0, float("nan")), 1)]),
            ValueError, "balls[0][0][1]")
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls(
                [((float("-inf"), 0), 1)]),
            ValueError, "balls[0][0][0]")
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls(
                [((0, 0), float("inf"))]),
            ValueError, "balls[0][1]")

    def test_bool_rejected_everywhere(self):
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls([((True, 0), 1)]),
            TypeError, "balls[0][0][0]", "bool")
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls([((0, 0), False)]),
            TypeError, "balls[0][1]", "bool")

    def test_bad_types(self):
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls([("x", 1)]),
            TypeError, "balls[0][0]")
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls([((0, 0), "r")]),
            TypeError, "balls[0][1]")
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls("ab"),
            TypeError, "balls")
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls([((0, 0),)]),
            ValueError, "balls[0]", "length 1")

    def test_ids_errors_reference_balls(self):
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls(
                [((0, 0), 1)], ids=[1, 2]),
            ValueError, "ids", "number of balls")
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls(
                [((0, 0), 1), ((1, 0), 1)], ids=["x", "x"]),
            ValueError, "ids[1]")
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls(
                [((0, 0), 1)], ids=([1],)),
            TypeError, "ids[0]")

    def test_tol_validation(self):
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls(
                [((0, 0), 1)], tol=-1e-9),
            ValueError, "tol")
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls(
                [((0, 0), 1)], tol=float("nan")),
            ValueError, "tol")
        self.assertRejects(
            lambda: minimum_enclosing_ball_of_balls(
                [((0, 0), 1)], tol=True),
            TypeError, "tol", "bool")
        with self.assertRaises(TypeError):
            minimum_enclosing_ball_of_balls([((0, 0), 1)], tol="0.1")

    def test_finite_float_accepted_exactly(self):
        res = minimum_enclosing_ball_of_balls([((0.5, 0.0), 0.25)])
        self.assertEqual(res.support[0].center, (F(1, 2), F(0)))
        self.assertEqual(res.support[0].radius, F(1, 4))


# ---------------------------------------------------------------------------
# 旧接口不受影响
# ---------------------------------------------------------------------------

class TestLegacyEntryPointUnchanged(unittest.TestCase):
    def test_points_api_still_works(self):
        r = miniball3d.minimum_enclosing_sphere(
            [(0, 0, 0), (4, 0, 0), (0, 3, 0)]
        )
        self.assertEqual(r.radius_squared, F(25, 4))
        self.assertEqual(len(r.support), 2)

    def test_no_feasible_candidate_error_still_importable(self):
        self.assertTrue(issubclass(NoFeasibleCandidateError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
