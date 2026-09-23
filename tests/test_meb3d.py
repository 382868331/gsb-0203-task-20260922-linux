"""meb3d 模块的单元测试。

参考实现 _reference_radius_squared 是在本测试文件内独立编写的暴力
枚举（前向消元 + 回代），不调用被测核心 meb3d 的任何函数；被测结果
与参考半径平方必须精确相等。此外用对偶恒等式（不依赖任何搜索算法）
独立核验最优性；meb3d.verify_certificate 的往返一致仅作补充。
"""

import random
import sys
import unittest
from fractions import Fraction
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import meb3d
from meb3d import (
    BallCertificate,
    InvalidInputError,
    SupportPoint,
    min_enclosing_ball,
    verify_certificate,
)


# ---------------------------------------------------------------------------
# 独立参考实现（不导入被测核心）
# ---------------------------------------------------------------------------


def _ref_solve(matrix, rhs):
    """前向消元 + 回代求精确解；奇异返回 None。"""
    n = len(matrix)
    a = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if a[r][col] != 0), None)
        if piv is None:
            return None
        a[col], a[piv] = a[piv], a[col]
        for r in range(col + 1, n):
            if a[r][col] != 0:
                f = a[r][col] / a[col][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        x[i] = (a[i][n] - sum(a[i][j] * x[j] for j in range(i + 1, n))) / a[i][i]
    return x


def _reference_radius_squared(raw_points):
    """独立暴力枚举：返回最小包围球半径平方（Fraction）。"""
    pts = []
    seen = set()
    for p in raw_points:
        q = tuple(Fraction(v) for v in p)
        if q not in seen:
            seen.add(q)
            pts.append(q)
    n = len(pts)
    best = None
    for k in range(1, min(4, n) + 1):
        for subset in combinations(range(n), k):
            if k == 1:
                center = pts[subset[0]]
            else:
                p0 = pts[subset[0]]
                matrix = [[Fraction(1)] * k]
                rhs = [Fraction(1)]
                for i in subset[1:]:
                    pi = pts[i]
                    matrix.append(
                        [
                            2 * sum(
                                (pi[a] - p0[a]) * pts[j][a] for a in range(3)
                            )
                            for j in subset
                        ]
                    )
                    rhs.append(
                        sum(v * v for v in pi) - sum(v * v for v in p0)
                    )
                lam = _ref_solve(matrix, rhs)
                if lam is None or any(w < 0 for w in lam):
                    continue
                center = tuple(
                    sum(lam[j] * pts[subset[j]][a] for j in range(k))
                    for a in range(3)
                )
            r2 = sum((center[a] - pts[subset[0]][a]) ** 2 for a in range(3))
            if best is not None and r2 >= best:
                continue
            if all(
                sum((center[a] - q[a]) ** 2 for a in range(3)) <= r2
                for q in pts
            ):
                best = r2
    return best


def _dual_objective(support_coords, weights):
    """对偶目标 g(λ) = Σλᵢ|sᵢ|² − |Σλᵢsᵢ|²，最优性的独立判据。"""
    combo = [
        sum(w * s[a] for s, w in zip(support_coords, weights))
        for a in range(3)
    ]
    return sum(
        w * sum(c * c for c in s) for s, w in zip(support_coords, weights)
    ) - sum(c * c for c in combo)


def _random_points(rng, count, lo=-6, hi=6):
    return [
        (rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi))
        for _ in range(count)
    ]


# ---------------------------------------------------------------------------
# 正确性：退化与特殊构型
# ---------------------------------------------------------------------------


class TestDegenerateAndSpecial(unittest.TestCase):
    def test_single_point(self):
        cert = min_enclosing_ball([(3, -2, 7)])
        self.assertEqual(cert.center, (Fraction(3), Fraction(-2), Fraction(7)))
        self.assertEqual(cert.radius_squared, 0)
        self.assertEqual(len(cert.support), 1)
        self.assertEqual(cert.support[0].weight, 1)
        self.assertEqual(cert.support[0].ids, (0,))

    def test_two_points_diameter_ball(self):
        cert = min_enclosing_ball([(0, 0, 0), (4, 0, 0)])
        self.assertEqual(cert.center, (Fraction(2), 0, 0))
        self.assertEqual(cert.radius_squared, 4)
        self.assertEqual(len(cert.support), 2)
        self.assertTrue(all(sp.weight == Fraction(1, 2) for sp in cert.support))

    def test_collinear_points(self):
        pts = [(0, 0, 0), (1, 0, 0), (3, 0, 0), (-2, 0, 0)]
        cert = min_enclosing_ball(pts)
        self.assertEqual(cert.center, (Fraction(1, 2), 0, 0))
        self.assertEqual(cert.radius_squared, Fraction(25, 4))
        self.assertEqual(len(cert.support), 2)
        self.assertEqual(cert.radius_squared, _reference_radius_squared(pts))

    def test_coplanar_points(self):
        pts = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (2, 1, 0)]
        cert = min_enclosing_ball(pts)
        self.assertEqual(cert.radius_squared, _reference_radius_squared(pts))
        self.assertEqual(cert.radius_squared, Fraction(5, 2))
        self.assertTrue(verify_certificate(pts, cert))

    def test_obtuse_triangle_uses_longest_edge(self):
        # 边长平方 16, 2, 10：16 > 2 + 10，钝角，包围球由最长边直径决定。
        pts = [(0, 0, 0), (4, 0, 0), (1, 1, 0)]
        cert = min_enclosing_ball(pts)
        self.assertEqual(cert.center, (Fraction(2), 0, 0))
        self.assertEqual(cert.radius_squared, 4)
        self.assertEqual(len(cert.support), 2)
        self.assertEqual(cert.radius_squared, _reference_radius_squared(pts))

    def test_tetrahedron_four_supports(self):
        # 正四面体，球心在原点，四个顶点等权。
        pts = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        cert = min_enclosing_ball(pts)
        self.assertEqual(cert.center, (Fraction(0),) * 3)
        self.assertEqual(cert.radius_squared, 3)
        self.assertEqual(len(cert.support), 4)
        self.assertTrue(all(sp.weight == Fraction(1, 4) for sp in cert.support))

    def test_duplicate_points_keep_id_provenance(self):
        pts = [(0, 0, 0), (2, 0, 0), (2, 0, 0), (0, 0, 0)]
        cert = min_enclosing_ball(pts, ids=["a", "b", "c", "d"])
        self.assertEqual(cert.radius_squared, 1)
        self.assertEqual(cert.center, (Fraction(1), 0, 0))
        id_map = {sp.coordinates: sp.ids for sp in cert.support}
        self.assertEqual(id_map[(Fraction(0),) * 3], ("a", "d"))
        self.assertEqual(id_map[(Fraction(2), 0, 0)], ("b", "c"))

    def test_all_points_identical(self):
        pts = [(5, 5, 5)] * 6
        cert = min_enclosing_ball(pts)
        self.assertEqual(cert.radius_squared, 0)
        self.assertEqual(len(cert.support), 1)
        self.assertEqual(cert.support[0].ids, tuple(range(6)))

    def test_fraction_and_mixed_numeric_inputs(self):
        pts = [
            (Fraction(1, 3), 0, 0),
            (0, Fraction(-2, 7), 0),
            (0, 0, 1.5),  # 有限 float 被精确转换
            (2, 1, -1),
        ]
        cert = min_enclosing_ball(pts)
        self.assertEqual(cert.radius_squared, _reference_radius_squared(pts))
        self.assertTrue(verify_certificate(pts, cert))


# ---------------------------------------------------------------------------
# 正确性：随机小规模（<= 8 点）对照独立参考 + 对偶恒等式
# ---------------------------------------------------------------------------


class TestAgainstIndependentReference(unittest.TestCase):
    def test_random_small_sets_match_reference(self):
        rng = random.Random(20260923)
        for trial in range(60):
            n = rng.randint(1, 8)
            pts = _random_points(rng, n)
            cert = min_enclosing_ball(pts)
            ref = _reference_radius_squared(pts)
            self.assertEqual(
                cert.radius_squared, ref, f"trial {trial}, points={pts}"
            )

    def test_random_rational_sets_match_reference(self):
        rng = random.Random(71077345)
        for trial in range(30):
            n = rng.randint(2, 8)
            pts = [
                tuple(Fraction(rng.randint(-12, 12), rng.randint(1, 5))
                      for _ in range(3))
                for _ in range(n)
            ]
            cert = min_enclosing_ball(pts)
            self.assertEqual(
                cert.radius_squared, _reference_radius_squared(pts),
                f"trial {trial}, points={pts}",
            )

    def test_optimality_via_duality_identity(self):
        # 独立最优性证明：原始可行（包含所有点）+ λ 在对偶单纯形上
        # + 对偶目标值等于半径平方 ⟹ 不存在更小的包围球。
        rng = random.Random(999331)
        for trial in range(40):
            n = rng.randint(1, 8)
            pts = _random_points(rng, n)
            cert = min_enclosing_ball(pts)
            weights = [sp.weight for sp in cert.support]
            self.assertTrue(all(w >= 0 for w in weights))
            self.assertEqual(sum(weights), 1)
            coords = [sp.coordinates for sp in cert.support]
            self.assertEqual(_dual_objective(coords, weights),
                             cert.radius_squared)
            for q in pts:
                d2 = sum(
                    (Fraction(q[a]) - cert.center[a]) ** 2 for a in range(3)
                )
                self.assertLessEqual(d2, cert.radius_squared)
            # 往返一致（补充）：模块内核验器也通过。
            self.assertTrue(verify_certificate(pts, cert))

    def test_support_is_affinely_independent_and_sized(self):
        rng = random.Random(31337)
        for _ in range(40):
            pts = _random_points(rng, rng.randint(1, 8))
            cert = min_enclosing_ball(pts)
            k = len(cert.support)
            self.assertGreaterEqual(k, 1)
            self.assertLessEqual(k, 4)
            if k >= 2:
                base = cert.support[0].coordinates
                # 仿射独立 ⟺ 差向量线性无关：检查格拉姆行列式非零。
                diffs = [
                    [sp.coordinates[a] - base[a] for a in range(3)]
                    for sp in cert.support[1:]
                ]
                gram = [
                    [sum(diffs[i][a] * diffs[j][a] for a in range(3))
                     for j in range(len(diffs))]
                    for i in range(len(diffs))
                ]
                self.assertIsNotNone(_ref_solve(gram, [Fraction(1)] * len(diffs)))


# ---------------------------------------------------------------------------
# 输入校验与错误语义
# ---------------------------------------------------------------------------


class TestInputValidation(unittest.TestCase):
    def test_rejects_nan(self):
        with self.assertRaises(InvalidInputError) as cm:
            min_enclosing_ball([(0, 0, 0), (float("nan"), 1, 2)])
        self.assertIn("points[1][0]", str(cm.exception))

    def test_rejects_infinity(self):
        for bad in (float("inf"), float("-inf")):
            with self.assertRaises(InvalidInputError):
                min_enclosing_ball([(0, bad, 0)])

    def test_rejects_bool_coordinate(self):
        with self.assertRaises(InvalidInputError) as cm:
            min_enclosing_ball([(True, 0, 0)])
        self.assertIn("points[0][0]", str(cm.exception))

    def test_rejects_wrong_dimension(self):
        with self.assertRaises(InvalidInputError) as cm:
            min_enclosing_ball([(1, 2), (3, 4, 5)])
        self.assertIn("points[0]", str(cm.exception))
        with self.assertRaises(InvalidInputError):
            min_enclosing_ball([(1, 2, 3, 4)])

    def test_rejects_unsupported_coordinate_type(self):
        with self.assertRaises(InvalidInputError):
            min_enclosing_ball([("1", 0, 0)])
        with self.assertRaises(InvalidInputError):
            min_enclosing_ball([(1 + 2j, 0, 0)])

    def test_rejects_empty_and_oversize(self):
        with self.assertRaises(InvalidInputError):
            min_enclosing_ball([])
        with self.assertRaises(InvalidInputError) as cm:
            min_enclosing_ball([(i, 0, 0) for i in range(31)])
        self.assertIn("30", str(cm.exception))

    def test_accepts_exactly_max_points(self):
        pts = [(i, (i * 7) % 5, (i * 13) % 3) for i in range(30)]
        cert = min_enclosing_ball(pts)
        self.assertTrue(verify_certificate(pts, cert))

    def test_rejects_bad_ids(self):
        with self.assertRaises(InvalidInputError):
            min_enclosing_ball([(0, 0, 0)], ids=[1, 2])
        with self.assertRaises(InvalidInputError):
            min_enclosing_ball([(0, 0, 0), (1, 0, 0)], ids=["x", "x"])
        with self.assertRaises(InvalidInputError) as cm:
            min_enclosing_ball([(0, 0, 0)], ids=[[1]])
        self.assertIn("ids[0]", str(cm.exception))

    def test_rejects_non_iterable_and_string_points(self):
        with self.assertRaises(InvalidInputError):
            min_enclosing_ball(42)
        with self.assertRaises(InvalidInputError):
            min_enclosing_ball("not points")

    def test_verify_rejects_forged_certificate(self):
        pts = [(0, 0, 0), (4, 0, 0), (1, 1, 0)]
        good = min_enclosing_ball(pts)
        # 半径被改小：包含性被破坏。
        forged = BallCertificate(
            center=good.center,
            radius_squared=good.radius_squared - 1,
            support=good.support,
        )
        with self.assertRaises(meb3d.CertificateVerificationError):
            verify_certificate(pts, forged)
        # 权重被篡改：凸包组合不变式被破坏。
        bad_support = (
            SupportPoint(ids=(0,), coordinates=(Fraction(0),) * 3,
                         weight=Fraction(2)),
            SupportPoint(ids=(1,), coordinates=(Fraction(4), 0, 0),
                         weight=Fraction(-1)),
        )
        forged2 = BallCertificate(
            center=good.center,
            radius_squared=good.radius_squared,
            support=bad_support,
        )
        with self.assertRaises(meb3d.CertificateVerificationError):
            verify_certificate(pts, forged2)


if __name__ == "__main__":
    unittest.main()
