"""trimmed_ball 的必要测试。

覆盖：k=0 与原接口一致、对称平局按被排除 ID 序列裁决、重复离群位置
（删一个副本不等于删全部）、共线点、全部点相同、剩余集合非空、
预算耗尽返回当前可行球 + 未完成状态、证书现场核验。
"""

from fractions import Fraction
import unittest

from miniball3d import (
    TrimmedBall,
    minimum_enclosing_sphere,
    trimmed_ball,
)

F = Fraction


def certificate_ok(ball, kept_points):
    """独立核验保留点集上的最优性证书（全部精确 Fraction）。"""
    weights = ball.weights
    if not weights or any(w < 0 for w in weights):
        return False
    if sum(weights, F(0)) != 1:
        return False
    if ball.certificate_residual() != (F(0), F(0), F(0)):
        return False
    for s in ball.support:
        d2 = sum((s.point[d] - ball.center[d]) ** 2 for d in range(3))
        if d2 != ball.radius_squared:
            return False
    if any(not ball.contains(p) for p in kept_points):
        return False
    return ball.dual_identity_radius_squared() == ball.radius_squared


class TrimmedBallBasicTests(unittest.TestCase):
    def test_k0_matches_plain_interface(self):
        points = [(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4), (1, 1, 1)]
        ids = ["a", "b", "c", "d", "e"]
        plain = minimum_enclosing_sphere(points, ids=ids)
        trimmed = trimmed_ball(points, 0, ids=ids)
        self.assertTrue(trimmed.complete)
        self.assertEqual(trimmed.excluded_ids, ())
        self.assertEqual(trimmed.ball, plain)
        self.assertEqual(trimmed.radius_squared, plain.radius_squared)
        self.assertEqual(trimmed.center, plain.center)

    def test_isolated_outlier_excluded(self):
        points = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0), (100, 0, 0)]
        result = trimmed_ball(points, 1)
        self.assertTrue(result.complete)
        self.assertEqual(result.excluded_ids, (4,))
        kept = [points[i] for i in range(5) if i != 4]
        self.assertEqual(result.ball, minimum_enclosing_sphere(kept))
        self.assertTrue(certificate_ok(result.ball, kept))

    def test_symmetric_tie_broken_by_id_sequence(self):
        # 两个对称离群点：排除 (0,) 或 (4,) 得到相同半径与排除数量，
        # 按被排除 ID 序列字典序取 (0,)。
        points = [(-10, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, 0), (10, 0, 0)]
        result = trimmed_ball(points, 1)
        self.assertTrue(result.complete)
        self.assertEqual(result.excluded_ids, (0,))
        # 排除任一端点后，剩余最远对为 (0,1,0)-(±10,0,0)，直径平方 101。
        self.assertEqual(result.radius_squared, F(101, 4))
        # 自定义 ID 时同样按 ID 序列裁决。
        ids = ["L", "p", "q", "r", "R"]
        result_ids = trimmed_ball(points, 1, ids=ids)
        self.assertEqual(result_ids.excluded_ids, ("L",))

    def test_duplicate_outlier_positions_are_distinct_observations(self):
        # 同一离群位置的两个副本是不同观测：k=1 最多删一个副本，
        # 另一个副本仍把半径撑大；k=2 才能同时删掉两个。
        points = [(0, 0, 0)] * 3 + [(50, 50, 50)] * 2
        one = trimmed_ball(points, 1)
        self.assertTrue(one.complete)
        # k=1 删不掉同一位置的两个副本：任何单点排除后两个位置都还在，
        # 半径不变；目标先比半径再比排除数量，因此最优是不排除。
        self.assertEqual(one.excluded_ids, ())
        # 直径球 r^2 = |(50,50,50)|^2 / 4 = 1875，离群位置仍在球面上。
        self.assertEqual(one.radius_squared, F(1875))
        outlier_on_sphere = sum(
            (F(50) - one.center[d]) ** 2 for d in range(3)
        )
        self.assertEqual(outlier_on_sphere, one.radius_squared)
        two = trimmed_ball(points, 2)
        self.assertTrue(two.complete)
        self.assertEqual(two.excluded_ids, (3, 4))
        self.assertEqual(two.radius_squared, F(0))
        self.assertLess(two.radius_squared, one.radius_squared)

    def test_collinear_points(self):
        points = [(i, 0, 0) for i in range(6)] + [(100, 0, 0)]
        result = trimmed_ball(points, 1)
        self.assertTrue(result.complete)
        self.assertEqual(result.excluded_ids, (6,))
        self.assertEqual(result.radius_squared, F(25, 4))
        self.assertEqual(result.center, (F(5, 2), F(0), F(0)))

    def test_all_points_identical(self):
        points = [(3, 3, 3)] * 5
        result = trimmed_ball(points, 3)
        self.assertTrue(result.complete)
        # 半径为 0 平局时排除更少优先 -> 不排除任何点。
        self.assertEqual(result.excluded_ids, ())
        self.assertEqual(result.radius_squared, F(0))
        self.assertEqual(result.center, (F(3), F(3), F(3)))

    def test_remaining_set_never_empty(self):
        result = trimmed_ball([(7, 8, 9)], 3)
        self.assertTrue(result.complete)
        self.assertEqual(result.excluded_ids, ())
        self.assertEqual(result.radius_squared, F(0))

    def test_budget_exhaustion_returns_feasible_incomplete(self):
        points = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0), (100, 0, 0)]
        result = trimmed_ball(points, 1, budget=1)
        self.assertFalse(result.complete)
        self.assertEqual(result.candidates_examined, 1)
        # 当前可行球：不排除任何点，仍须精确包围全部输入点。
        self.assertEqual(result.excluded_ids, ())
        self.assertTrue(all(result.ball.contains(p) for p in points))
        # 给足预算则完成且排除离群点。
        full = trimmed_ball(points, 1)
        self.assertTrue(full.complete)
        self.assertEqual(full.excluded_ids, (4,))

    def test_excluded_ids_use_original_ids(self):
        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (9, 9, 9)]
        ids = ["p0", "p1", "p2", "outlier"]
        result = trimmed_ball(points, 1, ids=ids)
        self.assertEqual(result.excluded_ids, ("outlier",))
        support_ids = {i for s in result.ball.support for i in s.ids}
        self.assertNotIn("outlier", support_ids)

    def test_result_is_frozen(self):
        result = trimmed_ball([(0, 0, 0), (1, 0, 0)], 1)
        self.assertIsInstance(result, TrimmedBall)
        with self.assertRaises(Exception):
            result.complete = False


class TrimmedBallValidationTests(unittest.TestCase):
    def test_k_out_of_range(self):
        for bad in (-1, 4, 100):
            with self.assertRaises(ValueError):
                trimmed_ball([(0, 0, 0)], bad)

    def test_k_bool_and_type(self):
        with self.assertRaises(TypeError):
            trimmed_ball([(0, 0, 0)], True)
        with self.assertRaises(TypeError):
            trimmed_ball([(0, 0, 0)], 1.5)
        with self.assertRaises(TypeError):
            trimmed_ball([(0, 0, 0)], "1")

    def test_points_validation_still_applies(self):
        with self.assertRaises(ValueError):
            trimmed_ball([], 1)
        with self.assertRaises(ValueError):
            trimmed_ball([(0, 0, 0), (1, float("nan"), 0)], 1)
        with self.assertRaises(ValueError):
            trimmed_ball([(0, 0, 0)] * 31, 1)

    def test_budget_invalid(self):
        with self.assertRaises(ValueError):
            trimmed_ball([(0, 0, 0)], 1, budget=0)
        with self.assertRaises(TypeError):
            trimmed_ball([(0, 0, 0)], 1, budget=True)


if __name__ == "__main__":
    unittest.main()
