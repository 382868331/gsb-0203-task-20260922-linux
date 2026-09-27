"""DynamicMiniball 增量 / 持久化行为的必要测试。

复用路径的正确性一律与全量批处理 minimum_enclosing_sphere 的精确结果
比对（半径平方必须相等；中心 / 证书按证书条件独立核验），并检查
“失败更新不改变点集、旧版本不受后续修改影响”的持久化契约。
"""

import unittest
from fractions import Fraction

from miniball3d import (
    MIN_BALL_MAX_POINTS,
    DynamicMiniball,
    EmptyPointSetError,
    minimum_enclosing_sphere,
)


def batch(points, ids):
    return minimum_enclosing_sphere(points, ids=ids)


def check_certificate(testcase, sphere, points):
    """对任意来源的球做完整证书核验（与批处理测试同一组充要条件）。"""
    testcase.assertEqual(sum(sphere.weights, Fraction(0)), 1)
    testcase.assertTrue(all(w >= 0 for w in sphere.weights))
    testcase.assertEqual(sphere.certificate_residual(), (0, 0, 0))
    testcase.assertEqual(
        sphere.dual_identity_radius_squared(), sphere.radius_squared
    )
    for s in sphere.support:
        testcase.assertEqual(
            sum((a - b) ** 2 for a, b in zip(s.point, sphere.center)),
            sphere.radius_squared,
        )
    for p in points:
        testcase.assertTrue(sphere.contains(p))


class DynamicMiniballTest(unittest.TestCase):
    def test_empty_sphere_rejected(self):
        dyn = DynamicMiniball()
        self.assertEqual(len(dyn), 0)
        with self.assertRaises(EmptyPointSetError):
            dyn.sphere()

    def test_from_scratch_matches_batch(self):
        dyn = DynamicMiniball()
        pts = [(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4)]
        for p in pts:
            dyn = dyn.insert(p)
        sphere = dyn.sphere()
        self.assertEqual(dyn.derivation, "recomputed")
        ref = batch(pts, list(range(4)))
        self.assertEqual(sphere.radius_squared, ref.radius_squared)
        self.assertEqual(sphere.center, ref.center)
        check_certificate(self, sphere, pts)

    def test_insert_inside_reuses_and_stays_correct(self):
        corners = [(x, y, z) for x in (0, 4) for y in (0, 4) for z in (0, 4)]
        dyn = DynamicMiniball()
        for p in corners:
            dyn = dyn.insert(p)
        base = dyn.sphere()
        inside = (1, 1, 1)  # 严格在旧球内部
        dyn2 = dyn.insert(inside)
        sphere2 = dyn2.sphere()
        self.assertEqual(dyn2.derivation, "insert-contained")
        self.assertEqual(sphere2.center, base.center)
        self.assertEqual(sphere2.radius_squared, base.radius_squared)
        self.assertEqual(sphere2.num_input_points, 9)
        ref = batch(corners + [inside], list(range(9)))
        self.assertEqual(sphere2.radius_squared, ref.radius_squared)
        check_certificate(self, sphere2, corners + [inside])

    def test_insert_outside_recomputes(self):
        dyn = DynamicMiniball()
        pts = [(0, 0, 0), (2, 0, 0)]
        for p in pts:
            dyn = dyn.insert(p)
        dyn2 = dyn.insert((10, 0, 0))
        sphere2 = dyn2.sphere()
        self.assertEqual(dyn2.derivation, "recomputed")
        ref = batch(pts + [(10, 0, 0)], [0, 1, 2])
        self.assertEqual(sphere2.radius_squared, ref.radius_squared)
        self.assertEqual(sphere2.center, ref.center)

    def test_delete_non_support_reuses(self):
        corners = [(x, y, z) for x in (0, 4) for y in (0, 4) for z in (0, 4)]
        dyn = DynamicMiniball()
        for p in corners:
            dyn = dyn.insert(p)
        dyn = dyn.insert((1, 1, 1), pid="inner")  # 非支撑点
        dyn.sphere()
        dyn2 = dyn.delete("inner")
        sphere2 = dyn2.sphere()
        self.assertEqual(dyn2.derivation, "delete-non-support")
        ref = batch(corners, list(range(8)))
        self.assertEqual(sphere2.radius_squared, ref.radius_squared)
        check_certificate(self, sphere2, corners)

    def test_delete_support_recomputes_not_kept(self):
        # 两点球：任一点都是支撑，删除后必须重算，绝不允许保留旧球。
        dyn = DynamicMiniball()
        dyn = dyn.insert((0, 0, 0)).insert((6, 0, 0))
        old = dyn.sphere()
        dyn2 = dyn.delete(1)
        sphere2 = dyn2.sphere()
        self.assertEqual(dyn2.derivation, "recomputed")
        self.assertEqual(sphere2.radius_squared, Fraction(0))
        self.assertNotEqual(sphere2.radius_squared, old.radius_squared)
        check_certificate(self, sphere2, [(0, 0, 0)])

    def test_delete_support_coordinate_with_duplicate_ids_reuses(self):
        # 支撑坐标仍有其他 ID 留存时，证书依旧有效，允许复用。
        dyn = DynamicMiniball()
        dyn = dyn.insert((0, 0, 0), pid="a").insert((0, 0, 0), pid="b")
        dyn = dyn.insert((4, 0, 0), pid="c")
        dyn.sphere()
        dyn2 = dyn.delete("a")
        sphere2 = dyn2.sphere()
        self.assertEqual(dyn2.derivation, "delete-non-support")
        ref = batch([(0, 0, 0), (4, 0, 0)], ["b", "c"])
        self.assertEqual(sphere2.radius_squared, ref.radius_squared)
        id_sets = {tuple(s.ids) for s in sphere2.support}
        self.assertIn(("b",), id_sets)

    def test_old_versions_are_persistent(self):
        dyn = DynamicMiniball()
        dyn = dyn.insert((0, 0, 0)).insert((6, 0, 0))
        old_sphere = dyn.sphere()
        old_ids = dyn.ids
        dyn2 = dyn.insert((100, 0, 0)).delete(0)
        dyn2.sphere()
        # 旧版本点集与结果均不受后续修改影响
        self.assertEqual(dyn.ids, old_ids)
        self.assertEqual(len(dyn), 2)
        again = dyn.sphere()
        self.assertEqual(again.center, old_sphere.center)
        self.assertEqual(again.radius_squared, old_sphere.radius_squared)

    def test_failed_insert_leaves_set_unchanged(self):
        dyn = DynamicMiniball().insert((0, 0, 0)).insert((1, 1, 1))
        before = dyn.ids
        for bad in (
            lambda: dyn.insert((float("nan"), 0, 0)),
            lambda: dyn.insert((1, 2)),            # 坐标数错误
            lambda: dyn.insert((True, 0, 0)),      # bool 坐标
            lambda: dyn.insert((9, 9, 9), pid=0),  # 重复 ID
            lambda: dyn.insert((9, 9, 9), pid=[]), # 不可哈希 ID
        ):
            with self.assertRaises((TypeError, ValueError)):
                bad()
        self.assertEqual(dyn.ids, before)
        self.assertEqual(len(dyn), 2)

    def test_failed_delete_leaves_set_unchanged(self):
        dyn = DynamicMiniball().insert((0, 0, 0))
        with self.assertRaises(KeyError):
            dyn.delete("missing")
        with self.assertRaises(TypeError):
            dyn.delete(["unhashable"])
        self.assertEqual(len(dyn), 1)

    def test_point_limit_enforced_on_insert(self):
        dyn = DynamicMiniball()
        for i in range(MIN_BALL_MAX_POINTS):
            dyn = dyn.insert((i, 0, 0))
        with self.assertRaises(ValueError):
            dyn.insert((999, 0, 0))
        self.assertEqual(len(dyn), MIN_BALL_MAX_POINTS)

    def test_delete_to_empty_then_reinsert(self):
        dyn = DynamicMiniball().insert((3, 4, 5), pid="x")
        dyn = dyn.delete("x")
        with self.assertRaises(EmptyPointSetError):
            dyn.sphere()
        dyn = dyn.insert((1, 2, 3), pid="y")
        sphere = dyn.sphere()
        self.assertEqual(sphere.radius_squared, Fraction(0))
        self.assertEqual(sphere.center, (Fraction(1), Fraction(2), Fraction(3)))

    def test_auto_ids_are_smallest_free_nonnegative_int(self):
        dyn = DynamicMiniball()
        dyn = dyn.insert((0, 0, 0)).insert((1, 0, 0)).insert((2, 0, 0))
        dyn = dyn.delete(1)
        dyn = dyn.insert((3, 0, 0))
        self.assertEqual(dyn.ids, (0, 2, 1))

    def test_mixed_sequence_stays_correct_vs_batch(self):
        # 确定性伪随机序列（线性同余，无随机库），交替插入 / 删除，
        # 每个版本都与全量批处理结果比对 r^2 并核验证书。
        state = 12345

        def rnd(mod):
            nonlocal state
            state = (state * 1103515245 + 12345) % (2 ** 31)
            return state % mod

        dyn = DynamicMiniball()
        live = []  # (pid, point)
        next_pid = 0
        for step in range(40):
            if not live or rnd(3) != 0:
                p = (rnd(9) - 4, rnd(9) - 4, rnd(9) - 4)
                dyn = dyn.insert(p, pid=next_pid)
                live.append((next_pid, p))
                next_pid += 1
            else:
                victim = live.pop(rnd(len(live)))
                dyn = dyn.delete(victim[0])
            if live:
                sphere = dyn.sphere()
                pts = [p for _, p in live]
                ids = [i for i, _ in live]
                ref = batch(pts, ids)
                self.assertEqual(
                    sphere.radius_squared, ref.radius_squared,
                    msg=f"step {step}, derivation={dyn.derivation}",
                )
                check_certificate(self, sphere, pts)


if __name__ == "__main__":
    unittest.main()
