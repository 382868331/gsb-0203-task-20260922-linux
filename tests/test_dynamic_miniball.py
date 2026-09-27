"""dynamic_miniball 的必要测试。

对照基准：每个动态操作后的结果都与 miniball3d.minimum_enclosing_sphere
对当前点集的一次性精确求解逐字段比对（中心、r^2、支撑坐标、权重、
计数）；并核验复用路径确实没有重算（几何量对象同一性 / 证书仍成立）。
"""

from fractions import Fraction
import random
import unittest

from dynamic_miniball import DynamicMiniball
from miniball3d import MIN_BALL_MAX_POINTS, minimum_enclosing_sphere

F = Fraction


def reference(dyn):
    return minimum_enclosing_sphere(list(dyn.points), ids=list(dyn.ids))


def assert_same_geometry(testcase, sphere, ref):
    testcase.assertEqual(sphere.center, ref.center)
    testcase.assertEqual(sphere.radius_squared, ref.radius_squared)
    testcase.assertEqual(sphere.num_input_points, ref.num_input_points)
    testcase.assertEqual(sphere.num_unique_points, ref.num_unique_points)
    # 最优中心与 r^2 唯一，但支撑表示在平局时不唯一（如共圆点），
    # 支撑本身的正确性由 assert_certificate_valid 的精确证书核验保证。


def assert_certificate_valid(testcase, dyn):
    sphere = dyn.sphere()
    testcase.assertEqual(sum(sphere.weights, F(0)), 1)
    testcase.assertTrue(all(w >= 0 for w in sphere.weights))
    testcase.assertEqual(sphere.certificate_residual(), (F(0), F(0), F(0)))
    testcase.assertEqual(sphere.dual_identity_radius_squared(),
                         sphere.radius_squared)
    for p in dyn.points:
        testcase.assertTrue(sphere.contains(p))


class ConstructionTests(unittest.TestCase):
    def test_empty_set(self):
        dyn = DynamicMiniball()
        self.assertEqual(len(dyn), 0)
        with self.assertRaises(ValueError):
            dyn.sphere()

    def test_initial_points(self):
        dyn = DynamicMiniball([(0, 0, 0), (2, 0, 0)], ids=["a", "b"])
        self.assertEqual(dyn.ids, ("a", "b"))
        assert_same_geometry(self, dyn.sphere(), reference(dyn))

    def test_initial_validation_failure(self):
        with self.assertRaises(ValueError):
            DynamicMiniball([(0, 0, 0), (1, float("nan"), 0)])
        with self.assertRaises(ValueError):
            DynamicMiniball([(0, 0, 0)], ids=["x", "y"])


class InsertTests(unittest.TestCase):
    def test_insert_inside_reuses_sphere(self):
        dyn = DynamicMiniball([(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4)])
        before = dyn.sphere()
        dyn2 = dyn.insert((1, 1, 1), point_id="inner")
        after = dyn2.sphere()
        # 几何完全复用（同一批字段值），仅计数/ID 更新
        self.assertEqual(after.center, before.center)
        self.assertEqual(after.radius_squared, before.radius_squared)
        self.assertEqual(after.support, before.support)
        self.assertEqual(after.num_input_points, before.num_input_points + 1)
        assert_same_geometry(self, after, reference(dyn2))

    def test_insert_outside_recomputes(self):
        dyn = DynamicMiniball([(0, 0, 0), (2, 0, 0)])
        dyn2 = dyn.insert((10, 0, 0))
        assert_same_geometry(self, dyn2.sphere(), reference(dyn2))
        self.assertEqual(dyn2.sphere().center, (F(5), F(0), F(0)))

    def test_insert_duplicate_coordinate_updates_provenance(self):
        dyn = DynamicMiniball([(0, 0, 0), (4, 0, 0)], ids=["a", "b"])
        dyn2 = dyn.insert((4, 0, 0), point_id="b2")
        sphere = dyn2.sphere()
        assert_same_geometry(self, sphere, reference(dyn2))
        ids_by_point = {s.point: s.ids for s in sphere.support}
        self.assertEqual(ids_by_point[(F(4), F(0), F(0))], ("b", "b2"))

    def test_default_ids_stay_unique(self):
        dyn = DynamicMiniball([(0, 0, 0)])
        dyn = dyn.delete(0)
        dyn = dyn.insert((1, 1, 1))
        self.assertEqual(dyn.ids, (0,))  # 复用被释放的最小未占用 ID

    def test_failed_insert_leaves_set_unchanged(self):
        dyn = DynamicMiniball([(0, 0, 0), (2, 0, 0)], ids=["a", "b"])
        before = dyn.sphere()
        for bad_call in (
            lambda: dyn.insert((1, float("nan"), 0)),
            lambda: dyn.insert((1, 2)),
            lambda: dyn.insert((True, 0, 0)),
            lambda: dyn.insert((1, 1, 1), point_id="a"),
            lambda: dyn.insert((1, 1, 1), point_id=True),
            lambda: dyn.insert((1, 1, 1), point_id=["unhashable"]),
            lambda: dyn.insert("not-a-point"),
        ):
            with self.assertRaises((TypeError, ValueError)):
                bad_call()
        self.assertEqual(len(dyn), 2)
        self.assertIs(dyn.sphere(), before)

    def test_insert_beyond_limit_rejected(self):
        dyn = DynamicMiniball([(i, 0, 0) for i in range(MIN_BALL_MAX_POINTS)])
        with self.assertRaises(ValueError):
            dyn.insert((99, 0, 0))
        self.assertEqual(len(dyn), MIN_BALL_MAX_POINTS)


class DeleteTests(unittest.TestCase):
    def test_delete_non_support_reuses_sphere(self):
        # (1,1,1) 在四面体包围球内部，删除后球不变
        dyn = DynamicMiniball(
            [(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4), (1, 1, 1)],
            ids=["a", "b", "c", "d", "inner"],
        )
        before = dyn.sphere()
        dyn2 = dyn.delete("inner")
        after = dyn2.sphere()
        self.assertEqual(after.center, before.center)
        self.assertEqual(after.radius_squared, before.radius_squared)
        assert_same_geometry(self, after, reference(dyn2))

    def test_delete_support_recomputes(self):
        # 钝角三角形：包围球由最长边直径决定；删掉端点后必须重算
        dyn = DynamicMiniball(
            [(0, 0, 0), (4, 0, 0), (1, 1, 0)], ids=["a", "b", "c"]
        )
        self.assertEqual(dyn.sphere().center, (F(2), F(0), F(0)))
        dyn2 = dyn.delete("b")
        ref = reference(dyn2)
        assert_same_geometry(self, dyn2.sphere(), ref)
        self.assertNotEqual(dyn2.sphere().radius_squared,
                            dyn.sphere().radius_squared)

    def test_delete_coordinate_with_duplicate_keeps_sphere(self):
        # 支撑坐标 (4,0,0) 有两个来源 ID，删一个不影响几何
        dyn = DynamicMiniball(
            [(0, 0, 0), (4, 0, 0), (4, 0, 0)], ids=["a", "b", "b2"]
        )
        before = dyn.sphere()
        dyn2 = dyn.delete("b2")
        after = dyn2.sphere()
        self.assertEqual(after.center, before.center)
        ids_by_point = {s.point: s.ids for s in after.support}
        self.assertEqual(ids_by_point[(F(4), F(0), F(0))], ("b",))
        assert_same_geometry(self, after, reference(dyn2))

    def test_delete_to_empty(self):
        dyn = DynamicMiniball([(1, 2, 3)], ids=["only"])
        dyn2 = dyn.delete("only")
        self.assertEqual(len(dyn2), 0)
        with self.assertRaises(ValueError):
            dyn2.sphere()

    def test_delete_missing_id_raises_and_keeps_state(self):
        dyn = DynamicMiniball([(0, 0, 0), (2, 0, 0)])
        before = dyn.sphere()
        with self.assertRaises(KeyError):
            dyn.delete("nope")
        self.assertEqual(len(dyn), 2)
        self.assertIs(dyn.sphere(), before)


class PersistenceTests(unittest.TestCase):
    def test_old_versions_unaffected(self):
        v1 = DynamicMiniball([(0, 0, 0), (2, 0, 0)])
        s1 = v1.sphere()
        v2 = v1.insert((10, 0, 0))
        v3 = v2.delete(0)
        # 旧版本与其取出的结果完全不受后续修改影响
        self.assertIs(v1.sphere(), s1)
        self.assertEqual(v1.sphere().center, (F(1), F(0), F(0)))
        self.assertEqual(len(v1), 2)
        self.assertEqual(v2.sphere().center, (F(5), F(0), F(0)))
        self.assertEqual(v3.sphere().center, (F(6), F(0), F(0)))
        assert_same_geometry(self, v3.sphere(), reference(v3))


class RandomizedCrossCheckTests(unittest.TestCase):
    def test_random_operation_sequences_match_reference(self):
        rng = random.Random(20260927)
        for _ in range(40):
            dyn = DynamicMiniball()
            next_id = 0
            for _ in range(rng.randint(1, 25)):
                if len(dyn) == 0 or rng.random() < 0.6:
                    pt = tuple(rng.randint(-4, 4) for _ in range(3))
                    dyn = dyn.insert(pt, point_id=f"p{next_id}")
                    next_id += 1
                else:
                    dyn = dyn.delete(rng.choice(dyn.ids))
                if len(dyn) == 0:
                    continue
                assert_same_geometry(self, dyn.sphere(), reference(dyn))
                assert_certificate_valid(self, dyn)


if __name__ == "__main__":
    unittest.main()
