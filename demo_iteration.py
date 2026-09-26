"""受约束包围球演示：球心限定在指定平面/直线上的最小包围球。

运行：python demo_iteration.py   （Python 3.14.7，标准库，离线，秒级）

第一部分：平面约束——正常结果，现场核验覆盖、等距、梯度凸组合为零。
第二部分：直线约束——最优中心落在直线上的例子。
第三部分：边界情况——线性相关方向被真实拒绝（秩检查）。
全部现场真实计算，无预录、无 sleep。
"""

from fractions import Fraction

from miniball3d import minimum_enclosing_sphere_constrained


def show(result, points, title):
    print(f"== {title} ==")
    print("约束维数:", result.subspace_dimension,
          "(1=直线, 2=平面)")
    print("精确球心:", tuple(str(c) for c in result.center))
    print("半径平方:", result.radius_squared)
    print("活动点（含原始 ID 来源与证书权重）:")
    for s in result.support:
        print(f"  point={tuple(str(c) for c in s.point)} ids={s.ids} "
              f"weight={s.weight}")
    print("逐点覆盖:", all(result.contains(p) for p in points))
    print("活动点梯度凸组合（低维残差，应为零）:",
          tuple(str(x) for x in result.certificate_residual()))
    print("对偶恒等式核验:",
          result.dual_identity_radius_squared() == result.radius_squared)
    print("投影退化活动集计数:", result.degenerate_combinations)
    print()


def main():
    # 1) 平面约束：z=0 平面，活动点含平面外点（不全部共面于约束平面）
    plane_points = [(0, 0, 1), (2, 0, 0), (0, 2, 0), (1, 1, -1)]
    r1 = minimum_enclosing_sphere_constrained(
        plane_points,
        subspace_origin=(0, 0, 0),
        subspace_directions=[(1, 0, 0), (0, 1, 0)],
        ids=["a", "b", "c", "d"],
    )
    show(r1, plane_points, "平面约束（z=0）：正常结果")

    # 2) 直线约束：x 轴，最优中心在直线上
    line_points = [(0, 1, 0), (2, -1, 0), (1, 0, 2)]
    r2 = minimum_enclosing_sphere_constrained(
        line_points,
        subspace_origin=(0, 0, 0),
        subspace_directions=[(1, 0, 0)],
    )
    show(r2, line_points, "直线约束（x 轴）：最优中心在直线上")

    # 3) 边界情况：两个方向线性相关，秩检查真实拒绝
    print("== 边界：相关方向被拒绝 ==")
    try:
        minimum_enclosing_sphere_constrained(
            [(0, 0, 0), (1, 1, 1)],
            subspace_origin=(0, 0, 0),
            subspace_directions=[(1, 2, 3), (2, 4, 6)],
        )
    except ValueError as exc:
        print("按预期拒绝:", exc)


if __name__ == "__main__":
    main()
