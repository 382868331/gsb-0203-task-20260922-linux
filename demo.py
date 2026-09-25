"""miniball3d 固定输入演示：真实计算最小包围球并逐项打印可验证证书。

运行：python demo.py   （Python 3.14.7，标准库，离线，约 1 秒内）

第一部分：对仓库内确定性代码生成的 8 个有理三维点求最小包围球，
打印精确中心、r^2（不开根）、支撑点及其原始 ID 来源与非负重心系数，
并现场核验：包含性、支撑等距、权重和为 1、平衡残差为 0、对偶恒等式。
第二部分：一个真实触发的接口拒绝边界（NaN 坐标），展示可定位错误。
"""

from fractions import Fraction

from miniball3d import minimum_enclosing_sphere, trimmed_ball


def build_deterministic_points():
    """确定性构造 8 个点：单位立方体 8 个顶点（整数坐标，无随机抖动）。"""
    return [(x, y, z) for x in (0, 4) for y in (0, 4) for z in (0, 4)]


def show(title, value):
    print(f"{title:<28}{value}")


def main():
    print("=== 正常计算：边长 4 的立方体 8 个顶点的最小包围球 ===")
    points = build_deterministic_points()
    ids = [f"v{i}" for i in range(len(points))]
    result = minimum_enclosing_sphere(points, ids=ids)

    show("输入点数 / 去重后", f"{result.num_input_points} / {result.num_unique_points}")
    show("精确中心 c", str(result.center))
    show("精确半径平方 r^2", f"{result.radius_squared}  (= {float(result.radius_squared)})")
    show("半径(仅展示) r", f"{result.radius_float:.12f}")
    print("支撑点与重心证书：")
    for s in result.support:
        print(
            f"  p={s.point}  来源ID={s.ids}  lambda={s.weight}"
            f"  (= {float(s.weight):.6f})"
        )

    # ---- 现场核验（全部用精确 Fraction，不用容差）----
    checks = []

    weight_sum = sum(result.weights, Fraction(0))
    checks.append(("重心系数非负", all(w >= 0 for w in result.weights)))
    checks.append(("重心系数之和 = 1", weight_sum == 1))

    residual = result.certificate_residual()
    checks.append(("平衡残差 sum λ(c-p) = 0", residual == (0, 0, 0)))

    on_sphere = all(
        sum((s.point[d] - result.center[d]) ** 2 for d in range(3))
        == result.radius_squared
        for s in result.support
    )
    checks.append(("每个支撑点恰在球面上", on_sphere))

    contained = all(result.contains(p) for p in points)
    checks.append(("全部 8 个点被球包含", contained))

    checks.append(
        (
            "对偶恒等式 r² = Σλ|p|² - |Σλp|²",
            result.dual_identity_radius_squared() == result.radius_squared,
        )
    )

    print("证书核验：")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert all(ok for _, ok in checks)

    print()
    print("=== 边界：NaN 坐标必须被拒绝且可定位 ===")
    bad = [(0, 0, 0), (1, float("nan"), 0), (2, 0, 0)]
    try:
        minimum_enclosing_sphere(bad)
    except ValueError as exc:
        print(f"  实际捕获 ValueError: {exc}")
        print("  -> 没有静默纠正；错误定位到具体参数路径 points[1][1]")
    else:
        raise AssertionError("NaN 输入应当被拒绝")

    print()
    print("=== trimmed_ball(k=1)：一个孤立观测被排除，而不是被统计判定为异常 ===")
    trimmed_points = [
        (-1, 0, 0),
        (0, 0, 0),
        (2, 0, 0),
        (0, 2, 0),
        (0, -2, 0),
        (10, 0, 0),
    ]
    trimmed_ids = ["left", "origin", "right", "top", "bottom", "isolated"]
    trimmed = trimmed_ball(trimmed_points, 1, ids=trimmed_ids)
    if not trimmed.complete or trimmed.ball is None:
        raise AssertionError("未加 budget 的穷举搜索应当完成并找到可行球")

    show("搜索状态", "complete=True（全部候选已枚举）")
    show("被排除原 ID", str(trimmed.excluded_ids))
    show("被排除输入位置", str(trimmed.excluded_indices))
    show("精确中心 c", str(trimmed.center))
    show("精确半径平方 r^2", str(trimmed.radius_squared))
    print("剩余点的支撑点与重心证书：")
    for support in trimmed.support:
        print(f"  p={support.point}  剩余来源ID={support.ids}  lambda={support.weight}")

    excluded_distance = sum(
        (
            trimmed_points[trimmed.excluded_indices[0]][d] - trimmed.center[d]
        ) ** 2
        for d in range(3)
    )
    retained_ok = all(
        sum((point[d] - trimmed.center[d]) ** 2 for d in range(3))
        <= trimmed.radius_squared
        for i, point in enumerate(trimmed_points)
        if i not in trimmed.excluded_indices
    )
    trimmed_checks = [
        ("恰好排除球外观测", excluded_distance > trimmed.radius_squared),
        ("其余观测全部在球内", retained_ok),
        ("证书平衡残差为 0", trimmed.ball.certificate_residual() == (0, 0, 0)),
        ("对偶恒等式成立", trimmed.ball.dual_identity_radius_squared() == trimmed.radius_squared),
        ("剩余集合非空", trimmed.ball.num_input_points >= 1),
    ]
    for name, ok in trimmed_checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert all(ok for _, ok in trimmed_checks)
    print("  说明：这是给定 k 后的精确几何最优，不提供概率或统计置信解释。")

    print()
    print("=== 预算失败示例：只允许检查 1 个候选时状态必须未完成 ===")
    incomplete = trimmed_ball(trimmed_points, 1, ids=trimmed_ids, budget=1)
    show(
        "搜索状态",
        f"complete={incomplete.complete}, "
        f"examined={incomplete.examined_candidates}/{incomplete.total_candidates}",
    )
    if incomplete.ball is not None:
        show("当前可行球 r^2", str(incomplete.ball.radius_squared))
        show("当前排除 ID", str(incomplete.excluded_ids))
    print("  -> 该球可核验且可行，但搜索未完成，不能声称是全局最优。")
    assert not incomplete.complete
    assert incomplete.examined_candidates == 1


if __name__ == "__main__":
    main()
