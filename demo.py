"""miniball3d 固定输入演示：真实计算并逐项打印可验证证书。

运行：python demo.py   （Python 3.14.7，标准库，离线，1 秒内）

第一部分：对仓库内确定性代码生成的 8 个有理三维*点*求最小包围球，
打印精确中心、r^2（不开根）、支撑点及其原始 ID 来源与非负重心系数，
并现场核验：包含性、支撑等距、权重和为 1、平衡残差为 0、对偶恒等式。
第二部分：1D/2D *测量球* 的最小覆盖球——正常三球相切结果（逐约束
独立核验距离+半径）、零半径退化为点集的边界、1D 精确区间结果，
以及一个真实触发的接口拒绝（负半径）。
第三部分（旧）：NaN 坐标的可定位拒绝。
"""

from fractions import Fraction

from miniball3d import minimum_enclosing_ball_of_balls, minimum_enclosing_sphere


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
    print("=== 测量球覆盖（2D）：3 个带半径的输入球，逐约束独立核验 ===")
    # r=1 球在 (0,0)、(6,0)；r=2 球在 (3,5)。
    # 解析解：对称性 c=(3, 9/4)，R=19/4=4.75（三球都与结果球相切）。
    balls = [((0, 0), 1), ((6, 0), 1), ((3, 5), 2)]
    res = minimum_enclosing_ball_of_balls(balls, ids=("A", "B", "C"), tol=1e-12)
    print(f"  输入球: {balls}")
    show("求解方法 / 容差", f"{res.method} / tol={res.tol:g}")
    show("结果球心 c", f"({res.center[0]:.12f}, {res.center[1]:.12f})")
    show("结果半径 R", f"{res.radius:.12f}  (手算精确值 19/4 = {Fraction(19,4)})")
    show("球心手算值", "(3.0, 2.25)")
    print("  逐球独立核验 |c-p_i| + r_i <= R（与求解完全独立的重算）：")
    for i, ((p, r), s) in enumerate(zip(balls, res.support)):
        d = ((res.center[0] - p[0]) ** 2 + (res.center[1] - p[1]) ** 2) ** 0.5
        lhs = d + r
        print(
            f"    球{chr(65+i)} p={p} r={r}: |c-p|={d:.12f}  "
            f"|c-p|+r={lhs:.12f}  {'==' if abs(lhs-res.radius)<1e-9 else '<='} R"
        )
    show("最坏约束违反 max(|c-p|+r-R)", f"{res.max_violation():.3e} (<= tol)")
    label_of = {(Fraction(c[0]), Fraction(c[1])): chr(65 + i)
                for i, (c, _r) in enumerate(balls)}
    print("  支撑（相切）球：",
          [label_of[(s.center[0], s.center[1])] for s in res.support])

    print()
    print("=== 边界：零半径退化回点集 + 1D 精确结果 ===")
    # 2D：全部 r=0 —— 与点集最小包围球一致（直角三角形 -> 斜边直径）
    pts_balls = [((0, 0), 0), ((4, 0), 0), ((0, 3), 0)]
    rb = minimum_enclosing_ball_of_balls(pts_balls)
    print(f"  2D 零半径输入 {pts_balls}")
    show("  c / R（数值）", f"({rb.center[0]:g}, {rb.center[1]:g}) / {rb.radius:g}")
    show("  点集接口对照 c / sqrt(r²)", "(2, 1.5) / 2.5")
    # 1D：区间 [-1,1] 与 [3,7] —— 精确 Fraction
    r1 = minimum_enclosing_ball_of_balls([((0,), 1), ((5,), 2)])
    show("  1D 两区间 c / R（精确）", f"{r1.center} / {r1.radius}")
    show("  1D 支撑系数", tuple(str(s.weight) for s in r1.support))

    print()
    print("=== 错误边界：负半径必须被拒绝且可定位 ===")
    try:
        minimum_enclosing_ball_of_balls([((0, 0), 1), ((2, -1), -3)])
    except ValueError as exc:
        print(f"  实际捕获 ValueError: {exc}")
        print("  -> 定位到 balls[1][1]；校验在求解前完成，无部分结果")
    else:
        raise AssertionError("负半径输入应当被拒绝")

    print()
    print("=== 边界（旧）：NaN 坐标必须被拒绝且可定位 ===")
    bad = [(0, 0, 0), (1, float("nan"), 0), (2, 0, 0)]
    try:
        minimum_enclosing_sphere(bad)
    except ValueError as exc:
        print(f"  实际捕获 ValueError: {exc}")
        print("  -> 没有静默纠正；错误定位到具体参数路径 points[1][1]")
    else:
        raise AssertionError("NaN 输入应当被拒绝")


if __name__ == "__main__":
    main()
