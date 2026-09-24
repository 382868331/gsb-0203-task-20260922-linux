"""miniball3d 固定输入演示：两部分，全部真实计算、无 sleep、无预录答案。

运行：python demo.py   （Python 3.14.7，标准库，离线，1 秒内）

第一部分（旧能力，保持不变）：8 个有理三维点的精确最小包围球与证书。
第二部分（本次迭代）：一维/二维 *输入球* 的数值最小包围球：
  2a. 正常结果：三个半径不等的二维输入球（解析构造，期望 c=(0,0), R=10），
      逐球独立核验 dist+半径 ≤ R、支撑紧贴、KKT 权重与法向平衡；
  2b. 边界：大球完整包含小球（单支撑）与零半径点输入退化；
  2c. 实际错误：超过 6 个球 / 负半径必须被拒绝且错误可定位。
"""

import math
from fractions import Fraction

from miniball3d import minimum_enclosing_ball, minimum_enclosing_sphere


def show(title, value):
    print(f"{title:<32}{value}")


def demo_points():
    print("=== [1] 三维点集（精确 Fraction）：边长 4 的立方体 8 顶点 ===")
    points = [(x, y, z) for x in (0, 4) for y in (0, 4) for z in (0, 4)]
    ids = [f"v{i}" for i in range(len(points))]
    result = minimum_enclosing_sphere(points, ids=ids)

    show("输入点数 / 去重后", f"{result.num_input_points} / {result.num_unique_points}")
    show("精确中心 c", str(result.center))
    show("精确半径平方 r^2", f"{result.radius_squared}  (= {float(result.radius_squared)})")
    show("半径(仅展示) r", f"{result.radius_float:.12f}")
    print("支撑点与重心证书：")
    for s in result.support:
        print(f"  p={s.point}  来源ID={s.ids}  lambda={s.weight}")

    checks = [
        ("重心系数非负且和为 1",
         all(w >= 0 for w in result.weights) and sum(result.weights, Fraction(0)) == 1),
        ("平衡残差 sum λ(c-p) = 0", result.certificate_residual() == (0, 0, 0)),
        ("每个支撑点恰在球面上", all(
            sum((s.point[d] - result.center[d]) ** 2 for d in range(3))
            == result.radius_squared for s in result.support)),
        ("全部 8 个点被球包含", all(result.contains(p) for p in points)),
        ("对偶恒等式", result.dual_identity_radius_squared() == result.radius_squared),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert all(ok for _, ok in checks)


def demo_balls_normal():
    print()
    print("=== [2a] 二维输入球（数值解）：三个不等半径球，解析构造期望 R=10 ===")
    # 构造：c*=0, R*=10；三条法向相隔 120°（Σn=0），距离分别为 8/5/6，
    # 故输入半径为 10-d：2/5/4。
    balls = [
        ((-8.0, 0.0), 2.0),
        ((2.5, -5.0 * math.sqrt(3.0) / 2.0), 5.0),
        ((3.0, 3.0 * math.sqrt(3.0)), 4.0),
    ]
    ids = ["B0", "B1", "B2"]
    r = minimum_enclosing_ball(balls, ids=ids, tol=1e-12)

    show("中心 c", f"({r.center[0]:.12g}, {r.center[1]:.12g})")
    show("半径 R", f"{r.radius:.12f}   (手算期望 10)")
    show("支撑球数", len(r.support))
    for s in r.support:
        print(
            f"  支撑 {s.ids}: 球心={s.center} 半径={s.radius}"
            f"  KKT权重 λ={s.weight:.12f}"
        )

    # 逐球独立核验：每个输入球单独一个谓词，不汇总
    print("逐球约束独立核验 dist(c, p_i) + a_i ≤ R：")
    all_ok = True
    for i, ((p, a), slack, value) in enumerate(
        zip(balls, r.slacks, r.constraint_values)
    ):
        dist = math.hypot(r.center[0] - p[0], r.center[1] - p[1])
        ok = dist + a <= r.radius + 1e-9 * max(1.0, r.radius)
        active = abs(slack) < 1e-8
        all_ok &= ok
        print(
            f"  [{'PASS' if ok else 'FAIL'}] 球{i}: dist={dist:.12f}"
            f" + a={a} = {value:.12f} ≤ R={r.radius:.12f}"
            f"  slack={slack:+.2e}  {'<紧贴>' if active else ''}"
        )

    bal = r.balance_residual()
    weight_ok = all(w >= -1e-9 for w in r.weights) and abs(sum(r.weights) - 1.0) < 1e-9
    checks = [
        ("三个输入球全部被包含（逐球独立检查）", all_ok),
        ("KKT 权重非负且和为 1", weight_ok),
        (f"法向平衡残差 Σλn ≈ 0（实测 {bal[0]:+.2e},{bal[1]:+.2e}）",
         abs(bal[0]) < 1e-8 and abs(bal[1]) < 1e-8),
        ("最大违反量 max(dist+a-R) ≤ 容差口径", r.max_violation() <= 1e-8),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    assert all(ok for _, ok in checks)


def demo_balls_boundary():
    print()
    print("=== [2b] 边界：大球完整包含小球（单支撑）；零半径退化为点 ===")
    # 一维：心 0 半径 5 的球完整包含心 3 半径 1 的球（3+1 ≤ 5）
    r = minimum_enclosing_ball([((0.0,), 5.0), ((3.0,), 1.0)], dim=1)
    show("一维两球中心", str(r.center))
    show("一维两球半径 R", r.radius)
    show("支撑数（期望 1）", len(r.support))
    show("两球 slack", str(tuple(f"{s:+.3f}" for s in r.slacks)))
    assert len(r.support) == 1
    assert r.slacks[1] > 0.9

    # 零半径 = 点输入：两点 (0,0) 与 (4,0)
    r2 = minimum_enclosing_ball([((0.0, 0.0), 0.0), ((4.0, 0.0), 0.0)])
    show("零半径两点中心", str(r2.center))
    show("零半径两点 R（期望 2）", r2.radius)
    assert abs(r2.radius - 2.0) < 1e-12

    # 容差生效：contains_ball 球面边界判定
    assert r2.contains_ball((2.0, 0.0), 2.0)      # 内切，恰好包含
    assert not r2.contains_ball((2.0, 0.0), 2.5)  # 越界
    print("  [PASS] 内切球被包含、更大的球被拒绝")


def demo_balls_errors():
    print()
    print("=== [2c] 实际错误：超上限与负半径必须被明确拒绝 ===")
    try:
        minimum_enclosing_ball([((float(i), 0.0), 0.0) for i in range(7)])
    except ValueError as exc:
        print(f"  [PASS] 7 个球 -> ValueError: {exc}")
    else:
        raise AssertionError("7 个球应当被拒绝")

    try:
        minimum_enclosing_ball([((0.0, 0.0), -0.25)])
    except ValueError as exc:
        print(f"  [PASS] 负半径 -> ValueError: {exc}")
    else:
        raise AssertionError("负半径应当被拒绝")

    try:
        minimum_enclosing_ball([((0.0, float("nan")), 1.0)])
    except ValueError as exc:
        print(f"  [PASS] NaN 球心 -> ValueError: {exc}")
    else:
        raise AssertionError("NaN 球心应当被拒绝")


def main():
    demo_points()
    demo_balls_normal()
    demo_balls_boundary()
    demo_balls_errors()
    print()
    print("全部演示核验通过。")


if __name__ == "__main__":
    main()
