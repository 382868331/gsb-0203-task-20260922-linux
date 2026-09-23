"""demo：最小包围球证书的真实计算与一次实际触发的输入拒绝。

运行：python demo.py（离线，标准库，数秒内完成）。
"""

from fractions import Fraction

from meb3d import InvalidInputError, min_enclosing_ball, verify_certificate


def main() -> None:
    # 确定性示例数据：固定种子生成，不依赖任何外部文件或网络。
    import random

    rng = random.Random(20260923)
    points = [
        (rng.randint(-5, 5), rng.randint(-5, 5), rng.randint(-5, 5))
        for _ in range(12)
    ]
    ids = [f"P{i}" for i in range(len(points))]

    print("输入点（12 个，确定性生成）：")
    for ident, p in zip(ids, points):
        print(f"  {ident}: {p}")

    cert = min_enclosing_ball(points, ids=ids)

    print("\n=== 最小包围球证书 ===")
    print(f"球心 center = {tuple(str(c) for c in cert.center)}")
    print(f"半径平方 r^2 = {cert.radius_squared} "
          f"(≈ {float(cert.radius_squared):.6f})")
    print(f"支撑点数 = {len(cert.support)}")
    for sp in cert.support:
        print(f"  支撑 {sp.coordinates} 来源ID={sp.ids} 权重 λ={sp.weight}")

    ok = verify_certificate(points, cert, ids=ids)
    print(f"\n证书核验（包含性 + 凸包 + 对偶恒等式）: {'通过' if ok else '失败'}")

    # 对偶恒等式独立展示：g(λ) = Σλᵢ|sᵢ|² − |Σλᵢsᵢ|² == r²
    ws = [sp.weight for sp in cert.support]
    ss = [sp.coordinates for sp in cert.support]
    combo = [sum(w * s[a] for s, w in zip(ss, ws)) for a in range(3)]
    dual = sum(w * sum(c * c for c in s) for s, w in zip(ss, ws)) - sum(
        c * c for c in combo
    )
    print(f"对偶目标 g(λ) = {dual} == r^2: {dual == cert.radius_squared}")

    # 实际触发一次拒绝：NaN 坐标必须被可定位地拒绝。
    print("\n=== 边界演示：非法输入拒绝 ===")
    bad = [(0, 0, 0), (float("nan"), 1, 2)]
    try:
        min_enclosing_ball(bad)
    except InvalidInputError as exc:
        print(f"按预期拒绝 NaN 输入：{type(exc).__name__}: {exc}")
    else:
        raise SystemExit("错误：NaN 输入未被拒绝")


if __name__ == "__main__":
    main()
