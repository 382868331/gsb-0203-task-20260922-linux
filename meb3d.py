"""三维有理点集的最小包围球（minimum enclosing ball）与最优性证书。

精确算法：最小包围球的边界上必存在 1..4 个仿射独立的支撑点，
且球心位于这些支撑点的凸包内（3 维 Carathéodory 界）。本模块枚举
所有 1..4 个（去重后）点的子集，对每个仿射独立子集用等距方程解出
其仿射包上唯一与所有支撑点等距的候选球心，只保留满足以下两条的候选：

1. 球心位于支撑点凸包内（全部重心系数 >= 0）；
2. 球包含全部输入点（距离平方 <= 半径平方，精确比较）。

在所有可行候选中取半径平方最小者。全部计算使用 fractions.Fraction
精确有理数运算，无浮点舍入、无随机抖动，结果对输入顺序确定。

对偶恒等式（用于证书核验）：对任意非负、和为 1 的权重 λ，都有

    g(λ) = Σ λᵢ |pᵢ|² − |Σ λᵢ pᵢ|²  <=  最优半径平方。

证书给出的支撑权重使等号成立，结合“球包含所有点”即构成最优性的
完整初等证明（原始可行 + 对偶可行 + 目标值相等）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import comb
from typing import Hashable, Optional, Sequence, Tuple

__all__ = [
    "DIMENSION",
    "MAX_POINTS",
    "SupportPoint",
    "BallCertificate",
    "MebError",
    "InvalidInputError",
    "SearchBudgetExhaustedError",
    "NoFeasibleCandidateError",
    "CertificateVerificationError",
    "min_enclosing_ball",
    "verify_certificate",
]

#: 点的空间维数（本模块只处理三维）。
DIMENSION = 3

#: 输入点数上限（接口边界，按原始输入条目计，去重前）。
MAX_POINTS = 30

#: 候选子集枚举的防御性搜索预算：n <= MAX_POINTS 时
#: C(n,1)+C(n,2)+C(n,3)+C(n,4) 的最大值。枚举本身受组合数上界约束，
#: 该预算只是防止实现被改动后失控的保险丝，正常路径不可达。
_SEARCH_BUDGET = sum(comb(MAX_POINTS, k) for k in range(1, DIMENSION + 2))


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupportPoint:
    """一个支撑点：坐标、来源 ID、最优证书中的重心系数。"""

    #: 该坐标对应的全部原始输入 ID（去重后保留来源，按输入顺序）。
    ids: Tuple[Hashable, ...]
    #: 精确坐标，3 个 Fraction。
    coordinates: Tuple[Fraction, Fraction, Fraction]
    #: 重心系数 λ，满足 λ >= 0 且所有支撑点的 λ 之和为 1。
    weight: Fraction


@dataclass(frozen=True)
class BallCertificate:
    """最小包围球证书：球心、半径平方、支撑点及重心系数。"""

    #: 球心，3 个 Fraction。
    center: Tuple[Fraction, Fraction, Fraction]
    #: 半径平方（精确 Fraction；不开根，避免无理数）。
    radius_squared: Fraction
    #: 支撑点元组，1..4 个，仿射独立。
    support: Tuple[SupportPoint, ...]


# ---------------------------------------------------------------------------
# 错误语义
# ---------------------------------------------------------------------------


class MebError(Exception):
    """本模块所有异常的基类。"""


class InvalidInputError(MebError, ValueError):
    """输入非法（类型、维数、NaN/Infinity、数量超限、ID 冲突等）。

    异常消息中携带可定位的位置信息（如 ``points[3][1]``）。
    """


class SearchBudgetExhaustedError(MebError, RuntimeError):
    """候选枚举超出搜索预算（防御性保险丝，正常输入不可达）。

    与“证明无解”区分：预算耗尽表示搜索未完成，不表示结论。
    """


class NoFeasibleCandidateError(MebError, RuntimeError):
    """枚举完整结束后仍无任何可行候选。

    数学上非空有限点集必存在最小包围球，出现此异常说明实现缺陷，
    与搜索预算耗尽是两种不同的失败。
    """


class CertificateVerificationError(MebError, ValueError):
    """证书核验失败，消息说明具体哪一条不变式被破坏。"""


# ---------------------------------------------------------------------------
# 输入校验与去重
# ---------------------------------------------------------------------------


def _to_fraction(value, where: str) -> Fraction:
    """把单个坐标值转换为 Fraction，非法值抛出可定位错误。"""
    if isinstance(value, bool):
        raise InvalidInputError(f"{where}: bool 不是合法坐标值")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise InvalidInputError(f"{where}: 拒绝 NaN/Infinity 坐标")
        # 二进制浮点值被精确转换为等值有理数（不四舍五入）。
        return Fraction(value)
    raise InvalidInputError(
        f"{where}: 不支持的坐标类型 {type(value).__name__}，"
        "只接受 int / Fraction / 有限 float"
    )


def _validate_and_dedup(points, ids):
    """校验输入并按坐标去重。

    返回 (unique_points, id_groups)：unique_points 为 Fraction 坐标元组
    的列表（按首次出现顺序），id_groups[i] 为对应的所有原始 ID 元组。
    """
    if isinstance(points, (str, bytes)):
        raise InvalidInputError("points: 必须是点的序列，不是字符串")
    try:
        raw = list(points)
    except TypeError:
        raise InvalidInputError("points: 必须是可迭代的点序列") from None
    if len(raw) == 0:
        raise InvalidInputError("points: 至少需要一个点")
    if len(raw) > MAX_POINTS:
        raise InvalidInputError(
            f"points: 共 {len(raw)} 个点，超过上限 MAX_POINTS={MAX_POINTS}"
        )

    if ids is None:
        id_list = list(range(len(raw)))
    else:
        if isinstance(ids, (str, bytes)):
            raise InvalidInputError("ids: 必须是 ID 序列，不是字符串")
        try:
            id_list = list(ids)
        except TypeError:
            raise InvalidInputError("ids: 必须是可迭代的 ID 序列") from None
        if len(id_list) != len(raw):
            raise InvalidInputError(
                f"ids: 长度 {len(id_list)} 与点数 {len(raw)} 不一致"
            )
        seen = set()
        for i, ident in enumerate(id_list):
            try:
                hash(ident)
            except TypeError:
                raise InvalidInputError(
                    f"ids[{i}]: ID 必须可哈希，得到 {type(ident).__name__}"
                ) from None
            if ident in seen:
                raise InvalidInputError(f"ids[{i}]: ID {ident!r} 重复")
            seen.add(ident)

    unique = []
    id_groups = []
    index_of = {}
    for i, p in enumerate(raw):
        if isinstance(p, (str, bytes)):
            raise InvalidInputError(f"points[{i}]: 点必须是 3 个数坐标，不是字符串")
        try:
            coords = list(p)
        except TypeError:
            raise InvalidInputError(
                f"points[{i}]: 点必须是含 3 个坐标的序列"
            ) from None
        if len(coords) != DIMENSION:
            raise InvalidInputError(
                f"points[{i}]: 维数为 {len(coords)}，必须为 {DIMENSION}"
            )
        key = tuple(
            _to_fraction(v, f"points[{i}][{a}]") for a, v in enumerate(coords)
        )
        j = index_of.get(key)
        if j is None:
            index_of[key] = len(unique)
            unique.append(key)
            id_groups.append((id_list[i],))
        else:
            id_groups[j] = id_groups[j] + (id_list[i],)
    return unique, id_groups


# ---------------------------------------------------------------------------
# 精确线性代数（仅用于 <= 4 阶小系统）
# ---------------------------------------------------------------------------


def _solve_linear(matrix, rhs) -> Optional[list]:
    """高斯-若尔当精确消元解 n 阶线性方程组；奇异时返回 None。"""
    n = len(matrix)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if aug[r][col] != 0:
                piv = r
                break
        if piv is None:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        inv = 1 / aug[col][col]
        aug[col] = [x * inv for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                f = aug[r][col]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[col])]
    return [aug[i][n] for i in range(n)]


def _candidate_from_support(dots, norms, pts, subset):
    """计算子集对应的候选 (center, r2, weights)；仿射相关返回 None。

    候选球心是子集仿射包上与所有子集点等距的唯一一点，用重心坐标
    λ 表示：Σλⱼ = 1 且 2(pᵢ−p₀)·(Σλⱼpⱼ) = |pᵢ|²−|p₀|²。
    """
    k = len(subset)
    if k == 1:
        return pts[subset[0]], Fraction(0), (Fraction(1),)
    i0 = subset[0]
    matrix = [[Fraction(1)] * k]
    rhs = [Fraction(1)]
    for i in subset[1:]:
        matrix.append(
            [Fraction(2) * (dots[i][j] - dots[i0][j]) for j in subset]
        )
        rhs.append(norms[i] - norms[i0])
    lam = _solve_linear(matrix, rhs)
    if lam is None:
        return None
    center = tuple(
        sum(lam[j] * pts[subset[j]][a] for j in range(k)) for a in range(DIMENSION)
    )
    r2 = sum(
        (center[a] - pts[i0][a]) ** 2 for a in range(DIMENSION)
    )
    return center, r2, tuple(lam)


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------


def min_enclosing_ball(points, ids=None) -> BallCertificate:
    """计算三维有理点集的最小包围球并返回最优性证书。

    参数：
        points: 1..MAX_POINTS 个点的可迭代序列，每个点是 3 个
            int / Fraction / 有限 float 坐标。拒绝 NaN、Infinity、
            bool、字符串及维数不为 3 的点。
        ids: 可选，与 points 等长的唯一可哈希 ID 序列；缺省为
            0..n-1。坐标重复的点被去重，但其全部 ID 保留在证书
            支撑点的 ``ids`` 字段中。

    返回：BallCertificate（精确 Fraction 结果）。

    异常：
        InvalidInputError: 输入非法，消息含位置信息。
        SearchBudgetExhaustedError: 搜索预算耗尽（防御性，正常不可达）。
        NoFeasibleCandidateError: 完整枚举后无可行候选（实现缺陷信号）。
    """
    pts, id_groups = _validate_and_dedup(points, ids)
    n = len(pts)

    norms = [sum(c * c for c in p) for p in pts]
    dots = [
        [sum(pts[i][a] * pts[j][a] for a in range(DIMENSION)) for j in range(n)]
        for i in range(n)
    ]

    best = None  # (r2, subset, center, weights)
    examined = 0
    for k in range(1, min(DIMENSION + 1, n) + 1):
        for subset in combinations(range(n), k):
            examined += 1
            if examined > _SEARCH_BUDGET:
                raise SearchBudgetExhaustedError(
                    f"候选枚举超过搜索预算 {_SEARCH_BUDGET}，搜索未完成"
                )
            cand = _candidate_from_support(dots, norms, pts, subset)
            if cand is None:
                continue
            center, r2, weights = cand
            if any(w < 0 for w in weights):
                continue  # 球心不在支撑凸包内
            if best is not None and r2 >= best[0]:
                continue  # 不可能更优（并列时保留先找到的更小支撑）
            center_norm = sum(c * c for c in center)
            feasible = True
            for j in range(n):
                d2 = (
                    center_norm
                    - 2 * sum(center[a] * pts[j][a] for a in range(DIMENSION))
                    + norms[j]
                )
                if d2 > r2:
                    feasible = False
                    break
            if feasible:
                best = (r2, subset, center, weights)
        if best is not None and best[0] == 0:
            break  # 半径为 0 已是最优下界

    if best is None:
        raise NoFeasibleCandidateError(
            "完整枚举后没有任何可行候选；非空有限点集理论上必有最小包围球，"
            "请报告此实现缺陷"
        )

    r2, subset, center, weights = best
    support = tuple(
        SupportPoint(ids=id_groups[i], coordinates=pts[i], weight=w)
        for i, w in zip(subset, weights)
    )
    return BallCertificate(center=center, radius_squared=r2, support=support)


# ---------------------------------------------------------------------------
# 证书核验
# ---------------------------------------------------------------------------


def verify_certificate(points, certificate: BallCertificate, ids=None) -> bool:
    """独立核验最小包围球证书；全部不变式满足时返回 True。

    核验内容（任一不满足即抛 CertificateVerificationError）：
      1. 证书结构合法（球心 3 维、半径平方为非负 Fraction、1..4 个支撑点）；
      2. 每个支撑点坐标确实出现在输入点集中；
      3. 重心系数均 >= 0 且和为 1；
      4. 球心等于支撑点的加权组合（球心在支撑凸包内）；
      5. 每个支撑点到球心距离平方等于半径平方（支撑点在边界上）；
      6. 所有输入点到球心距离平方 <= 半径平方（球包含所有点）；
      7. 对偶恒等式 r² = Σλᵢ|sᵢ|² − |c|² 成立。

    其中 3、6、7 合起来构成最优性的完整初等证明（对偶可行解的目标值
    等于原始可行解的目标值），不依赖本模块的搜索过程。
    """
    pts, _ = _validate_and_dedup(points, ids)
    pt_set = set(pts)

    if not isinstance(certificate, BallCertificate):
        raise CertificateVerificationError(
            f"certificate: 类型必须为 BallCertificate，得到 "
            f"{type(certificate).__name__}"
        )
    center = certificate.center
    r2 = certificate.radius_squared
    support = certificate.support

    if not (isinstance(center, tuple) and len(center) == DIMENSION):
        raise CertificateVerificationError("certificate.center: 必须是 3 元组")
    if any(not isinstance(c, Fraction) for c in center):
        raise CertificateVerificationError(
            "certificate.center: 坐标必须全部是 Fraction"
        )
    if not isinstance(r2, Fraction):
        raise CertificateVerificationError(
            "certificate.radius_squared: 必须是 Fraction"
        )
    if r2 < 0:
        raise CertificateVerificationError(
            f"certificate.radius_squared: 不能为负，得到 {r2}"
        )
    if not (1 <= len(support) <= DIMENSION + 1):
        raise CertificateVerificationError(
            f"certificate.support: 支撑点数为 {len(support)}，必须在 1..4"
        )

    for t, sp in enumerate(support):
        if not isinstance(sp, SupportPoint):
            raise CertificateVerificationError(
                f"certificate.support[{t}]: 类型必须为 SupportPoint"
            )
        if not isinstance(sp.weight, Fraction):
            raise CertificateVerificationError(
                f"certificate.support[{t}].weight: 必须是 Fraction"
            )
        if sp.weight < 0:
            raise CertificateVerificationError(
                f"certificate.support[{t}].weight: 重心系数为负 ({sp.weight})"
            )
        if tuple(sp.coordinates) not in pt_set:
            raise CertificateVerificationError(
                f"certificate.support[{t}]: 坐标 {tuple(sp.coordinates)} "
                "不在输入点集中"
            )

    weight_sum = sum((sp.weight for sp in support), Fraction(0))
    if weight_sum != 1:
        raise CertificateVerificationError(
            f"重心系数之和为 {weight_sum}，必须等于 1"
        )

    for a in range(DIMENSION):
        combo = sum(
            (sp.weight * sp.coordinates[a] for sp in support), Fraction(0)
        )
        if combo != center[a]:
            raise CertificateVerificationError(
                f"球心第 {a} 坐标为 {center[a]}，不等于支撑点加权组合 {combo}"
            )

    for t, sp in enumerate(support):
        d2 = sum(
            (sp.coordinates[a] - center[a]) ** 2 for a in range(DIMENSION)
        )
        if d2 != r2:
            raise CertificateVerificationError(
                f"支撑点 {t} 到球心距离平方为 {d2}，不等于半径平方 {r2}"
            )

    for j, q in enumerate(pts):
        d2 = sum((q[a] - center[a]) ** 2 for a in range(DIMENSION))
        if d2 > r2:
            raise CertificateVerificationError(
                f"去重后第 {j} 个点 {q} 到球心距离平方 {d2} 超过半径平方 {r2}"
            )

    dual = sum(
        (sp.weight * sum(c * c for c in sp.coordinates) for sp in support),
        Fraction(0),
    ) - sum(c * c for c in center)
    if dual != r2:
        raise CertificateVerificationError(
            f"对偶恒等式不成立：Σλᵢ|sᵢ|² − |c|² = {dual} ≠ 半径平方 {r2}"
        )

    return True
