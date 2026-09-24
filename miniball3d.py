"""miniball3d — 最小包围球：三维点集的精确证书，以及 1D/2D 测量球覆盖。

仅使用 Python 标准库（fractions / itertools / dataclasses / math）。

公开接口
========
    minimum_enclosing_sphere(points, *, ids=None, budget=None)
        三维有理*点*集（1..30 个点）的精确 Fraction 最小包围球 + 最优证书。

    minimum_enclosing_ball_of_balls(balls, *, ids=None, tol=1e-10)
        1D/2D *测量球*（带非负半径的输入球，1..6 个）的最小覆盖球。
        半径全为 0 时即退化为点集最小包围球。1D 精确（Fraction）；
        2D 为带显式容差 tol 的 float64 数值解，并对每个输入球独立核验
        “距离 + 半径”约束。

核心思路（题面规定的枚举证书法）
--------------------------------
1. 输入坐标去重（保留全部原始 ID 来源），唯一坐标数 m <= 30。
2. 枚举 1..4 个仿射独立唯一坐标的全部子集 S（3 维空间中确定等距球所需
   且足够的支撑规模）。
3. 等距方程 |c-p_i|^2 = r^2 在线性化后用 Fraction 高斯消元精确求解，
   得到候选中心 c 与 c 关于 S 的重心系数 lambda。
4. 候选成立当且仅当：
     (a) 所有点满足 |q-c|^2 <= r^2（精确包含）；
     (b) 全部 lambda_i >= 0（且天然 sum lambda_i = 1），即 c 位于 S 的凸包内。
   后者正是最小包围球的最优性证书：0 = sum lambda_i (c - p_i)，
   支撑点等距、其余点不越界。
5. 在全部成立候选中取 r^2 最小者；平局先取支撑数更少、再取下标字典序，
   结果完全确定。

本模块无状态、不修改输入、不使用随机数或浮点容差：所有几何判定都是
Fraction 精确比较。浮点仅用于展示（见 README“误差口径”）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from numbers import Rational

__all__ = [
    "MIN_BALL_MAX_POINTS",
    "MIN_BALL_MAX_BALLS",
    "DEFAULT_BALL_TOL",
    "SupportPoint",
    "MinimumEnclosingSphere",
    "BallSupport",
    "MinimumEnclosingBall",
    "MiniballError",
    "NoFeasibleCandidateError",
    "SearchBudgetExhaustedError",
    "minimum_enclosing_sphere",
    "minimum_enclosing_ball_of_balls",
]

#: 接口允许的输入点数量上限（题面规定 30）。
MIN_BALL_MAX_POINTS = 30

#: 球覆盖接口允许的输入球数量上限（本轮题面规定 6）。
MIN_BALL_MAX_BALLS = 6

#: 2D 数值求解的默认容差（绝对量；见 minimum_enclosing_ball_of_balls）。
DEFAULT_BALL_TOL = 1e-10


class MiniballError(Exception):
    """本模块所有自定义异常的基类。"""


class NoFeasibleCandidateError(MiniballError, RuntimeError):
    """穷举全部候选后证明不存在可行包围球。

    对非空有限点集该情况在数学上不可能发生；保留此异常是为了与
    “搜索预算耗尽”这一 *未证明* 情形严格区分。
    """


class SearchBudgetExhaustedError(MiniballError, RuntimeError):
    """给定 budget 在枚举完成前耗尽：结果未知，绝不冒充无解或最优。"""


@dataclass(frozen=True)
class SupportPoint:
    """一个支撑坐标及其最优证书信息。

    Attributes:
        point: 去重后的三维坐标（Fraction 三元组）。
        ids:   映射到该坐标的全部原始输入 ID（按首次出现顺序）。
        weight: 该支撑点的非负重心系数；全部支撑权重之和恰为 1。
    """

    point: tuple[Fraction, Fraction, Fraction]
    ids: tuple
    weight: Fraction


@dataclass(frozen=True)
class MinimumEnclosingSphere:
    """最小包围球结果。

    全部几何字段都是精确 Fraction；float 视图仅通过 *_float 属性按需得到。
    """

    center: tuple[Fraction, Fraction, Fraction]
    radius_squared: Fraction
    support: tuple[SupportPoint, ...]
    num_input_points: int
    num_unique_points: int

    @property
    def weights(self) -> tuple[Fraction, ...]:
        """与 support 同序的重心系数。"""
        return tuple(s.weight for s in self.support)

    @property
    def center_float(self) -> tuple[float, float, float]:
        """中心的 float64 近似（逐坐标一次精确舍入）。"""
        return tuple(float(x) for x in self.center)

    @property
    def radius_float(self) -> float:
        """半径（sqrt(r^2)）的 float64 近似；精确量请用 radius_squared。"""
        return math.sqrt(float(self.radius_squared))

    def contains(self, point) -> bool:
        """精确判定点是否被球包含（边界算包含）。"""
        q = _validate_coordinate_sequence(point, where="contains() argument")
        return _dist2(q, self.center) <= self.radius_squared

    def dual_identity_radius_squared(self) -> Fraction:
        """对偶恒等式右侧：sum_i lambda_i |p_i|^2 - |sum_i lambda_i p_i|^2。

        最优时它与 radius_squared 精确相等（等于
        sum_{i<j} lambda_i lambda_j |p_i-p_j|^2），可用于独立核验证书。
        """
        weighted_norm = sum(
            (s.weight * _norm2(s.point) for s in self.support), Fraction(0)
        )
        wc = tuple(
            sum((s.weight * s.point[d] for s in self.support), Fraction(0))
            for d in range(3)
        )
        return weighted_norm - _norm2(wc)

    def certificate_residual(self) -> tuple[Fraction, Fraction, Fraction]:
        """sum_i lambda_i (c - p_i)，最优证书下恰为 (0,0,0)。"""
        return tuple(
            sum(
                (s.weight * (self.center[d] - s.point[d]) for s in self.support),
                Fraction(0),
            )
            for d in range(3)
        )


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------

def _reject_bool(value, where: str) -> None:
    # bool 是 int 的子类型，按契约必须显式拒绝，不得被当成 0/1。
    if isinstance(value, bool):
        raise TypeError(f"{where}: bool is not accepted (got {value!r})")


def _coerce_scalar(value, where: str) -> Fraction:
    _reject_bool(value, where)
    if isinstance(value, Rational):
        return Fraction(value.numerator, value.denominator)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where}: NaN/Infinity is not allowed (got {value!r})")
        # 有限 float 按其二进制原值精确转成分数，不引入任何抖动或“贴近”。
        return Fraction(value)
    raise TypeError(
        f"{where}: rational number expected (int/Fraction, or finite float), "
        f"got {type(value).__name__}"
    )


def _validate_coordinate_sequence(point, *, where: str) -> tuple[Fraction, Fraction, Fraction]:
    if isinstance(point, (str, bytes)) or not hasattr(point, "__len__"):
        raise TypeError(f"{where}: sequence of 3 coordinates expected")
    if len(point) != 3:
        raise ValueError(f"{where}: expected exactly 3 coordinates, got {len(point)}")
    return tuple(
        _coerce_scalar(v, where=f"{where}[{i}]") for i, v in enumerate(point)
    )


def _normalize_points(points):
    if isinstance(points, (str, bytes)) or not hasattr(points, "__len__"):
        raise TypeError("points: a finite sequence of 3D points is required")
    n = len(points)
    if n == 0:
        raise ValueError("points: at least 1 point is required")
    if n > MIN_BALL_MAX_POINTS:
        raise ValueError(
            f"points: at most {MIN_BALL_MAX_POINTS} points are allowed, got {n}"
        )
    normalized = []
    for i, p in enumerate(points):
        normalized.append(
            _validate_coordinate_sequence(p, where=f"points[{i}]")
        )
    return normalized


def _normalize_ids(ids, n: int, *, what: str = "points"):
    if ids is None:
        return tuple(range(n))
    if isinstance(ids, (str, bytes)) or not hasattr(ids, "__len__"):
        raise TypeError(f"ids: a finite sequence aligned with {what} is required")
    if len(ids) != n:
        raise ValueError(
            f"ids: length {len(ids)} does not match number of {what} {n}"
        )
    out = []
    seen = set()
    for i, x in enumerate(ids):
        try:
            if x in seen:
                raise ValueError(f"ids[{i}]: duplicate id {x!r}")
            seen.add(x)
        except TypeError as exc:
            raise TypeError(f"ids[{i}]: id must be hashable, got {type(x).__name__}") from exc
        out.append(x)
    return tuple(out)


def _normalize_budget(budget) -> int | None:
    if budget is None:
        return None
    _reject_bool(budget, "budget")
    if not isinstance(budget, int):
        raise TypeError("budget: positive int or None expected")
    if budget <= 0:
        raise ValueError(f"budget: must be a positive integer, got {budget}")
    return budget


# ---------------------------------------------------------------------------
# 精确向量运算与线性求解
# ---------------------------------------------------------------------------

def _norm2(p) -> Fraction:
    return p[0] * p[0] + p[1] * p[1] + p[2] * p[2]


def _dist2(a, b) -> Fraction:
    return (
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    )


def _solve_linear(A, b):
    """Fraction 高斯消元（选主元）。奇异返回 None。"""
    n = len(A)
    mat = [
        [Fraction(A[r][c]) for c in range(n)] + [Fraction(b[r])]
        for r in range(n)
    ]
    for col in range(n):
        pivot = next(
            (r for r in range(col, n) if mat[r][col] != 0), None
        )
        if pivot is None:
            return None
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
        pivot_value = mat[col][col]
        for r in range(col + 1, n):
            factor = mat[r][col] / pivot_value
            if factor != 0:
                mat[r] = [
                    mat[r][j] - factor * mat[col][j] for j in range(n + 1)
                ]
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        x[i] = (
            mat[i][n] - sum(mat[i][j] * x[j] for j in range(i + 1, n))
        ) / mat[i][i]
    return x


def _candidate(unique_points, combo):
    """由等距方程求候选中心、r^2、重心系数。

    令 c = p0 + sum_{j>=1} alpha_j (p_j - p_0)，代入
    2 c . (p_i - p_0) = |p_i|^2 - |p_0|^2 得 Gram 方程 G alpha = rhs。
    G 奇异当且仅当这些点仿射相关，此时跳过（返回 None）。
    """
    k = len(combo)
    pts = [unique_points[i] for i in combo]
    p0 = pts[0]
    if k == 1:
        return p0, Fraction(0), (Fraction(1),)

    vectors = [
        (pts[j][0] - p0[0], pts[j][1] - p0[1], pts[j][2] - p0[2])
        for j in range(1, k)
    ]
    size = k - 1
    gram = [
        [
            vectors[a][0] * vectors[b][0]
            + vectors[a][1] * vectors[b][1]
            + vectors[a][2] * vectors[b][2]
            for b in range(size)
        ]
        for a in range(size)
    ]
    rhs = [_norm2(v) / 2 for v in vectors]
    alpha = _solve_linear(gram, rhs)
    if alpha is None:
        return None

    center = tuple(
        p0[d] + sum((alpha[j] * vectors[j][d] for j in range(size)), Fraction(0))
        for d in range(3)
    )
    weights = (Fraction(1) - sum(alpha, Fraction(0)), *alpha)
    return center, _dist2(center, p0), weights


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def minimum_enclosing_sphere(points, *, ids=None, budget=None) -> MinimumEnclosingSphere:
    """计算三维有理点集的最小包围球及其最优证书。

    Args:
        points: 1..30 个点的有限序列；每个点是长度恰为 3 的序列，坐标为
            int / Fraction（或有限 float，按其精确二进制值转 Fraction）。
            bool 坐标拒绝；NaN / Infinity 拒绝并给出可定位下标。
        ids: 可选，与 points 等长、元素可哈希且互不重复；缺省为 0..n-1。
        budget: 可选正整数，最多检查的候选子集数。耗尽时抛
            SearchBudgetExhaustedError（结果未知）；与“证明无解”的
            NoFeasibleCandidateError 严格区分。

    Returns:
        MinimumEnclosingSphere：精确中心、r^2、带原始 ID 来源与非负重心
        系数（和为 1）的支撑点列表。

    Raises:
        TypeError / ValueError: 见各信息，均带参数路径（如 points[2][0]）。
        SearchBudgetExhaustedError: budget 耗尽。
        NoFeasibleCandidateError: 穷举证明无可行候选（非空输入下不会发生）。
    """
    normalized = _normalize_points(points)
    n = len(normalized)
    original_ids = _normalize_ids(ids, n)
    budget_value = _normalize_budget(budget)

    # 坐标去重但保留全部原始 ID 来源；首次出现顺序保证枚举确定性。
    provenance: dict[tuple, list] = {}
    unique_points: list[tuple[Fraction, Fraction, Fraction]] = []
    for pid, p in zip(original_ids, normalized):
        if p not in provenance:
            provenance[p] = []
            unique_points.append(p)
        provenance[p].append(pid)

    m = len(unique_points)
    best = None  # (r2, k, combo, center, weights)
    examined = 0
    for k in range(1, min(4, m) + 1):
        for combo in combinations(range(m), k):
            examined += 1
            if budget_value is not None and examined > budget_value:
                raise SearchBudgetExhaustedError(
                    f"search budget of {budget_value} candidate(s) exhausted "
                    f"before enumeration completed; result is unknown "
                    f"(examined {examined - 1}, unique points {m})"
                )
            found = _candidate(unique_points, combo)
            if found is None:
                continue
            center, r2, weights = found
            if any(w < 0 for w in weights):
                continue  # 中心不在支撑凸包内，不构成最优证书
            if any(_dist2(q, center) > r2 for q in unique_points):
                continue  # 不能包围全部点
            key = (r2, k, combo)
            if best is None or _less(key, best):
                best = (r2, k, combo, center, weights)

    if best is None:
        # 数学上不可达：单点子集 k=1 对任意非空点集至少有一个可行候选
        # （全部点重合时即为该点）。
        raise NoFeasibleCandidateError(
            "exhaustive enumeration proved no feasible enclosing sphere exists"
        )

    r2, _k, combo, center, weights = best
    support = tuple(
        SupportPoint(
            point=unique_points[idx],
            ids=tuple(provenance[unique_points[idx]]),
            weight=weights[pos],
        )
        for pos, idx in enumerate(combo)
    )
    return MinimumEnclosingSphere(
        center=center,
        radius_squared=r2,
        support=support,
        num_input_points=n,
        num_unique_points=m,
    )


def _less(key, best):
    """比较 (r2, k, combo)；平局时支撑更少优先，再按下标字典序。"""
    if key[0] != best[0]:
        return key[0] < best[0]
    if key[1] != best[1]:
        return key[1] < best[1]
    return key[2] < best[2]


# ===========================================================================
# 测量球（带半径的输入球）的最小覆盖球 —— 仅限 1D / 2D、1..6 个球
# ===========================================================================
#
# 问题：给定输入球 B_i = (p_i, r_i)，求使每个 B_i 都被包含的最小球
# (c, R)，即对每个 i 独立满足  |c - p_i| + r_i <= R。
#
# 几何事实（决定算法）：最小覆盖球的边界由 1 个、2 个或（仅 2D）3 个
# 输入球唯一确定，且这些输入球与结果球 *相切*（|c-p_i| = R - r_i）。
#   1D：解恒为区间 [min_i (p_i-r_i), max_i (p_i+r_i)] 的中点/半径，
#       最多 2 个支撑，可用 Fraction 精确计算。
#   2D：
#     - 1 支撑：c = p_i，R = r_i（某球完全覆盖其余全部球）；
#     - 2 支撑 i,j：c 在 p_i,p_j 连线上且满足 R-r_i = |c-p_i|、
#       R-r_j = |c-p_j|。令 u 为 p_i→p_j 单位向量、d=|p_j-p_i|，
#       则 c = p_i + t u，t = (d + r_i - r_j)/2，R = r_i + t。
#       t<0 表示 i 内含 j，单球 i 才是该情形的解，跳过（1 支撑已枚举）。
#     - 3 支撑 i,j,k：外切阿波罗尼乌斯方程 |c-p_i| = R-r_i。对 i,j 与
#       i,k 线性化（R 消去）得到两条直线，求交得到 c 关于基 {u1,u2}
#       的坐标，再代回求 R；要求所有 rho_a = R-r_a >= 0。
#   枚举 ≤6 个球的全部 1/2/3 元组（共 41 个候选），逐个独立核验
#   |c-p_i|+r_i <= R+tol，取 R 最小者（平局取支撑更少、下标字典序）。
#
# 2D 路径是 float64 数值计算；合法性（非负半径、有限数、维度一致等）
# 全部在任何求解之前校验，失败不会产生任何部分结果。本路径无持久状态。


@dataclass(frozen=True)
class BallSupport:
    """结果球边界上的一个输入球（相切支撑）。

    Attributes:
        center: 该输入球的中心（Fraction 元组；2D 结果中的权重为 float）。
        radius: 该输入球的半径（Fraction）。
        ids:    映射到该输入球（中心+半径全等）的全部原始 ID。
        weight: 1D 精确支撑系数（和为 1）；2D 数值路径下恒为 None。
    """

    center: tuple
    radius: Fraction
    ids: tuple
    weight: Fraction | None


@dataclass(frozen=True)
class MinimumEnclosingBall:
    """测量球集合的最小覆盖球结果。

    1D（``dimension == 1``）：``center``/``radius`` 为精确 Fraction，
    ``BallSupport.weight`` 为精确非负系数（和为 1），所有判定零容差。
    2D（``dimension == 2``）：``center`` 为 float 二元组、``radius`` 为
    float；``weight`` 为 None。精确字段（``centers_exact`` 等）仅记录
    经校验的输入，几何结果以 float 给出并按 ``tol`` 核验。
    """

    center: tuple
    radius: Fraction | float
    dimension: int
    support: tuple[BallSupport, ...]
    num_input_balls: int
    num_unique_balls: int
    tol: Fraction | float
    method: str
    centers_exact: tuple = ()
    radii_exact: tuple = ()

    @property
    def center_float(self) -> tuple[float, ...]:
        """中心的 float64 视图（1D 时由精确值一次舍入）。"""
        return tuple(float(x) for x in self.center)

    @property
    def radius_float(self) -> float:
        """半径的 float64 视图（1D 时由精确值一次舍入）。"""
        return float(self.radius)

    def max_violation(self, balls=None) -> float:
        """max_i (|c-p_i| + r_i - R)；应 <= tol（1D 精确解下 <= 0）。

        ``balls`` 为 None 时使用结果保存的、经校验的精确输入副本。
        每个距离+半径约束都在此被独立重算，不使用求解时的缓存。
        """
        if balls is None:
            centers, radii = self.centers_exact, self.radii_exact
        else:
            pairs = _balls_to_exact(balls, self.dimension, where="max_violation()")
            centers, radii = pairs
        cf = tuple(float(x) for x in self.center)
        R = float(self.radius)
        worst = -math.inf
        for p, r in zip(centers, radii):
            d = math.sqrt(sum((float(p[a]) - cf[a]) ** 2 for a in range(self.dimension)))
            worst = max(worst, d + float(r) - R)
        return worst

    def covers(self, ball, *, tol=None) -> bool:
        """独立判定给定输入球是否被结果球覆盖（边界算覆盖）。

        2D 默认按结果的 ``tol`` 判定：``|c-p| + r <= R + tol``；
        1D 为 Fraction 精确比较（tol 不参与）。
        """
        p, r = _one_ball_exact(ball, self.dimension, where="covers() argument")
        if self.dimension == 1:
            return abs(self.center[0] - p[0]) + r <= self.radius
        eps = float(self.tol) if tol is None else float(tol)
        cf = tuple(float(x) for x in self.center)
        d = math.sqrt(sum((float(p[a]) - cf[a]) ** 2 for a in range(2)))
        return d + float(r) <= float(self.radius) + eps


def _validate_tol(tol) -> float:
    if isinstance(tol, bool):
        raise TypeError(f"tol: bool is not accepted (got {tol!r})")
    if isinstance(tol, Rational):
        value = float(tol)
    elif isinstance(tol, float):
        value = tol
    else:
        raise TypeError("tol: non-negative real number expected (int/float/Fraction)")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"tol: must be a finite non-negative number, got {tol!r}")
    return value


def _validate_dimension(dim) -> int:
    if isinstance(dim, bool):
        raise TypeError(f"dimension: bool is not accepted (got {dim!r})")
    if not isinstance(dim, int):
        raise TypeError("dimension: 1 or 2 expected (int)")
    if dim not in (1, 2):
        raise ValueError(f"dimension: only 1 or 2 are supported, got {dim}")
    return dim


def _one_ball_exact(ball, dim, *, where):
    """校验单个输入球 (center, radius)，返回 (Fraction 中心元组, Fraction 半径)。"""
    if isinstance(ball, (str, bytes)) or not hasattr(ball, "__len__"):
        raise TypeError(f"{where}: a (center, radius) pair is required")
    if len(ball) != 2:
        raise ValueError(f"{where}: expected a (center, radius) pair, got length {len(ball)}")
    center_seq, radius = ball[0], ball[1]
    if isinstance(center_seq, (str, bytes)) or not hasattr(center_seq, "__len__"):
        raise TypeError(f"{where}[0]: center sequence of {dim} coordinate(s) expected")
    if len(center_seq) != dim:
        raise ValueError(
            f"{where}[0]: expected {dim} coordinate(s), got {len(center_seq)}"
        )
    center = tuple(
        _coerce_scalar(v, where=f"{where}[0][{i}]") for i, v in enumerate(center_seq)
    )
    if isinstance(radius, bool):
        raise TypeError(f"{where}[1]: bool is not accepted as radius")
    if isinstance(radius, Rational):
        rf = Fraction(radius.numerator, radius.denominator)
    elif isinstance(radius, float):
        if not math.isfinite(radius):
            raise ValueError(
                f"{where}[1]: NaN/Infinity is not allowed (got {radius!r})"
            )
        rf = Fraction(radius)
    else:
        raise TypeError(
            f"{where}[1]: non-negative rational radius expected, "
            f"got {type(radius).__name__}"
        )
    if rf < 0:
        raise ValueError(f"{where}[1]: radius must be non-negative, got {rf}")
    return center, rf


def _balls_to_exact(balls, dim, *, where):
    return list(
        zip(
            *(
                _one_ball_exact(b, dim, where=f"{where}[{i}]")
                for i, b in enumerate(balls)
            )
        )
    ) or ([], [])


def _normalize_balls(balls, dimension):
    if isinstance(balls, (str, bytes)) or not hasattr(balls, "__len__"):
        raise TypeError("balls: a finite sequence of (center, radius) balls is required")
    n = len(balls)
    if n == 0:
        raise ValueError("balls: at least 1 ball is required")
    if n > MIN_BALL_MAX_BALLS:
        raise ValueError(
            f"balls: at most {MIN_BALL_MAX_BALLS} balls are allowed, got {n}"
        )
    if dimension is None:
        first = balls[0]
        if isinstance(first, (str, bytes)) or not hasattr(first, "__len__"):
            raise TypeError("balls[0]: a (center, radius) pair is required")
        if len(first) != 2:
            raise ValueError(
                f"balls[0]: expected a (center, radius) pair, got length {len(first)}"
            )
        center_seq = first[0]
        if isinstance(center_seq, (str, bytes)) or not hasattr(center_seq, "__len__"):
            raise TypeError("balls[0][0]: center sequence expected")
        dim = len(center_seq)
        if dim not in (1, 2):
            raise ValueError(
                f"balls[0][0]: only 1D/2D balls are supported, got {dim} coordinate(s)"
            )
    else:
        dim = _validate_dimension(dimension)
    centers, radii = _balls_to_exact(balls, dim, where="balls")
    return centers, list(radii), n, dim


def minimum_enclosing_ball_of_balls(balls, *, ids=None, tol=DEFAULT_BALL_TOL, dimension=None):
    """求覆盖全部输入*测量球*的最小球（1D 精确 / 2D 带显式容差）。

    Args:
        balls: 1..6 个球的有限序列，每个球是 ``(center, radius)``：
            center 长度为 1 或 2（所有球维度必须一致；可由第一个球推断，
            或用 dimension= 显式指定），radius 为非负有限有理数
            （int/Fraction，或有限 float 按其二进制值精确转 Fraction）。
            半径全为 0 时问题退化为点集最小包围球。
        ids: 可选，与 balls 等长、可哈希且互不重复；缺省 0..n-1。
            中心与半径全等的球共享支撑记录上的全部来源 ID。
        tol: 2D 数值求解的非负有限容差（默认 1e-10）。候选可行条件为
            对每个球 *独立* 成立 ``|c-p_i| + r_i <= R + tol``。
            1D 为 Fraction 精确求解，tol 不参与几何判定。
        dimension: 可选的显式维度（1 或 2）；缺省按第一个球中心长度推断。

    Returns:
        MinimumEnclosingBall。2D 下另可用 :meth:`max_violation` 查看最坏
        约束违反量（求解后独立重算，不含任何缓存）。

    Raises:
        TypeError / ValueError: 非法输入，信息均含参数路径
            （如 ``balls[2][1]``、``balls[0][0][1]``）。所有校验在求解前
            完成，失败时无任何状态被修改（本函数本身无状态、不改输入）。
    """
    centers, radii, n, dim = _normalize_balls(balls, dimension)
    original_ids = _normalize_ids(ids, n, what="balls")
    tol_value = _validate_tol(tol)

    # 全等球（中心+半径）去重，保留全部原始 ID；首次出现序保证确定性。
    provenance: dict[tuple, list] = {}
    uniq_center: list[tuple] = []
    uniq_radius: list[Fraction] = []
    for pid, c, r in zip(original_ids, centers, radii):
        key = (c, r)
        if key not in provenance:
            provenance[key] = []
            uniq_center.append(c)
            uniq_radius.append(r)
        provenance[key].append(pid)

    if dim == 1:
        return _solve_1d(uniq_center, uniq_radius, provenance, n)
    return _solve_2d(uniq_center, uniq_radius, provenance, n, tol_value)


# ---------------------------------------------------------------------------
# 1D：精确。把每球看作区间 [p-r, p+r]，覆盖球即覆盖区间并集的最小区间。
# ---------------------------------------------------------------------------

def _solve_1d(uniq_center, uniq_radius, provenance, n):
    lefts = [c[0] - r for c, r in zip(uniq_center, uniq_radius)]
    rights = [c[0] + r for c, r in zip(uniq_center, uniq_radius)]
    lo_idx = min(range(len(uniq_center)), key=lambda i: (lefts[i], i))
    hi_idx = min(range(len(uniq_center)), key=lambda i: (-rights[i], i))
    lo, hi = lefts[lo_idx], rights[hi_idx]
    center = ((lo + hi) / 2,)
    radius = (hi - lo) / 2

    if lo_idx == hi_idx:
        # 单个球覆盖全部（区间并集就是它自己）。
        combo = (lo_idx,)
        weights = (Fraction(1),)
    else:
        combo = (lo_idx, hi_idx) if lo_idx < hi_idx else (hi_idx, lo_idx)
        weights = (Fraction(1, 2), Fraction(1, 2))

    support = tuple(
        BallSupport(
            center=uniq_center[idx],
            radius=uniq_radius[idx],
            ids=tuple(provenance[(uniq_center[idx], uniq_radius[idx])]),
            weight=weights[pos],
        )
        for pos, idx in enumerate(combo)
    )
    return MinimumEnclosingBall(
        center=center,
        radius=radius,
        dimension=1,
        support=support,
        num_input_balls=n,
        num_unique_balls=len(uniq_center),
        tol=Fraction(0),
        method="1d-exact-interval",
        centers_exact=tuple(uniq_center),
        radii_exact=tuple(uniq_radius),
    )


# ---------------------------------------------------------------------------
# 2D：float64 数值枚举 + 显式容差，逐约束独立核验
# ---------------------------------------------------------------------------

def _f2(p):
    return (float(p[0]), float(p[1]))


def _fadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _fsub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _fscale(t, a):
    return (t * a[0], t * a[1])


def _fdot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _fdist(a, b):
    d = _fsub(a, b)
    return math.sqrt(_fdot(d, d))


def _candidate_pair(p_i, r_i, p_j, r_j):
    """两球外切确定的候选；退化为内含（t<0）或同心时返回 None。

    c = p_i + t u，约束 t + r_i = R、(d-t) + r_j = R：
    t = (d + r_j - r_i)/2，R = r_i + t。t<0 表示 j 内含于 i，
    此时正确解是单球 i（已单独枚举）。
    """
    v = _fsub(p_j, p_i)
    d = math.sqrt(_fdot(v, v))
    if d == 0.0:
        return None  # 全等已去重；半径不同的同心圆由单球候选覆盖
    t = (d + r_j - r_i) / 2.0
    if t < 0.0:
        return None
    u = (v[0] / d, v[1] / d)
    center = _fadd(p_i, _fscale(t, u))
    radius = r_i + t
    return center, radius


def _candidate_triple(centers, radii, triple):
    """三球外切阿波罗尼乌斯候选；返回全部 rho>=0 的 (center, R) 根。

    平方相切方程 |c-p_i|^2 = (R-r_i)^2，对 (j-i),(k-i) 相减消去 R²。
    令 c = p0 + x u1 + y u2（u1=p1-p0, u2=p2-p0），常数项中的
    2 p0·u 项相消，得到：
        2x u1·u1 + 2y u1·u2 = |u1|² - (r1²-r0²) + 2R(r1-r0)
        2x u1·u2 + 2y u2·u2 = |u2|² - (r2²-r0²) + 2R(r2-r0)
    解出 x=xv+xr R, y=yv+yr R，再代入 |x u1+y u2| = R-r0 得 R 的二次。
    共线/退化返回 []；只回传有限且 rho0,rho1,rho2>=0 的外切根。
    """
    i, j, k = triple
    p0 = centers[i]
    r0, r1, r2 = radii[i], radii[j], radii[k]
    u1 = _fsub(centers[j], p0)
    u2 = _fsub(centers[k], p0)
    det = u1[0] * u2[1] - u1[1] * u2[0]
    if det == 0.0 or not math.isfinite(det):
        return []  # 三中心共线：2 球/单球候选已覆盖

    g11, g22, g12 = _fdot(u1, u1), _fdot(u2, u2), _fdot(u1, u2)
    A11, A12, A22 = 2.0 * g11, 2.0 * g12, 2.0 * g22
    b1v = g11 - (r1 * r1 - r0 * r0)
    b2v = g22 - (r2 * r2 - r0 * r0)
    b1r = 2.0 * (r1 - r0)
    b2r = 2.0 * (r2 - r0)
    detA = A11 * A22 - A12 * A12
    if detA == 0.0 or not math.isfinite(detA):
        return []
    inv = 1.0 / detA
    xv = (A22 * b1v - A12 * b2v) * inv
    xr = (A22 * b1r - A12 * b2r) * inv
    yv = (-A12 * b1v + A11 * b2v) * inv
    yr = (-A12 * b1r + A11 * b2r) * inv

    vconst = _fadd(_fscale(xv, u1), _fscale(yv, u2))
    vlin = _fadd(_fscale(xr, u1), _fscale(yr, u2))
    # |vconst + R vlin|² = (R - r0)²
    aa = _fdot(vlin, vlin) - 1.0
    bb = 2.0 * (_fdot(vconst, vlin) + r0)
    cc = _fdot(vconst, vconst) - r0 * r0

    roots = []
    if aa == 0.0:
        if bb != 0.0:
            roots.append(-cc / bb)
    else:
        disc = bb * bb - 4.0 * aa * cc
        if math.isfinite(disc) and disc >= 0.0:
            root = math.sqrt(disc)
            roots.extend(((-bb + root) / (2.0 * aa), (-bb - root) / (2.0 * aa)))

    out = []
    for R in roots:
        if not math.isfinite(R):
            continue
        x, y = xv + xr * R, yv + yr * R
        center = _fadd(p0, _fadd(_fscale(x, u1), _fscale(y, u2)))
        if not all(math.isfinite(v) for v in center):
            continue
        rho_min = min(R - r0, R - r1, R - r2)
        # 真内切时 rho=0，机器舍入可能给出 -1e-14，按相对 epsilon 放行。
        if rho_min < -1e-12 * (1.0 + abs(R)):
            continue
        out.append((center, R))
    return out


def _solve_2d(uniq_center, uniq_radius, provenance, n, tol_value):
    centers = [_f2(c) for c in uniq_center]
    radii = [float(r) for r in uniq_radius]
    m = len(centers)

    def machine_slack(radius, center):
        # float64 舍入余量，与用户契约容差分开：坐标 ~1e3 时约 1e-11，
        # 永远远小于默认 tol=1e-10；tol=0 的严格模式下它也只掩盖机器误差。
        return 1e-12 * (1.0 + abs(radius) + math.hypot(center[0], center[1]))

    def feasible(center, radius):
        """逐球独立核验距离+半径约束；任一不过即否决。"""
        slack = tol_value + machine_slack(radius, center)
        for q, s in zip(centers, radii):
            if _fdist(center, q) + s > radius + slack:
                return False
        return True

    best = None  # ((R, k, combo), center, radius)

    def consider(combo, got):
        nonlocal best
        for center, radius in got:
            if not math.isfinite(radius) or not all(math.isfinite(v) for v in center):
                continue
            if radius < -machine_slack(radius, center):
                continue
            if not feasible(center, radius):
                continue
            key = (radius, len(combo), combo)
            if best is None or key < best[0]:
                best = (key, center, radius)

    for i in range(m):
        consider((i,), [(centers[i], radii[i])])
    for i, j in combinations(range(m), 2):
        got = _candidate_pair(centers[i], radii[i], centers[j], radii[j])
        consider((i, j), [] if got is None else [got])
    for triple in combinations(range(m), 3):
        consider(triple, _candidate_triple(centers, radii, triple))

    if best is None:
        raise NoFeasibleCandidateError(
            "no feasible covering ball found within the stated tolerance"
        )

    _key, center, radius = best
    combo = _key[2]
    support = tuple(
        BallSupport(
            center=uniq_center[idx],
            radius=uniq_radius[idx],
            ids=tuple(provenance[(uniq_center[idx], uniq_radius[idx])]),
            weight=None,
        )
        for idx in combo
    )

    # 独立终检：用 Fraction 精确输入重新转 float，用 math.fsum 重算每一个
    # 约束（独立的舍入路径），与候选筛选完全分离，避免同一计算路径的
    # 误差互相“背书”。机器 epsilon 余量只用于吸收 float64 舍入。
    slack = tol_value + machine_slack(radius, center)
    worst_slack = math.inf
    for q_exact, r_exact in zip(uniq_center, uniq_radius):
        q = _f2(q_exact)
        d = math.sqrt(math.fsum((q[a] - center[a]) ** 2 for a in range(2)))
        expr = math.fsum([d, float(r_exact), -radius])
        worst_slack = min(worst_slack, slack - expr)
    if worst_slack < 0.0:
        raise NoFeasibleCandidateError(
            f"independent final verification failed; worst slack {worst_slack:.3e}"
        )

    return MinimumEnclosingBall(
        center=center,
        radius=radius,
        dimension=2,
        support=support,
        num_input_balls=n,
        num_unique_balls=m,
        tol=tol_value,
        method="2d-numeric-enum",
        centers_exact=tuple(uniq_center),
        radii_exact=tuple(uniq_radius),
    )
