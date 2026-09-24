"""miniball3d — 最小包围球：三维有理点集的精确证书，及一维/二维输入球的数值解。

仅使用 Python 标准库（fractions / itertools / dataclasses / math）。

公开接口
========
    minimum_enclosing_sphere(points, *, ids=None, budget=None)
        -> MinimumEnclosingSphere
        三维、1..30 个有理点；Fraction 精确枚举 + 最优证书（见下）。

    minimum_enclosing_ball(balls, *, dim=2, ids=None, tol=1e-10)
        -> MinimumEnclosingBall
        一维或二维（dim ∈ {1,2}）、1..6 个输入球 (center, radius)；
        float64 数值枚举求解，容差 tol 显式可配；半径 0 的球即点，
        退化为点集最小包围球。

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
    "MIN_BALL_TOL",
    "SupportPoint",
    "MinimumEnclosingSphere",
    "InputBall",
    "SupportBall",
    "MinimumEnclosingBall",
    "MiniballError",
    "NoFeasibleCandidateError",
    "SearchBudgetExhaustedError",
    "minimum_enclosing_sphere",
    "minimum_enclosing_ball",
]

#: 接口允许的输入点数量上限（题面规定 30）。
MIN_BALL_MAX_POINTS = 30

#: 输入球接口允许的球数量上限（本次迭代题面规定 6）。
MIN_BALL_MAX_BALLS = 6

#: minimum_enclosing_ball 的默认相对/绝对判定容差。
MIN_BALL_TOL = 1e-10


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
# 输入球（一维/二维）的数值最小包围球
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SupportBall:
    """一个紧贴包围球的输入球及其证书信息。

    Attributes:
        center: 输入球心（float 元组，长度等于 dim）。
        radius: 输入球半径（float，非负）。
        ids:    映射到该输入球的全部原始 ID（通常只有一个）。
        weight: KKT 权重 λ≥0；全部支撑权重之和为 1，且满足
            ``sum_i λ_i · (c-p_i)/|c-p_i| = 0``（法向力平衡）。
            点输入（半径 0）退化为旧点集证书的重心系数。
    """

    center: tuple[float, ...]
    radius: float
    ids: tuple
    weight: float


@dataclass(frozen=True)
class MinimumEnclosingBall:
    """一维/二维输入球的最小包围球结果（float64 数值解）。"""

    center: tuple[float, ...]
    radius: float
    support: tuple[SupportBall, ...]
    num_input_balls: int
    tol: float
    #: 归一化后的输入球（(center, radius) 元组），按输入顺序保存，
    #: 供使用方独立复核每个约束，不依赖调用现场的对象。
    balls: tuple[tuple[tuple[float, ...], float], ...]

    @property
    def weights(self) -> tuple[float, ...]:
        """与 support 同序的 KKT 权重。"""
        return tuple(s.weight for s in self.support)

    @property
    def constraint_values(self) -> tuple[float, ...]:
        """逐输入球的 ``距离(球心) + 输入半径``，按输入顺序排列。

        包围球合法当且仅当每个值都 ≤ ``radius``（在 tol 口径内）。
        """
        return tuple(
            _dist_float(self.center, p) + ri for p, ri in self.balls
        )

    @property
    def slacks(self) -> tuple[float, ...]:
        """逐输入球的松弛量 ``radius - (距离 + 输入半径)``；支撑球约为 0。"""
        return tuple(self.radius - v for v in self.constraint_values)

    def max_violation(self) -> float:
        """``max(距离+半径 − radius)``；> tol 口径即有输入球越界。"""
        return max(self.constraint_values, default=0.0) - self.radius

    def balance_residual(self) -> tuple[float, ...]:
        """支撑法向力平衡残差 ``sum λ_i (c-p_i)/|c-p_i|``（数值上应约为 0）。

        球心恰好落在某个支撑球心上（该支撑 dist=0，例如单球包围全部）时，
        该方向无定义，按无平衡条件处理并返回全 0。
        """
        dim = len(self.center)
        acc = [0.0] * dim
        active = False
        for s in self.support:
            d = _dist_float(self.center, s.center)
            if d == 0.0:
                continue
            active = True
            for k in range(dim):
                acc[k] += s.weight * (self.center[k] - s.center[k]) / d
        return tuple(acc) if active else tuple(0.0 for _ in range(dim))

    def contains_point(self, point, *, tol=None) -> bool:
        """点是否在包围球内（边界算包含）；tol 缺省用结果的 tol。"""
        tol = self.tol if tol is None else tol
        q = _validate_float_coordinate_sequence(
            point, dim=len(self.center), where="contains_point() argument"
        )
        scale = max(1.0, max((abs(x) for x in q), default=0.0), self.radius)
        return _dist_float(q, self.center) <= self.radius + tol * scale

    def contains_ball(self, center, radius, *, tol=None) -> bool:
        """输入球是否被整体包含：``dist(c, center) + radius ≤ radius``。"""
        tol = self.tol if tol is None else tol
        p, r = _validate_one_ball(
            (center, radius), dim=len(self.center), index=0, prefix="contains_ball()"
        )
        scale = max(
            1.0, self.radius, r, max((abs(x) for x in p), default=0.0)
        )
        return _dist_float(p, self.center) + r <= self.radius + tol * scale


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


def _normalize_ids(ids, n: int, *, noun: str = "points"):
    if ids is None:
        return tuple(range(n))
    if isinstance(ids, (str, bytes)) or not hasattr(ids, "__len__"):
        raise TypeError(f"ids: a finite sequence aligned with {noun} is required")
    if len(ids) != n:
        raise ValueError(
            f"ids: length {len(ids)} does not match number of {noun} {n}"
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


# ---------------------------------------------------------------------------
# 输入球接口：float 数值工具
# ---------------------------------------------------------------------------

def _to_float(value, where: str) -> float:
    """int/Fraction/有限 float 统一转 float；bool、NaN/Inf、字符串显式拒绝。"""
    _reject_bool(value, where)
    if isinstance(value, Rational):
        try:
            # Fraction.__float__ 做一次正确舍入；int 同样走这里。
            out = float(value)
        except OverflowError as exc:
            raise ValueError(
                f"{where}: value too large to represent as float64"
            ) from exc
    elif isinstance(value, float):
        out = value
    else:
        raise TypeError(
            f"{where}: real number expected (int/float/Fraction), "
            f"got {type(value).__name__}"
        )
    if not math.isfinite(out):
        raise ValueError(f"{where}: NaN/Infinity is not allowed (got {value!r})")
    return out


def _validate_float_coordinate_sequence(point, *, dim: int, where: str):
    if isinstance(point, (str, bytes)) or not hasattr(point, "__len__"):
        raise TypeError(f"{where}: sequence of {dim} coordinate(s) expected")
    if len(point) != dim:
        raise ValueError(
            f"{where}: expected exactly {dim} coordinate(s), got {len(point)}"
        )
    return tuple(
        _to_float(v, where=f"{where}[{i}]") for i, v in enumerate(point)
    )


def _validate_one_ball(ball, *, dim: int, index: int, prefix: str = "balls"):
    """校验单个输入球 (center, radius)，返回 (center_tuple_float, radius_float)。"""
    where = f"{prefix}[{index}]"
    if isinstance(ball, (str, bytes)) or not hasattr(ball, "__len__"):
        raise TypeError(f"{where}: a (center, radius) pair is expected")
    if len(ball) != 2:
        raise ValueError(
            f"{where}: expected a (center, radius) pair with 2 elements, "
            f"got {len(ball)}"
        )
    center = _validate_float_coordinate_sequence(
        ball[0], dim=dim, where=f"{where}[0](center)"
    )
    radius = _to_float(ball[1], where=f"{where}[1](radius)")
    if radius < 0.0:
        raise ValueError(
            f"{where}[1](radius): radius must be non-negative, got {radius}"
        )
    return center, radius


def _normalize_dim(dim) -> int:
    _reject_bool(dim, "dim")
    if not isinstance(dim, int):
        raise TypeError("dim: 1 or 2 expected (int)")
    if dim not in (1, 2):
        raise ValueError(f"dim: only 1 or 2 are supported, got {dim}")
    return dim


def _normalize_tol(tol) -> float:
    if tol is None:
        return MIN_BALL_TOL
    value = _to_float(tol, "tol")
    if value < 0.0:
        raise ValueError(f"tol: must be non-negative, got {tol}")
    return value


def _dist_float(a, b) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


# ---------------------------------------------------------------------------
# 输入球最小包围球：候选求解（dim ∈ {1,2}，k ≤ dim+1 个活动球）
# ---------------------------------------------------------------------------
#
# 活动约束 |c - p_i| = r - a_i（a_i 为输入球半径）。参数化
# c = p0 + V α，线性化等距条件得 Gram 方程 G α = β + r γ：
#   β_j = (|v_j|² - a_j² + a_0²)/2，γ_j = a_j - a_0，v_j = p_j - p0。
# 仿射独立 ⇔ G 非奇异。令 α = u + r w（u=G⁻¹β, w=G⁻¹γ），代入
# |c-p0| = r - a_0 得到关于 r 的一元二次方程：
#   (wᵀGw - 1) r² + 2(uᵀGw + a_0) r + (uᵀGu - a_0²) = 0。
# 每个实根还要过两道独立的关：r ≥ max 活动半径；非负 KKT 权重；
# 之后再逐球独立检查 dist(c,p) + a ≤ r（容差 tol）。

def _solve2(A, b):
    """float 高斯消元；规模 1 或 2。奇异（近零主元）返回 None。"""
    n = len(A)
    mat = [list(A[i]) + [b[i]] for i in range(n)]
    scale = max(
        (abs(mat[i][j]) for i in range(n) for j in range(n)),
        default=0.0,
    )
    pivot_eps = 1e-14 * max(1.0, scale)
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) <= pivot_eps:
            return None
        if pivot != col:
            mat[col], mat[pivot] = mat[pivot], mat[col]
        pv = mat[col][col]
        for r in range(col + 1, n):
            factor = mat[r][col] / pv
            mat[r] = [
                mat[r][j] - factor * mat[col][j] for j in range(n + 1)
            ]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (
            mat[i][n] - sum(mat[i][j] * x[j] for j in range(i + 1, n))
        ) / mat[i][i]
    return x


def _candidate_balls(centers, radii, combo, tol):
    """枚举一个支撑子集，返回候选列表 (center, r, weights_in_combo)。

    k=3 时二次方程可能有两个实根，故返回列表；不可行/退化返回空。
    """
    k = len(combo)
    pts = [centers[i] for i in combo]
    rad = [radii[i] for i in combo]
    dim = len(pts[0])

    if k == 1:
        return [(pts[0], rad[0], (1.0,))]

    p0, a0 = pts[0], rad[0]
    vecs = [
        tuple(pts[j][d] - p0[d] for d in range(dim)) for j in range(1, k)
    ]
    size = k - 1
    gram = [
        [sum(vecs[a][d] * vecs[b][d] for d in range(dim)) for b in range(size)]
        for a in range(size)
    ]
    beta = [
        (sum(x * x for x in vecs[j]) - rad[j + 1] ** 2 + a0 ** 2) / 2.0
        for j in range(size)
    ]
    gamma = [rad[j + 1] - a0 for j in range(size)]

    u = _solve2(gram, beta)
    if u is None:
        return []  # 球心仿射相关（2D 三心共线等），跳过该子集
    w = _solve2(gram, gamma)
    if w is None:
        return []

    def center_of(alpha):
        return tuple(
            p0[d] + sum(alpha[j] * vecs[j][d] for j in range(size))
            for d in range(dim)
        )

    wGw = sum(
        w[a] * sum(gram[a][b] * w[b] for b in range(size)) for a in range(size)
    )
    uGw = sum(
        u[a] * sum(gram[a][b] * w[b] for b in range(size)) for a in range(size)
    )
    uGu = sum(
        u[a] * sum(gram[a][b] * u[b] for b in range(size)) for a in range(size)
    )

    A = wGw - 1.0
    B = uGw + a0
    C = uGu - a0 ** 2
    roots = []
    near = max(1e-14, tol) * max(1.0, abs(B * B), abs(A * C))
    if abs(A) <= 1e-14 * max(1.0, abs(wGw), 1.0):
        # 退化为线性方程 2 B r + C = 0
        if abs(B) > 1e-14 * max(1.0, abs(C)):
            roots.append(-C / (2.0 * B))
    else:
        disc = B * B - A * C
        if disc < -near:
            return []
        if disc < 0.0:
            disc = 0.0
        s = math.sqrt(disc)
        roots.append((-B - s) / A)
        roots.append((-B + s) / A)

    candidates = []
    min_active_r = max(rad)
    for r in roots:
        if not math.isfinite(r) or r < min_active_r - tol * max(1.0, r):
            continue
        alpha = [u[j] + r * w[j] for j in range(size)]
        c = center_of(alpha)
        dists = [_dist_float(c, p) for p in pts]
        # 任一活动球心与 c 重合而 k>1：法向无定义，等价情形由 k=1 候选覆盖
        if any(d <= 1e-14 * max(1.0, r) for d in dists):
            continue
        weights = _kkt_weights(pts, c, dists, tol)
        if weights is None:
            continue
        candidates.append((c, r, weights))
    return candidates


def _kkt_weights(pts, c, dists, tol):
    """解 Σ λ_i n_i = 0、Σ λ_i = 1 的非负权重；不成立返回 None。

    k=2：法向必须反向（c 位于两心之间），权重各 1/2。
    k=3（2D）：解 2x2 线性方程 (n_j - n_0) λ' = -n_0。
    解出后按 tol 口径接受微小负值并钳为 0，再归一化。
    """
    k = len(pts)
    normals = [
        tuple((c[d] - pts[i][d]) / dists[i] for d in range(len(c)))
        for i in range(k)
    ]
    if k == 2:
        dot = sum(normals[0][d] * normals[1][d] for d in range(len(c)))
        # 法向必须反向（c 位于两心线段上的可行区间内）
        if dot > -1.0 + math.sqrt(max(tol, 1e-14)):
            return None
        return (0.5, 0.5)

    # k == 3，dim == 2
    n0 = normals[0]
    A = [
        [normals[j + 1][d] - n0[d] for j in range(2)]
        for d in range(2)
    ]
    rhs = [-n0[d] for d in range(2)]
    sol = _solve2(A, rhs)
    if sol is None:
        return None
    lam = [1.0 - sol[0] - sol[1], sol[0], sol[1]]
    w_eps = math.sqrt(max(tol, 1e-14)) * max(
        1.0, *(abs(x) for x in lam)
    )
    if any(x < -w_eps for x in lam):
        return None
    lam = [max(0.0, x) for x in lam]
    total = sum(lam)
    if total <= 0.0:
        return None
    return tuple(x / total for x in lam)


def minimum_enclosing_ball(
    balls, *, dim: int = 2, ids=None, tol=MIN_BALL_TOL
) -> MinimumEnclosingBall:
    """计算一维或二维输入球的最小包围球（float64 数值解）。

    Args:
        balls: 1..6 个输入球的有限序列，每个球是长度 2 的配对
            ``(center, radius)``：center 是长度恰为 dim 的坐标序列，
            radius 是非负有限实数（0 表示点）。坐标接受 int / Fraction /
            有限 float；bool、NaN / Infinity、负半径拒绝并定位到具体下标。
        dim: 1 或 2（缺省 2）。
        ids: 可选，与 balls 等长、可哈希且互不重复；缺省 0..n-1。
            完全相同的球（球心与半径都相等）去重后合并 ID 来源。
        tol: 显式数值容差（缺省 1e-10）。逐球可行性按
            ``dist + radius ≤ r + tol·max(1, r, |坐标|)`` 判定。

    Returns:
        MinimumEnclosingBall：中心、半径、支撑球与非负 KKT 权重，
        以及可逐球独立复核的 constraint_values / slacks / max_violation。

    Raises:
        TypeError / ValueError: 非法输入，信息含参数路径（如 balls[2][1]）。
        NoFeasibleCandidateError: 枚举后无可行候选（非空输入下不应发生；
            若发生说明容差/病态配置使全部候选被拒，错误信息照实说明）。
    """
    dim_value = _normalize_dim(dim)
    if isinstance(balls, (str, bytes)) or not hasattr(balls, "__len__"):
        raise TypeError("balls: a finite sequence of (center, radius) balls is required")
    n = len(balls)
    if n == 0:
        raise ValueError("balls: at least 1 ball is required")
    if n > MIN_BALL_MAX_BALLS:
        raise ValueError(
            f"balls: at most {MIN_BALL_MAX_BALLS} balls are allowed, got {n}"
        )
    normalized = tuple(
        _validate_one_ball(balls[i], dim=dim_value, index=i) for i in range(n)
    )
    original_ids = _normalize_ids(ids, n, noun="balls")
    tol_value = _normalize_tol(tol)

    # 完全相同的球去重（保留全部 ID 来源），约束对每个原始球等价。
    provenance: dict[tuple, list] = {}
    centers: list[tuple[float, ...]] = []
    radii: list[float] = []
    for pid, (p, r) in zip(original_ids, normalized):
        key = (p, r)
        if key not in provenance:
            provenance[key] = []
            centers.append(p)
            radii.append(r)
        provenance[key].append(pid)

    m = len(centers)
    best = None  # (r, k, combo, center, weights)
    for k in range(1, min(dim_value + 1, m) + 1):
        for combo in combinations(range(m), k):
            if k == 1:
                candidates = [(centers[combo[0]], radii[combo[0]], (1.0,))]
            else:
                candidates = _candidate_balls(
                    centers, radii, combo, tol_value
                )
            for c, r, weights in candidates:
                feasible = True
                for i in range(m):
                    scale = max(
                        1.0, r, radii[i], *(abs(x) for x in c),
                        *(abs(x) for x in centers[i]),
                    )
                    # 逐球独立检查：一个输入球一个谓词，不做混合汇总
                    if _dist_float(c, centers[i]) + radii[i] > r + tol_value * scale:
                        feasible = False
                        break
                if not feasible:
                    continue
                key = (r, k, combo)
                if best is None or key < best[0]:
                    best = (key, c, r, weights, combo)

    if best is None:
        raise NoFeasibleCandidateError(
            "no feasible enclosing ball found within the enumeration; "
            "this can occur for an ill-conditioned configuration at the "
            f"given tol={tol_value!r} (input balls: {m} unique in dim={dim_value})"
        )

    _key, c, r, weights, combo = best
    support = tuple(
        SupportBall(
            center=centers[idx],
            radius=radii[idx],
            ids=tuple(provenance[(centers[idx], radii[idx])]),
            weight=weights[pos],
        )
        for pos, idx in enumerate(combo)
    )
    return MinimumEnclosingBall(
        center=tuple(c),
        radius=r,
        support=support,
        num_input_balls=n,
        tol=tol_value,
        balls=normalized,
    )
