"""miniball3d — 三维有理点集最小包围球的精确求解与可验证最优证书。

仅使用 Python 标准库（fractions / itertools / dataclasses / math）。

公开接口
========
    minimum_enclosing_sphere(points, *, ids=None, budget=None) -> MinimumEnclosingSphere

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
    "SupportPoint",
    "MinimumEnclosingSphere",
    "MiniballError",
    "NoFeasibleCandidateError",
    "SearchBudgetExhaustedError",
    "minimum_enclosing_sphere",
]

#: 接口允许的输入点数量上限（题面规定 30）。
MIN_BALL_MAX_POINTS = 30


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


def _normalize_ids(ids, n: int):
    if ids is None:
        return tuple(range(n))
    if isinstance(ids, (str, bytes)) or not hasattr(ids, "__len__"):
        raise TypeError("ids: a finite sequence aligned with points is required")
    if len(ids) != n:
        raise ValueError(
            f"ids: length {len(ids)} does not match number of points {n}"
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
# 受约束包围球：球心限定在调用方给定的直线或平面上
# ---------------------------------------------------------------------------

__all__ += [
    "ConstrainedMinimumEnclosingSphere",
    "minimum_enclosing_sphere_constrained",
]


@dataclass(frozen=True)
class ConstrainedMinimumEnclosingSphere:
    """球心被限制在指定仿射子空间（直线/平面）上的最小包围球结果。

    几何字段均为精确 Fraction。``support`` 中活动点到球心的真实三维距离
    平方都等于 ``radius_squared``；证书在低维坐标中核验（见
    ``certificate_residual`` / ``dual_identity_radius_squared``）。
    """

    center: tuple[Fraction, Fraction, Fraction]
    radius_squared: Fraction
    support: tuple[SupportPoint, ...]
    num_input_points: int
    num_unique_points: int
    subspace_origin: tuple[Fraction, Fraction, Fraction]
    subspace_directions: tuple[tuple[Fraction, Fraction, Fraction], ...]
    subspace_dimension: int
    #: 枚举中因活动集投影仿射相关（Gram 奇异）而被跳过的候选数。
    #: 大于 0 表示输入在约束子空间上的投影存在数值退化情形。
    degenerate_combinations: int

    @property
    def weights(self) -> tuple[Fraction, ...]:
        """与 support 同序的重心系数；和恰为 1，且全部非负。"""
        return tuple(s.weight for s in self.support)

    @property
    def center_float(self) -> tuple[float, float, float]:
        return tuple(float(x) for x in self.center)

    @property
    def radius_float(self) -> float:
        return math.sqrt(float(self.radius_squared))

    def contains(self, point) -> bool:
        """精确判定点是否被球包含（边界算包含）。"""
        q = _validate_coordinate_sequence(point, where="contains() argument")
        return _dist2(q, self.center) <= self.radius_squared

    def normal_distance_squared(self, point) -> Fraction:
        """点到约束子空间的法向距离平方 h（精确）。"""
        q = _validate_coordinate_sequence(
            point, where="normal_distance_squared() argument"
        )
        _, h = _reduce_point(
            q, self.subspace_origin, self.subspace_directions, _gram_matrix(
                self.subspace_directions
            )
        )
        return h

    def certificate_residual(self) -> tuple[Fraction, ...]:
        """低维梯度平衡残差 B^T * sum_i lambda_i (c - p_i)。

        最优证书下恰为长度 subspace_dimension 的零元组：活动点梯度沿
        约束子空间方向的凸组合为零（法向分量被子空间约束吸收）。
        """
        residual3 = tuple(
            sum(
                (s.weight * (self.center[d] - s.point[d]) for s in self.support),
                Fraction(0),
            )
            for d in range(3)
        )
        return tuple(
            sum(residual3[d] * v[d] for d in range(3))
            for v in self.subspace_directions
        )

    def dual_identity_radius_squared(self) -> Fraction:
        """对偶恒等式右侧（低维坐标，含各点法向距离常数项 h_i）。

        等于 sum_i lambda_i (|x_i|_G^2 + h_i) - |sum_i lambda_i x_i|_G^2，
        最优时与 radius_squared 精确相等。
        """
        gram = _gram_matrix(self.subspace_directions)
        reduced = [
            _reduce_point(s.point, self.subspace_origin,
                          self.subspace_directions, gram)
            for s in self.support
        ]
        weighted_norm = sum(
            (s.weight * (_metric_norm2(x, gram) + h)
             for s, (x, h) in zip(self.support, reduced)),
            Fraction(0),
        )
        dim = self.subspace_dimension
        wx = tuple(
            sum((s.weight * x[j] for s, (x, _h) in zip(self.support, reduced)),
                Fraction(0))
            for j in range(dim)
        )
        return weighted_norm - _metric_norm2(wx, gram)


def _gram_matrix(directions):
    """方向向量的 Gram 矩阵 G = B^T B（精确）。"""
    return [
        [sum(u[d] * v[d] for d in range(3)) for v in directions]
        for u in directions
    ]


def _metric_norm2(x, gram):
    """x^T G x：低维坐标在约束子空间度量下的长度平方。"""
    return sum(
        x[a] * gram[a][b] * x[b]
        for a in range(len(x)) for b in range(len(x))
    )


def _metric_dot(x, y, gram):
    return sum(
        x[a] * gram[a][b] * y[b]
        for a in range(len(x)) for b in range(len(y))
    )


def _reduce_point(p, origin, directions, gram):
    """把三维点分解为低维坐标 x 与法向距离平方 h。

    p - origin = B x + w，B x 在子空间内、w 为法向分量；
    x = G^{-1} B^T (p - origin)，h = |w|^2。全部精确 Fraction。
    """
    delta = tuple(p[d] - origin[d] for d in range(3))
    rhs = [sum(v[d] * delta[d] for d in range(3)) for v in directions]
    x = _solve_linear(gram, rhs)
    if x is None:  # 方向已做秩检查，数学上不可达
        raise MiniballError("internal: subspace Gram matrix is singular")
    x = tuple(x)
    h = _norm2(delta) - _metric_norm2(x, gram)
    return x, h


def _normalize_subspace(origin, directions):
    """校验并规范化约束子空间；返回 (origin, directions, gram)。

    方向个数必须为 1（直线）或 2（平面），且先做精确秩检查：
    零方向、重复或线性相关的方向一律拒绝（Gram 行列式为 0）。
    """
    o = _validate_coordinate_sequence(origin, where="subspace_origin")
    if isinstance(directions, (str, bytes)) or not hasattr(directions, "__len__"):
        raise TypeError("subspace_directions: a sequence of 1 or 2 direction vectors is required")
    dim = len(directions)
    if dim not in (1, 2):
        raise ValueError(
            f"subspace_directions: exactly 1 (line) or 2 (plane) direction "
            f"vectors are required, got {dim}"
        )
    vecs = tuple(
        _validate_coordinate_sequence(v, where=f"subspace_directions[{i}]")
        for i, v in enumerate(directions)
    )
    gram = _gram_matrix(vecs)
    det = gram[0][0] if dim == 1 else gram[0][0] * gram[1][1] - gram[0][1] ** 2
    if det == 0:
        raise ValueError(
            "subspace_directions: direction vectors must be linearly "
            "independent (zero, duplicate or dependent directions are "
            "rejected; exact rank check failed)"
        )
    return o, vecs, gram


def _constrained_candidate(reduced, combo, gram):
    """低维带法向常数项的等值候选：对活动集求使 |x-x_i|_G^2 + h_i 相等
    且 x 落在活动集仿射包内的中心；返回 (x, t, weights) 或 None。"""
    k = len(combo)
    pts = [reduced[i] for i in combo]
    x0, h0 = pts[0]
    if k == 1:
        return x0, h0, (Fraction(1),)

    vectors = [tuple(pts[j][0][d] - x0[d] for d in range(len(x0)))
               for j in range(1, k)]
    size = k - 1
    gram_sys = [
        [_metric_dot(vectors[a], vectors[b], gram) for b in range(size)]
        for a in range(size)
    ]
    rhs = [
        (_metric_norm2(vectors[a], gram) + pts[a + 1][1] - h0) / 2
        for a in range(size)
    ]
    alpha = _solve_linear(gram_sys, rhs)
    if alpha is None:
        return None  # 活动集投影仿射相关：数值退化，由调用方计数标注

    center = tuple(
        x0[d] + sum((alpha[j] * vectors[j][d] for j in range(size)), Fraction(0))
        for d in range(len(x0))
    )
    weights = (Fraction(1) - sum(alpha, Fraction(0)), *alpha)
    t = _metric_norm2(tuple(center[d] - x0[d] for d in range(len(x0))), gram) + h0
    return center, t, weights


def minimum_enclosing_sphere_constrained(
    points, *, subspace_origin, subspace_directions, ids=None, budget=None
) -> ConstrainedMinimumEnclosingSphere:
    """球心限定在给定直线/平面上的最小包围球（精确，带最优证书）。

    点仍全部要覆盖；球心限定为 subspace_origin + span(subspace_directions)。
    方法：转到低维坐标后求带各点法向距离常数项 h_i 的最小最大二次距离
    min_c max_i (|c - x_i|_G^2 + h_i)，不是先求自由球心再投影（投影一般
    不再最优）。

    Args:
        points: 同 minimum_enclosing_sphere（1..30 个有理三维点）。
        subspace_origin: 子空间上一点，3 坐标。
        subspace_directions: 1 个（直线）或 2 个（平面）方向向量；
            先做精确秩检查，零/重复/线性相关方向抛 ValueError。
        ids / budget: 语义同 minimum_enclosing_sphere。

    Returns:
        ConstrainedMinimumEnclosingSphere：精确中心、r^2、活动点
        （SupportPoint，含原始 ID 来源与非负重心系数）与最优性证书。
        degenerate_combinations > 0 时表明枚举中遇到投影仿射相关的
        退化活动集（已跳过并明确计数）。

    Raises:
        TypeError / ValueError: 输入或子空间参数非法（含方向相关）。
        SearchBudgetExhaustedError: budget 耗尽，结果未知。
        NoFeasibleCandidateError: 穷举证明无可行候选（非空输入下不发生）。
    """
    normalized = _normalize_points(points)
    n = len(normalized)
    original_ids = _normalize_ids(ids, n)
    budget_value = _normalize_budget(budget)
    origin, directions, gram = _normalize_subspace(
        subspace_origin, subspace_directions
    )
    dim = len(directions)

    # 转到低维坐标；按 (x, h) 去重但保留全部原始 ID 来源。
    provenance: dict[tuple, list] = {}
    reduced: list[tuple[tuple[Fraction, ...], Fraction]] = []
    originals: dict[tuple, tuple] = {}
    for pid, p in zip(original_ids, normalized):
        key = _reduce_point(p, origin, directions, gram)
        if key not in provenance:
            provenance[key] = []
            reduced.append(key)
            originals[key] = p
        provenance[key].append(pid)

    m = len(reduced)
    best = None  # (t, k, combo, x, weights)
    degenerate = 0
    examined = 0
    for k in range(1, min(dim + 1, m) + 1):
        for combo in combinations(range(m), k):
            examined += 1
            if budget_value is not None and examined > budget_value:
                raise SearchBudgetExhaustedError(
                    f"search budget of {budget_value} candidate(s) exhausted "
                    f"before enumeration completed; result is unknown "
                    f"(examined {examined - 1}, unique reduced points {m})"
                )
            found = _constrained_candidate(reduced, combo, gram)
            if found is None:
                degenerate += 1
                continue
            x, t, weights = found
            if any(w < 0 for w in weights):
                continue  # 中心不在活动集凸包内，不构成最优证书
            if any(
                _metric_norm2(
                    tuple(x[d] - xi[d] for d in range(dim)), gram
                ) + hi > t
                for xi, hi in reduced
            ):
                continue  # 不能包围全部点
            key = (t, k, combo)
            if best is None or _less(key, best):
                best = (t, k, combo, x, weights)

    if best is None:
        raise NoFeasibleCandidateError(
            "exhaustive enumeration proved no feasible constrained enclosing "
            "sphere exists"
        )

    t, _k, combo, x, weights = best
    center = tuple(
        origin[d] + sum(x[j] * directions[j][d] for j in range(dim))
        for d in range(3)
    )
    support = tuple(
        SupportPoint(
            point=originals[reduced[idx]],
            ids=tuple(provenance[reduced[idx]]),
            weight=weights[pos],
        )
        for pos, idx in enumerate(combo)
    )
    return ConstrainedMinimumEnclosingSphere(
        center=center,
        radius_squared=t,
        support=support,
        num_input_points=n,
        num_unique_points=m,
        subspace_origin=origin,
        subspace_directions=directions,
        subspace_dimension=dim,
        degenerate_combinations=degenerate,
    )
