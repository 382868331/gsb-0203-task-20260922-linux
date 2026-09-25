"""miniball3d — 三维有理点集最小包围球的精确求解与可验证最优证书。

仅使用 Python 标准库（fractions / itertools / dataclasses / math）。

公开接口
========
    minimum_enclosing_sphere(points, *, ids=None, budget=None) -> MinimumEnclosingSphere
    trimmed_ball(points, k, *, ids=None, budget=None) -> TrimmedBallResult

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
6. trimmed_ball 枚举同一组候选支撑球，精确统计球外观测；球外数不超过
   k 时就是一个可行的剩余集合最小球。这等价于枚举所有非空排除子集后
   独立求球，重复坐标仍按不同观测处理。

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
    "MAX_TRIMMED_EXCLUSIONS",
    "SupportPoint",
    "MinimumEnclosingSphere",
    "TrimmedBallResult",
    "MiniballError",
    "NoFeasibleCandidateError",
    "SearchBudgetExhaustedError",
    "minimum_enclosing_sphere",
    "trimmed_ball",
]

#: 接口允许的输入点数量上限（题面规定 30）。
MIN_BALL_MAX_POINTS = 30

#: trimmed_ball 允许排除的观测点数量上限。
MAX_TRIMMED_EXCLUSIONS = 3


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


@dataclass(frozen=True)
class TrimmedBallResult:
    """允许排除至多 k 个观测点后的全局搜索结果。

    当 ``complete`` 为 False 时，``ball`` 只是当前已找到的可行球（可能为
    None），绝不能解释成全局最优；``excluded_ids`` 也只对应当前球。
    """

    ball: MinimumEnclosingSphere | None
    excluded_ids: tuple
    excluded_indices: tuple[int, ...]
    k: int
    complete: bool
    examined_candidates: int
    total_candidates: int
    budget: int | None

    @property
    def center(self):
        if self.ball is None:
            raise ValueError("search is incomplete and no feasible ball is available")
        return self.ball.center

    @property
    def radius_squared(self) -> Fraction:
        if self.ball is None:
            raise ValueError("search is incomplete and no feasible ball is available")
        return self.ball.radius_squared

    @property
    def support(self) -> tuple[SupportPoint, ...]:
        if self.ball is None:
            raise ValueError("search is incomplete and no feasible ball is available")
        return self.ball.support

    @property
    def weights(self) -> tuple[Fraction, ...]:
        if self.ball is None:
            raise ValueError("search is incomplete and no feasible ball is available")
        return self.ball.weights


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


def _normalize_trim_k(k) -> int:
    _reject_bool(k, "k")
    if not isinstance(k, int):
        raise TypeError("k: integer from 0 to 3 expected")
    if not 0 <= k <= MAX_TRIMMED_EXCLUSIONS:
        raise ValueError(
            f"k: must be between 0 and {MAX_TRIMMED_EXCLUSIONS}, got {k}"
        )
    return k


def _candidate_count(m: int) -> int:
    return sum(
        math.comb(m, size)
        for size in range(1, min(4, m) + 1)
    )


def _prepare_inputs(points, ids):
    normalized = _normalize_points(points)
    n = len(normalized)
    original_ids = _normalize_ids(ids, n)

    provenance: dict[tuple, list] = {}
    unique_points: list[tuple[Fraction, Fraction, Fraction]] = []
    for pid, p in zip(original_ids, normalized):
        if p not in provenance:
            provenance[p] = []
            unique_points.append(p)
        provenance[p].append(pid)
    return normalized, original_ids, provenance, unique_points


def _make_result(unique_points, provenance, combo, center, r2, weights):
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
        num_input_points=sum(len(v) for v in provenance.values()),
        num_unique_points=len(provenance),
    )


def _enumerate_miniball(unique_points, provenance, budget_value):
    """枚举唯一坐标的候选球。返回 (best, examined, total)。"""
    m = len(unique_points)
    total = _candidate_count(m)
    best = None
    examined = 0
    for support_size in range(1, min(4, m) + 1):
        for combo in combinations(range(m), support_size):
            if budget_value is not None and examined >= budget_value:
                return best, examined, total
            examined += 1
            found = _candidate(unique_points, combo)
            if found is None:
                continue
            center, r2, weights = found
            if any(weight < 0 for weight in weights):
                continue
            if any(_dist2(q, center) > r2 for q in unique_points):
                continue
            key = (r2, support_size, combo)
            if best is None or key < _best_key(best):
                best = (center, r2, weights, combo, key)
    return best, examined, total


def _enumerate_trimmed(normalized, original_ids, unique_points, k, budget_value):
    """枚举候选支撑球，并把球外的每个观测 ID 作为必排除点。

    对任一固定剩余集合，其最小包围球必有 1..4 个支撑点；因此枚举所有
    支撑球、精确统计球外观测，等价于枚举全部非空排除子集后独立求球。
    """
    m = len(unique_points)
    total = _candidate_count(m)
    best = None
    examined = 0
    for support_size in range(1, min(4, m) + 1):
        for combo in combinations(range(m), support_size):
            if budget_value is not None and examined >= budget_value:
                return best, examined, total
            examined += 1
            found = _candidate(unique_points, combo)
            if found is None:
                continue
            center, r2, weights = found
            if any(weight < 0 for weight in weights):
                continue
            outside_indices = tuple(
                index
                for index, point in enumerate(normalized)
                if _dist2(point, center) > r2
            )
            outside_ids = tuple(original_ids[index] for index in outside_indices)
            if len(outside_indices) > k:
                continue
            key = (
                r2,
                len(outside_indices),
                outside_ids,
                outside_indices,
                support_size,
                combo,
            )
            candidate = (
                center, r2, weights, combo, outside_indices, outside_ids, key
            )
            if best is None or _trimmed_candidate_preferred(candidate, best):
                best = candidate
    return best, examined, total


def _trimmed_candidate_preferred(candidate, best):
    if candidate[1] != best[1]:
        return candidate[1] < best[1]
    if len(candidate[4]) != len(best[4]):
        return len(candidate[4]) < len(best[4])
    if _id_tuple_less(candidate[5], best[5]):
        return True
    if _id_tuple_less(best[5], candidate[5]):
        return False
    return candidate[6][3:] < best[6][3:]


def _id_tuple_less(left, right):
    for left_value, right_value in zip(left, right):
        if left_value == right_value:
            continue
        try:
            return left_value < right_value
        except TypeError:
            return False
    return len(left) < len(right)


def _best_key(best):
    return best[-1]


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
    normalized, original_ids, provenance, unique_points = _prepare_inputs(points, ids)
    budget_value = _normalize_budget(budget)
    best, examined, total = _enumerate_miniball(
        unique_points, provenance, budget_value
    )
    if examined < total and budget_value is not None:
        raise SearchBudgetExhaustedError(
            f"search budget of {budget_value} candidate(s) exhausted "
            f"before enumeration completed; result is unknown "
            f"(examined {examined}, of {total}, unique points {len(unique_points)})"
        )
    if best is None:
        # 数学上不可达：单点子集 k=1 对任意非空点集至少有一个可行候选
        # （全部点重合时即为该点）。
        raise NoFeasibleCandidateError(
            "exhaustive enumeration proved no feasible enclosing sphere exists"
        )

    center, _r2, weights, combo, _key = best
    return _make_result(unique_points, provenance, combo, center, _r2, weights)


def trimmed_ball(points, k, *, ids=None, budget=None) -> TrimmedBallResult:
    """求允许排除至多 k 个观测点的最优包围球。

    目标字典序为：半径平方、实际排除数量、被排除原始 ID 序列。若不同
    ID 类型之间不可比较，则用输入位置序列兜底；最后再用支撑规模和坐
    标下标保证同一排除集合上的输出确定性。

    budget 耗尽时不抛异常；返回 complete=False 和当前已证明可行的最佳
    球。没有找到可行球时 ball/excluded_ids 为空值。
    """
    normalized, original_ids, provenance, unique_points = _prepare_inputs(points, ids)
    k_value = _normalize_trim_k(k)
    budget_value = _normalize_budget(budget)

    best, examined, total = _enumerate_trimmed(
        normalized, original_ids, unique_points, k_value, budget_value
    )
    complete = examined >= total
    if best is None:
        return TrimmedBallResult(
            ball=None,
            excluded_ids=(),
            excluded_indices=(),
            k=k_value,
            complete=complete,
            examined_candidates=examined,
            total_candidates=total,
            budget=budget_value,
        )

    center, r2, weights, combo, outside_indices, _outside_ids, _key = best
    excluded_set = set(outside_indices)
    retained_provenance = {}
    for idx, (point, pid) in enumerate(zip(normalized, original_ids)):
        if idx in excluded_set:
            continue
        retained_provenance.setdefault(point, []).append(pid)

    ball = _make_result(
        unique_points,
        retained_provenance,
        combo,
        center,
        r2,
        weights,
    )
    return TrimmedBallResult(
        ball=ball,
        excluded_ids=tuple(original_ids[idx] for idx in outside_indices),
        excluded_indices=outside_indices,
        k=k_value,
        complete=complete,
        examined_candidates=examined,
        total_candidates=total,
        budget=budget_value,
    )
