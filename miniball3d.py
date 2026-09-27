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
    "EmptyPointSetError",
    "DynamicMiniball",
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


class EmptyPointSetError(MiniballError, ValueError):
    """对空点集查询最小包围球。与输入校验错误同属 ValueError 家族。"""


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
# 可持续插入 / 删除的增量对象
# ---------------------------------------------------------------------------

#: sphere() 结果来源的说明字符串（derivation 属性取值）。
_DERIVE_SCRATCH = "recomputed"          # 全量重算
_DERIVE_INSERT_INSIDE = "insert-contained"   # 新点落在旧球内，复用旧证书
_DERIVE_DELETE_FREE = "delete-non-support"   # 删除的不是支撑坐标，复用旧证书


class DynamicMiniball:
    """支持持续插入 / 删除点的最小包围球对象（持久化、不可变版本链）。

    每个实例代表一个确定版本的点集；insert / delete 返回 *新* 版本，
    旧版本及其查询结果不受后续修改影响（版本间共享不可变数据，绝不
    原地改写）。任何一步更新失败（非法坐标、重复 / 缺失 ID、超上限）
    都在产生新版本之前抛出，当前点集保持不变。

    结果复用规则（复用后仍给出当前点集的精确正确球与完整证书）：

    * 插入点 q：若 q 被旧球精确包含，则旧球对新点集仍最优
      （MEB(S) 是 S∪{q} 的可行球，且任何包围 S∪{q} 的球也包围 S，
      不可能更小），直接复用旧证书，仅更新计数与 ID 来源。
    * 删除点 p：若 p 的坐标不是旧证书的支撑坐标（或该坐标仍有其他
      重复 ID 留存），旧证书对剩余点集依然成立（支撑仍在、其余点
      仍被包含），直接复用；否则全量重算——绝不把旧球冒充最优。
    * 复用前会对当前点集做一次完整精确核验（支撑坐标仍在、所有点
      被包含、权重和非负且为 1），核验不过则自动回退全量重算。
    """

    __slots__ = ("_entries", "_parent", "_delta", "_sphere", "_derivation")

    def __init__(self, _entries=None, _parent=None, _delta=None):
        # _entries: 不可再变的 dict(id -> 精确坐标三元组)；只在构造时写入。
        self._entries = {} if _entries is None else _entries
        self._parent = _parent
        self._delta = _delta  # ("insert", id, point) 或 ("delete", id, point)
        self._sphere = None
        self._derivation = None

    # -- 只读视图 --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, pid) -> bool:
        return pid in self._entries

    @property
    def ids(self) -> tuple:
        """当前版本的全部点 ID（插入顺序）。"""
        return tuple(self._entries)

    @property
    def derivation(self) -> str | None:
        """最近一次 sphere() 结果的来源；尚未计算时为 None。"""
        return self._derivation

    # -- 更新（返回新版本，失败不留下部分变更） --------------------------

    def insert(self, point, *, pid=None) -> "DynamicMiniball":
        """返回插入 point 后的新版本；当前版本不变。

        pid 缺省时自动取未被占用的最小非负整数。坐标非法、pid 重复
        或超出 30 点上限时抛错，且当前点集不受任何影响。
        """
        coord = _validate_coordinate_sequence(point, where="insert() point")
        if pid is None:
            pid = 0
            while pid in self._entries:
                pid += 1
        else:
            try:
                already = pid in self._entries
            except TypeError as exc:
                raise TypeError(
                    f"insert() pid: id must be hashable, got {type(pid).__name__}"
                ) from exc
            if already:
                raise ValueError(f"insert() pid: duplicate id {pid!r}")
        if len(self._entries) >= MIN_BALL_MAX_POINTS:
            raise ValueError(
                f"insert(): at most {MIN_BALL_MAX_POINTS} points are allowed"
            )
        entries = dict(self._entries)
        entries[pid] = coord
        return DynamicMiniball(entries, self, ("insert", pid, coord))

    def delete(self, pid) -> "DynamicMiniball":
        """返回删除指定 ID 后的新版本；当前版本不变。"""
        try:
            coord = self._entries[pid]
        except TypeError as exc:
            raise TypeError(
                f"delete() pid: id must be hashable, got {type(pid).__name__}"
            ) from exc
        except KeyError:
            raise KeyError(f"delete() pid: no such id {pid!r}") from None
        entries = dict(self._entries)
        del entries[pid]
        return DynamicMiniball(entries, self, ("delete", pid, coord))

    # -- 查询 ------------------------------------------------------------

    def sphere(self) -> MinimumEnclosingSphere:
        """当前点集的最小包围球与最优证书（精确，结果缓存于本版本）。"""
        if self._sphere is None:
            self._sphere, self._derivation = self._compute()
        return self._sphere

    def _compute(self):
        if not self._entries:
            raise EmptyPointSetError(
                "sphere(): cannot compute a minimum enclosing sphere "
                "of an empty point set"
            )
        parent_sphere = None
        if self._parent is not None:
            try:
                parent_sphere = self._parent.sphere()
            except EmptyPointSetError:
                parent_sphere = None
        if parent_sphere is not None:
            reused = self._try_reuse(parent_sphere)
            if reused is not None:
                return reused
        ids = tuple(self._entries)
        result = minimum_enclosing_sphere(
            [self._entries[i] for i in ids], ids=ids
        )
        return result, _DERIVE_SCRATCH

    def _try_reuse(self, parent_sphere):
        kind, _pid, coord = self._delta
        if kind == "insert":
            # 新点在旧球内 => 旧球对新点集仍最优（见类 docstring）。
            if not parent_sphere.contains(coord):
                return None
            derivation = _DERIVE_INSERT_INSIDE
        else:
            # 删除：只有当被删坐标不再是必要支撑时旧证书才仍然有效。
            support_coords = {s.point for s in parent_sphere.support}
            still_present = coord in self._entries.values()
            if coord in support_coords and not still_present:
                return None  # 删掉了决定边界的支撑点，必须重算
            derivation = _DERIVE_DELETE_FREE
        rebuilt = self._rebuild_certificate(parent_sphere)
        if rebuilt is None:
            return None
        return rebuilt, derivation

    def _rebuild_certificate(self, parent_sphere):
        """用旧球的中心 / r^2 / 权重对当前点集重建并完整核验证书。"""
        provenance: dict[tuple, list] = {}
        for pid, p in self._entries.items():
            provenance.setdefault(p, []).append(pid)
        center = parent_sphere.center
        r2 = parent_sphere.radius_squared
        support = []
        for s in parent_sphere.support:
            ids = provenance.get(s.point)
            if ids is None:
                return None  # 支撑坐标已不在当前点集中
            support.append(
                SupportPoint(point=s.point, ids=tuple(ids), weight=s.weight)
            )
        if any(w < 0 for w in (s.weight for s in support)):
            return None
        if sum((s.weight for s in support), Fraction(0)) != 1:
            return None
        # 当前点集逐点精确包含核验（复用的最后一道防线）。
        for p in provenance:
            if _dist2(p, center) > r2:
                return None
        return MinimumEnclosingSphere(
            center=center,
            radius_squared=r2,
            support=tuple(support),
            num_input_points=len(self._entries),
            num_unique_points=len(provenance),
        )
