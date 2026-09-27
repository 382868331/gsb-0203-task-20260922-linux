"""dynamic_miniball — 可持续插入/删除点的最小包围球对象。

在 miniball3d 的精确一次性求解之上提供动态封装：

- `DynamicMiniball` 是不可变（持久化）对象：`insert` / `delete` 返回新
  版本，旧版本（及其已取出的球与证书）不受后续修改影响。
- 复用旧结果的正确性依据（都是最小包围球的标准充要条件）：
  * 新增点落在旧球内（含边界）时，旧球仍包含整个新点集，且其支撑证书
    不变，故旧球就是新点集的最优解 —— 不做无谓重算；
  * 删除点后全部支撑坐标仍存在于当前点集时，旧证书在新点集上依然成立
    （支撑等距、重心非负、和为 1、全局包含），故旧球仍最优；
  * 仅当某个支撑坐标被彻底删除（决定球边界的点消失）时，旧球可能不再
    最优，此时对当前点集从头精确重算，绝不保留旧球冒充最优。
- 复用时仅更新计数与支撑点的原始 ID 来源，几何量（中心、r^2、权重）
  原样保留，证书仍可独立核验。
- 失败的更新（非法坐标、重复 ID、超上限、删除不存在的 ID）在构造新
  对象之前抛出异常，当前点集不发生任何变化。
"""

from __future__ import annotations

from miniball3d import (
    MIN_BALL_MAX_POINTS,
    MinimumEnclosingSphere,
    SupportPoint,
    _dist2,
    _normalize_ids,
    _normalize_points,
    _validate_coordinate_sequence,
    minimum_enclosing_sphere,
)

__all__ = ["DynamicMiniball"]


class DynamicMiniball:
    """可增删点的三维点集，缓存并按需复用最小包围球结果。

    对象不可变：所有修改方法返回新的 `DynamicMiniball`，旧版本继续可用。
    """

    __slots__ = ("_entries", "_sphere")

    def __init__(self, points=(), *, ids=None):
        """从初始点集构造（默认空集）。

        校验失败时抛 TypeError/ValueError，不产生任何对象状态。
        """
        normalized = _normalize_points(points) if len(points) else ()
        if ids is None:
            id_tuple = tuple(range(len(normalized)))
        else:
            id_tuple = _normalize_ids(ids, len(normalized))
        self._entries = tuple(zip(id_tuple, normalized))
        self._sphere = self._compute_from(self._entries) if self._entries else None

    # ------------------------------------------------------------------
    # 只读视图
    # ------------------------------------------------------------------
    @property
    def ids(self) -> tuple:
        """当前点集的 ID（插入顺序）。"""
        return tuple(e[0] for e in self._entries)

    @property
    def points(self) -> tuple:
        """当前点集的精确坐标（与 ids 同序）。"""
        return tuple(e[1] for e in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, point_id) -> bool:
        return any(e[0] == point_id for e in self._entries)

    def sphere(self) -> MinimumEnclosingSphere:
        """当前点集的最小包围球（含可核验证书）。空集抛 ValueError。"""
        if self._sphere is None:
            raise ValueError("sphere(): the point set is empty")
        return self._sphere

    # ------------------------------------------------------------------
    # 修改（返回新版本，旧版本不受影响）
    # ------------------------------------------------------------------
    def insert(self, point, *, point_id=None) -> "DynamicMiniball":
        """插入一个点，返回新版本。

        新增点落在旧球内时复用旧球（仅更新计数与 ID 来源），否则重算。
        任何校验失败都在修改前抛出，当前对象不变。
        """
        coord = _validate_coordinate_sequence(point, where="insert(point)")
        if point_id is None:
            point_id = self._fresh_id()
        else:
            if isinstance(point_id, bool):
                raise TypeError(f"point_id: bool is not accepted (got {point_id!r})")
            try:
                taken = point_id in self
            except TypeError as exc:
                raise TypeError(
                    f"point_id: id must be hashable, got {type(point_id).__name__}"
                ) from exc
            if taken:
                raise ValueError(f"point_id: duplicate id {point_id!r}")
        if len(self._entries) >= MIN_BALL_MAX_POINTS:
            raise ValueError(
                f"insert: at most {MIN_BALL_MAX_POINTS} points are allowed"
            )

        entries = self._entries + ((point_id, coord),)
        sphere = self._sphere_after_insert(entries, coord)
        return self._spawn(entries, sphere)

    def delete(self, point_id) -> "DynamicMiniball":
        """按 ID 删除一个点，返回新版本。

        被删坐标不是支撑坐标（或仍有其他输入点占据该坐标）时复用旧球；
        支撑坐标彻底消失时从头重算。ID 不存在抛 KeyError，当前对象不变。
        """
        index = next(
            (i for i, e in enumerate(self._entries) if e[0] == point_id), None
        )
        if index is None:
            raise KeyError(f"delete: no point with id {point_id!r}")

        removed_coord = self._entries[index][1]
        entries = self._entries[:index] + self._entries[index + 1 :]
        sphere = self._sphere_after_delete(entries, removed_coord)
        return self._spawn(entries, sphere)

    # ------------------------------------------------------------------
    # 内部：复用 / 重算
    # ------------------------------------------------------------------
    @classmethod
    def _spawn(cls, entries, sphere) -> "DynamicMiniball":
        obj = cls.__new__(cls)
        obj._entries = entries
        obj._sphere = sphere
        return obj

    def _fresh_id(self):
        """缺省 ID：取未占用的最小非负整数，保证可哈希且不冲突。"""
        used = {e[0] for e in self._entries}
        candidate = 0
        while candidate in used:
            candidate += 1
        return candidate

    @staticmethod
    def _compute_from(entries) -> MinimumEnclosingSphere:
        ids = [e[0] for e in entries]
        pts = [e[1] for e in entries]
        return minimum_enclosing_sphere(pts, ids=ids)

    def _sphere_after_insert(self, entries, coord):
        old = self._sphere
        if old is None:
            return self._compute_from(entries)
        if _dist2(coord, old.center) > old.radius_squared:
            return self._compute_from(entries)  # 越出旧球：必须重算
        # 旧球仍包含全部点且证书不变 -> 仍最优，仅更新计数与 ID 来源。
        return self._refresh(old, entries)

    def _sphere_after_delete(self, entries, removed_coord):
        if not entries:
            return None
        old = self._sphere
        remaining_coords = {e[1] for e in entries}
        support_coords = {s.point for s in old.support}
        if removed_coord in support_coords and removed_coord not in remaining_coords:
            # 决定球边界的支撑坐标被彻底删除：旧球可能不再最优，必须重算。
            return self._compute_from(entries)
        # 旧证书在当前点集上依然成立 -> 复用，仅更新计数与 ID 来源。
        return self._refresh(old, entries)

    @staticmethod
    def _refresh(old: MinimumEnclosingSphere, entries) -> MinimumEnclosingSphere:
        """几何不变地复用旧球，重建计数与支撑点的原始 ID 来源。"""
        provenance: dict[tuple, list] = {}
        unique = 0
        for pid, coord in entries:
            if coord not in provenance:
                provenance[coord] = []
                unique += 1
            provenance[coord].append(pid)
        support = tuple(
            SupportPoint(point=s.point, ids=tuple(provenance[s.point]), weight=s.weight)
            for s in old.support
        )
        return MinimumEnclosingSphere(
            center=old.center,
            radius_squared=old.radius_squared,
            support=support,
            num_input_points=len(entries),
            num_unique_points=unique,
        )
