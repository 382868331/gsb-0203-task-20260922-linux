# miniball3d — 三维有理点集最小包围球与最优证书

纯 Python 标准库实现：对最多 30 个有理三维点，用 **精确 `Fraction` 算术**
求最小包围球（minimum enclosing ball），并输出可独立核验的最优性证书
（支撑点 + 非负重心系数）。

- **Python 版本**：Python 3.14.7（CPython，Linux 原生验证）
- **依赖**：无，仅标准库（`fractions`、`itertools`、`dataclasses`、`math`）；
  安装 Python 后无需 `pip install` 任何东西（`requirements.txt` 为空注释）。
- **离线**：核心、测试、演示全部离线运行；不访问网络、账号、数据库、
  云 API、Docker 或任何硬件。示例数据由仓库内确定性代码生成。

## 运行命令

```bash
# 必要测试（44 个用例）
python -m unittest discover -s tests -v

# 固定输入演示（正常结果 + 一个真实触发的拒绝边界；约 0.1 秒）
python demo.py
```

## 公开接口

```python
from miniball3d import minimum_enclosing_sphere

result = minimum_enclosing_sphere(points, *, ids=None, budget=None)
```

- `points`：1..30 个点的有限序列；每个点是长度恰为 3 的序列。
  坐标接受 `int`、`fractions.Rational`（含 `Fraction`）以及**有限**
  `float`（按其 IEEE-754 二进制值精确转为 `Fraction`，不做任何“取整”
  或随机抖动）。
- `ids`：可选，与 `points` 等长、元素可哈希且互不重复；缺省为
  `0 .. n-1`。坐标重复时，支撑点记录映射到该坐标的**全部原始 ID**。
- `budget`：可选正整数，最多检查的候选子集数。

返回不可变（frozen dataclass）`MinimumEnclosingSphere`：

| 字段/属性 | 含义 |
|---|---|
| `center` | 球心，`tuple[Fraction, Fraction, Fraction]`（精确） |
| `radius_squared` | 半径平方 `Fraction`（精确；不开根） |
| `support` | `SupportPoint` 元组：`point`（精确坐标）、`ids`（原始 ID 来源）、`weight`（非负重心系数 λ） |
| `weights` | 与 `support` 同序的 λ；`sum(weights) == 1` 精确成立 |
| `num_input_points` / `num_unique_points` | 输入点数 / 坐标去重后点数 |
| `center_float` / `radius_float` | 仅用于展示的 float64 近似（`radius_float = sqrt(radius_squared)`） |
| `contains(p)` | 精确包含判定（边界算包含） |
| `certificate_residual()` | `sum λ_i (c − p_i)`，证书下精确为 `(0,0,0)` |
| `dual_identity_radius_squared()` | 对偶恒等式右侧，证书下精确等于 `radius_squared` |

### 证书如何验证最优性

输出的 λ 满足且可由使用方独立核验：

1. λ_i ≥ 0 且 Σ λ_i = 1（凸组合）；
2. c = Σ λ_i p_i，即 `Σ λ_i (c − p_i) = 0`（平衡残差为 0）；
3. 每个支撑点到 c 的距离平方都等于 r²（支撑等距）；
4. 其余每个输入点满足 |q − c|² ≤ r²（全局包含）；
5. 对偶恒等式（对任意凸组合成立，非“往返”自检）：

   r² = Σ_i λ_i |p_i|² − |Σ_i λ_i p_i|² = Σ_{i<j} λ_i λ_j |p_i − p_j|²。

这些条件合在一起是最小包围球的标准充要条件（Welzl/凸分析证书）。

## 可持续插入 / 删除的增量对象

```python
from miniball3d import DynamicMiniball

dyn = DynamicMiniball()                 # 空点集版本
dyn = dyn.insert((1, 2, 3), pid="a")    # 返回新版本；旧版本不变
dyn = dyn.delete("a")                   # 同上
sphere = dyn.sphere()                   # 当前点集的精确 MEB + 完整证书
dyn.derivation                          # 本次结果来源（见下）
```

- **持久化**：每个实例是不可变版本；`insert` / `delete` 返回新版本，
  旧版本的点集与已算出的球不受后续修改影响。`insert` 的 `pid` 缺省时
  自动取未被占用的最小非负整数。
- **失败原子性**：非法坐标（NaN/Inf/bool/维度错误）、重复或不可哈希
  ID、删除不存在的 ID、超过 30 点上限，都在产生新版本之前抛
  `ValueError` / `TypeError` / `KeyError`，当前点集保持原样。
- **空集查询**：`sphere()` 抛 `EmptyPointSetError`（`MiniballError`
  与 `ValueError` 的子类）。

### 结果复用规则（`derivation` 属性）

- `"insert-contained"`：新点被旧球**精确**包含。此时旧球对新点集仍是
  最小包围球（MEB(S) 是 S∪{q} 的可行球，而任何包围 S∪{q} 的球也包围
  S，不可能更小），直接复用旧证书，只更新点数与 ID 来源，不做无谓重算。
- `"delete-non-support"`：被删坐标不是旧证书的支撑坐标（或该坐标仍有
  其他重复 ID 留存）。支撑仍在、其余点仍被包含，旧证书对剩余点集依然
  是最优性证书，直接复用。
- `"recomputed"`：删除了决定边界的支撑点（或父版本无可用结果）时
  **绝不保留旧球冒充最优**，对当前点集全量重算。
- 复用前会对当前点集做一次完整精确核验（支撑坐标仍在、所有点被包含、
  权重非负且和为 1），核验不过自动回退全量重算；复用结果与全量结果
  一样携带完整可核验证书（同一 `MinimumEnclosingSphere` 类型）。


## 算法与设计取舍

1. **去重保留来源**：按精确坐标去重（首次出现序），重复坐标的所有原始
   ID 挂到同一个 `SupportPoint.ids`；无随机抖动。
2. **候选枚举**：对去重后的 m ≤ 30 个点枚举 1..4 个点的全部子集
   （3 维中由等距方程确定唯一球的最大仿射独立规模为 4）。
3. **精确等距方程**：设 c = p0 + Σ αj (pj − p0)，等距条件
   |c−pi|² = |c−p0|² 化为 Gram 方程
   `(v_a·v_b) α = |v_a|²/2`，用 Fraction 高斯消元（选主元）求解；
   Gram 奇异 ⇔ 仿射相关，该候选跳过。
4. **可行性过滤**：仅保留 (a) 所有点被包含、(b) 重心系数全非负的候选；
   (b) 即“中心位于支撑凸包”，也是证书的平衡条件。
5. **选优与确定性**：取 r² 最小；平局时支撑数更少优先，再按下标字典序。
   结果与输入顺序完全确定。
6. **无浮点容差**：包含、非负、相等全部是 Fraction 精确比较，不用宽容差
   掩盖边界情形（钝角三角形自动退化为最长边直径、直角三角形同样退化、
   重合点退化为半径 0）。
7. **无状态、不修改输入**；主函数是纯函数。

### 枚举规模（输入上限 30 的依据）

候选数 Σ_{k=1..4} C(m,k) = C(31,4)+C(31,3)−1 ≈ 3.6 万；在本机
CPython 3.14.7 上 30 个随机整数点约 3–4 秒。该规模符合“离线一次性
精确求解 + 可验证证书”的定位；**不是**面向海量点的增量产品算法。

## 错误语义（不静默纠正）

- 坐标为 `NaN` / `+Inf` / `-Inf` → `ValueError`，信息含参数路径
  （如 `points[1][1]: NaN/Infinity is not allowed`）。
- `bool` 坐标或 `bool` 型整数参数（如 `budget=True`）→ `TypeError`
  （`bool` 虽是 `int` 子类，按契约显式拒绝）。
- 空点集、点数 > 30、坐标数 ≠ 3、坐标为字符串等 → 带定位的
  `ValueError` / `TypeError`。
- `ids` 长度不符 → `ValueError`；重复 ID → `ValueError`（如 `ids[1]`）；
  不可哈希 → `TypeError`。
- `budget` 非正整数 → `ValueError`/`TypeError`。
- **搜索预算耗尽与证明无解严格分开**：
  `SearchBudgetExhaustedError`（结果未知，绝不冒充最优或无解）
  ≠ `NoFeasibleCandidateError`（穷举证明不可行；非空输入下数学上不会
  发生，保留用于契约完整性）。
- 本模块没有多步批次/可变异状态，不存在部分变更问题；输入对象不会被
  修改。

## 浮点结果的误差口径

- **对外的几何结果只有精确量**：`center`、`radius_squared`、`weight`
  全部是 `Fraction`，判定与证书均精确，误差为 0。
- `center_float` 是逐坐标“一次精确舍入”的 IEEE-754 binary64
  （相对误差 ≤ 2⁻⁵³）；`radius_float = math.sqrt(float(radius_squared))`
  的相对误差同样约 ≤ 2⁻⁵³（一次 Fraction→float 舍入 + 一次正确舍入的
  `sqrt`）。它们仅供打印展示，**不参与**任何包含性或最优性判定，也不
  应用于复核证书；复核请用 Fraction 精确字段。

## 已知限制

- 仅实现本题契约的数学子集：三维（恰好 3 个坐标）、1..30 个有理点；
  不宣称是任意维度 / 任意规模 / 行业标准（如不处理二维 API、流式点、
  近似几何谓词）。
- 枚举复杂度 O(n⁴) 候选、每候选 O(n) Fraction 包含检查；n > 30 在接口
  层直接拒绝。
- Fraction 分子分母随坐标位宽增长，输入极大整数时运算会变慢（无位宽
  上限之外的额外保护）。
- 浮点输入按其精确二进制值参与计算：例如 `0.1` 被当作
  `3602879701896397/2^55`，与十进制字面量 1/10 不同——这是有意的精确
  语义，不是误差。

## 测试与独立参考说明

`tests/test_miniball3d.py`（30 个用例）覆盖：单点/两点、共线、共面
（正方形对角直径）、锐角三角形（3 支撑）、**钝角与直角三角形退化为
最长边直径**、正四面体（4 支撑等权）、重复点保留多 ID、全部重合、
30 点上限边界、Fraction 与有限 float 输入、全部输入校验与错误定位、
budget 耗尽/无解两类异常、不可变性。

文件内含一份**独立参考实现**：叉积 + 行列式 Cramer 法则，k=2/3/4 分别
走中点、平面正交基、R³ 直接线性化的不同代码路径；它不 import 也不调用
被测核心的任何求解函数。对固定点集及 60 组确定性随机点集（≤8 点，
即题面要求的“≤8 点枚举所有支撑作参考”规模）逐字段比对中心、r²、
支撑集与重心系数，并对每个结果做精确证书核验（含对偶恒等式）。
“往返一致”类检查仅作补充。

`tests/test_dynamic.py`（14 个用例）针对 `DynamicMiniball`：插入球内点 /
删除非支撑点的复用结果与全量批处理逐版本比对 r² 并做完整证书核验；
删除支撑点（两点直径球）必须重算且结果改变；支撑坐标有重复 ID 留存时
允许复用；40 步确定性混合插入 / 删除序列每步与全量结果比对；旧版本
持久化；失败插入 / 删除不改变点集；空集查询与 30 点上限。

## 文件清单

- `miniball3d.py` — 本题核心库（可直接 import）。
- `tests/test_miniball3d.py` — 必要测试 + 独立参考实现。
- `tests/test_dynamic.py` — 增量对象测试：复用路径与全量结果逐版本比对、
  支撑删除强制重算、持久化、失败更新原子性、空集与上限边界。
- `demo.py` — 固定输入演示（立方体 8 顶点；真实计算 + 真实拒绝边界 + 增量复用/重算演示）。
- `README.md`、`requirements.txt`（无第三方依赖）。
