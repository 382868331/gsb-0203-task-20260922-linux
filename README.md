# miniball3d — 三维有理点集最小包围球与最优证书（含 trimmed_ball）

纯 Python 标准库实现：对最多 30 个有理三维点，用 **精确 `Fraction` 算术**
求最小包围球（minimum enclosing ball），并输出可独立核验的最优性证书
（支撑点 + 非负重心系数）。另提供 `trimmed_ball(points, k)`：允许排除
最多 k 个（k ∈ 0..3）输入点后的最小包围球，仍是精确全局枚举，不是
启发式离群点猜测，也不是统计置信球。

- **Python 版本**：Python 3.14.7（CPython，Linux 原生验证）
- **依赖**：无，仅标准库（`fractions`、`itertools`、`dataclasses`、`math`）；
  安装 Python 后无需 `pip install` 任何东西（`requirements.txt` 为空注释）。
- **离线**：核心、测试、演示全部离线运行；不访问网络、账号、数据库、
  云 API、Docker 或任何硬件。示例数据由仓库内确定性代码生成。

## 运行命令

```bash
# 必要测试（44 个用例）
python -m unittest discover -s tests -v

# 固定输入演示（正常结果 + trimmed_ball + 真实触发的拒绝边界；约 0.1 秒）
python demo.py
```

以上命令在仓库根目录执行；题目环境解释器为 `./.venv/bin/python`
（Python 3.14.7），即 `./.venv/bin/python -m unittest discover -s tests -v`
与 `./.venv/bin/python demo.py`。全程离线、标准库、无凭据。

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

## trimmed_ball：允许排除最多 k 个点

```python
from miniball3d import trimmed_ball

result = trimmed_ball(points, k, *, ids=None, budget=None)
```

- `k`：允许排除的最大点数，整数 0..3（`bool` 拒绝，越界
  `ValueError`）。`k=0` 时 `result.ball` 与
  `minimum_enclosing_sphere(points, ids=ids)` 完全一致。
- 重复坐标的每个副本都是**独立观测**：排除子集按输入下标枚举，删一个
  副本不会删除该位置的其他点。
- 剩余集合不许为空（排除子集大小 ≤ n−1，自动收紧）。
- **目标排序**：按 `(半径平方, 排除数量, 被排除 ID 序列)` 字典序取
  最优——先半径最小，平局排除更少，再平局被排除 ID 序列（按输入顺序
  排列）字典序更小；ID 不可互相比较时退化为输入下标序，结果仍确定。
- **无启发式猜测**：枚举所有大小 ≤ min(k, n−1) 的排除子集，每个候选在
  保留点上用与原接口相同的精确候选支撑 + 重心证书求解；候选结果列出
  全部被排除的原始 ID。
- `budget`：可选正整数，限制评估的排除子集数。耗尽时**不抛异常**，
  返回 `complete=False` 的当前最优可行球（状态未完成，绝不冒充全局
  最优）。

返回 frozen dataclass `TrimmedBall`：

| 字段/属性 | 含义 |
|---|---|
| `ball` | 保留点集上的 `MinimumEnclosingSphere`（含完整证书） |
| `excluded_ids` | 被排除点的全部原始 ID（按输入顺序） |
| `complete` | 全部子集枚举完毕为 True；预算耗尽为 False |
| `candidates_examined` | 实际评估的排除子集数 |
| `center` / `radius_squared` | 委托给 `ball` 的精确量 |

### 可复现使用路径

```bash
# 1) 直接跑演示（约 0.1 秒，现场计算，含孤立点被排除的完整证书）
./.venv/bin/python demo.py
```

```python
# 2) 等价的最小可复现代码
from miniball3d import trimmed_ball

points = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0), (100, 0, 0)]
result = trimmed_ball(points, 1, ids=["c0", "c1", "c2", "c3", "isolated"])
assert result.complete
assert result.excluded_ids == ("isolated",)
assert result.radius_squared == 2          # 精确 Fraction
assert result.ball.certificate_residual() == (0, 0, 0)
```

### 失败示例（真实触发，不静默纠正）

```python
trimmed_ball(points, 4)
# ValueError: k: must be in 0..3, got 4

trimmed_ball(points, 1, budget=1)
# 不抛异常：返回 complete=False 的 TrimmedBall，
# ball 只是当前最优可行球（excluded_ids == ()），结果未知不冒充最优。
```

### trimmed_ball 的复杂度与定位

排除子集数为 Σ_{j=0..k} C(n, j)（n=30、k=3 时约 4.5 千），每个子集内部
仍是 O(m⁴) 候选的精确枚举，因此 k=3 且 n 接近 30 时总耗时显著高于
k=0；请用 `budget` 限制并以 `complete` 判断是否完成。这是“离线精确
全局枚举 + 可验证证书”的取舍，**不是**统计置信球，接口也不接受任何
带半径的输入球。

### 证书如何验证最优性

输出的 λ 满足且可由使用方独立核验：

1. λ_i ≥ 0 且 Σ λ_i = 1（凸组合）；
2. c = Σ λ_i p_i，即 `Σ λ_i (c − p_i) = 0`（平衡残差为 0）；
3. 每个支撑点到 c 的距离平方都等于 r²（支撑等距）；
4. 其余每个输入点满足 |q − c|² ≤ r²（全局包含）；
5. 对偶恒等式（对任意凸组合成立，非“往返”自检）：

   r² = Σ_i λ_i |p_i|² − |Σ_i λ_i p_i|² = Σ_{i<j} λ_i λ_j |p_i − p_j|²。

这些条件合在一起是最小包围球的标准充要条件（Welzl/凸分析证书）。

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
- `trimmed_ball` 的 k 限定 0..3；排除子集枚举叠加上述内层枚举，
  大 n 大 k 时慢，预算耗尽只回当前可行球 + `complete=False`。
- Fraction 分子分母随坐标位宽增长，输入极大整数时运算会变慢（无位宽
  上限之外的额外保护）。
- 浮点输入按其精确二进制值参与计算：例如 `0.1` 被当作
  `3602879701896397/2^55`，与十进制字面量 1/10 不同——这是有意的精确
  语义，不是误差。

## 测试与独立参考说明

`tests/test_miniball3d.py`（28 个用例）覆盖：单点/两点、共线、共面
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

`tests/test_trimmed_ball.py`（16 个用例）覆盖：k=0 与原接口逐字段一致、
孤立点被排除、对称平局按被排除 ID 序列裁决、重复离群位置（k=1 删不掉
两个副本、k=2 才能）、共线点、全部点相同、剩余集合不为空、预算耗尽
返回当前可行球 + `complete=False`、原始 ID 透传、结果不可变、
k/budget/点集的全部输入校验。

## 文件清单

- `miniball3d.py` — 本题核心库（可直接 import）。
- `tests/test_miniball3d.py` — 必要测试 + 独立参考实现。
- `tests/test_trimmed_ball.py` — trimmed_ball 必要测试。
- `demo.py` — 固定输入演示（立方体 8 顶点 + trimmed_ball 孤立点排除；
  真实计算 + 真实拒绝边界）。
- `README.md`、`requirements.txt`（无第三方依赖）。
