# miniball3d — 最小包围球：三维点集精确证书 + 一维/二维输入球数值解

纯 Python 标准库实现，提供两个公开入口：

1. **`minimum_enclosing_sphere`**：对最多 30 个有理**三维点**，用精确
   `Fraction` 算术求最小包围球，并输出可独立核验的最优性证书
   （支撑点 + 非负重心系数）。
2. **`minimum_enclosing_ball`**（本次迭代新增）：对一维或二维、最多 6 个
   **输入球** `(球心, 半径)` 求覆盖全部输入球的最小包围球（float64 数值解、
   容差显式可配）；输入球半径为 0 时即点，退化为点集最小包围球。

- **Python 版本**：Python 3.14.7（CPython，Linux 原生验证）
- **依赖**：无，仅标准库（`fractions`、`itertools`、`dataclasses`、`math`）；
  安装 Python 后无需 `pip install` 任何东西（`requirements.txt` 为空注释）。
- **离线**：核心、测试、演示全部离线运行；不访问网络、账号、数据库、
  云 API、Docker 或任何硬件。示例数据由仓库内确定性代码生成。

## 运行命令

```bash
# 全部测试（68 个用例：原 30 个三维点集用例 + 38 个输入球用例）
python -m unittest discover -s tests -v

# 固定输入演示（正常结果 + 边界 + 真实触发的拒绝；0.1 秒内）
python demo.py
```

## 公开接口一：三维点集（精确）

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

## 公开接口二：一维/二维输入球（数值）

```python
from miniball3d import minimum_enclosing_ball

result = minimum_enclosing_ball(balls, *, dim=2, ids=None, tol=1e-10)
```

- `balls`：1..6 个输入球的有限序列；每个球是长度恰为 2 的配对
  `(center, radius)`。
  - `center`：长度恰为 `dim` 的坐标序列，接受 `int` / `Fraction` /
    **有限** `float`（统一转 float64；bool、NaN/Infinity 拒绝）。
  - `radius`：非负有限实数；**半径 0 表示点**。
  - 求的是覆盖全部输入球的最小球（含边界），即最小化 R 使
    `dist(c, p_i) + a_i ≤ R` 对每个输入球独立成立。
- `dim`：`1` 或 `2`（缺省 `2`）；坐标数必须与 `dim` 一致。
- `ids`：可选，等长、可哈希、互不重复；完全相同的球（球心与半径都相等）
  去重后合并其 ID 来源，与点接口语义一致。
- `tol`：显式数值容差（缺省 `1e-10`，常量 `MIN_BALL_TOL`）。逐球可行性
  判定为 `dist + radius ≤ R + tol·max(1, R, radius, |坐标|)`；
  `tol=0` 合法。容差只用于数值谓词，**不参与**候选选择的排序。

返回不可变 `MinimumEnclosingBall`：

| 字段/属性 | 含义 |
|---|---|
| `center` | 球心，`tuple[float, ...]`（长度 = dim） |
| `radius` | 半径 `float` |
| `support` | `SupportBall` 元组：`center`、`radius`、`ids`、`weight`（非负 KKT 权重） |
| `weights` | 与 `support` 同序的权重；和为 1 |
| `num_input_balls` | 输入球数（去重前） |
| `tol` | 本次求解实际使用的容差 |
| `balls` | 归一化后的输入球快照（按输入顺序），供独立复核 |
| `constraint_values` | **逐球**的 `dist(c, p_i) + a_i`，按输入顺序 |
| `slacks` | **逐球**的 `R − (dist + a_i)`；支撑球约为 0，其余为正 |
| `max_violation()` | `max(dist + a_i − R)`；正常结果 ≤ 容差口径 |
| `balance_residual()` | 支撑法向力平衡残差 `Σ λ_i (c−p_i)/|c−p_i|` |
| `contains_point(p)` / `contains_ball(center, radius)` | 点/球包含判定（边界算包含） |

### 输入球结果如何独立核验

每个输入球的约束都是**一个独立谓词**（`constraint_values`/`slacks`
逐球给出，不做混合汇总），使用方可直接复核：

1. 对每个输入球：`dist(c, p_i) + a_i ≤ R`；
2. 每个支撑球紧贴：`dist(c, p_i) + a_i = R`；
3. KKT 权重 λ_i ≥ 0、Σ λ_i = 1，且支撑处单位法向
   `n_i = (c − p_i)/|c − p_i|` 满足 `Σ λ_i n_i = 0`
   （`balance_residual()`，数值上约为 0）；
4. 目标函数 `f(c) = max_i (dist(c, p_i) + a_i)` 是凸函数，条件 2–3
   即 `0 ∈ ∂f(c)`，凸性保证这是全局最小的**充要**条件，不是往返自检。
   球心与某支撑球心重合时（如单个大球覆盖全部）该支撑法向无定义，
   退化为单支撑、权重 1，由 `R = a_max ≥ dist + a_i` 直接证明最优。

## 算法与设计取舍

### 三维点集（精确）

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

枚举规模：候选数 Σ_{k=1..4} C(m,k) ≈ 3.6 万（m=30）；本机 CPython
3.14.7 上 30 个随机整数点约 3–4 秒。

### 一维/二维输入球（数值）

1. **枚举支撑子集 1..d+1 个**（1D 至多 2 个、2D 至多 3 个；m ≤ 6 时
   候选总数 ≤ 41）。完全相同的球先去重并合并 ID。
2. **候选求解**：令活动约束 `|c − p_i| = R − a_i`。参数化
   `c = p0 + Vα`，等距条件线性化为 Gram 方程 `Gα = β + Rγ`
   （`β_j = (|v_j|² − a_j² + a_0²)/2`，`γ_j = a_j − a_0`）。
   设 `α = u + Rw` 代入 `|c − p0| = R − a_0`，得到关于 R 的一元二次
   方程；取满足 `R ≥ max 活动半径` 的实根。两球情形是其解析特例
   （中点推广：`R = (d + a0 + a1)/2`）。
3. **非负权重**：由活动法向解 `Σ λ_i n_i = 0、Σ λ_i = 1`；权重为负或
   法向线性相关的候选拒绝（对应情形由更小的支撑子集覆盖）。
4. **逐球独立可行性检查**：对每个输入球单独判定
   `dist(c, p_i) + a_i ≤ R + tol·scale`，任一不过即拒绝该候选。
5. **选优与确定性**：取 R 最小；平局支撑更少优先、下标字典序。
   容差只决定候选是否被接受，不影响排序。
6. **天然处理的边界**：大球完整包含小球（单支撑）、两球外切/内切、
   同心球、零半径点输入、共线球心（2D 三心共线时 Gram 奇异，跳过该
   三球候选，由两球候选覆盖）。

## 错误语义（不静默纠正）

两个入口共通：

- 坐标为 `NaN` / `+Inf` / `-Inf` → `ValueError`，信息含参数路径
  （点接口如 `points[1][1]`；球接口如 `balls[2][0](center)[0]`）。
- `bool` 坐标或 `bool` 型整数参数（`budget=True`、`dim=True`、
  `radius=True`）→ `TypeError`。
- 空输入、超数量上限、坐标数与维度不符、字符串等 → 带定位的
  `ValueError` / `TypeError`。
- `ids` 长度不符 → `ValueError`；重复 ID → `ValueError`（如 `ids[1]`）；
  不可哈希 → `TypeError`。
- 输入对象不会被修改；校验在任何求解之前完成，非法输入不会产生
  半更新状态（模块本身无状态）。

点接口另有：

- `budget` 非正整数 → `ValueError`/`TypeError`。
- **搜索预算耗尽与证明无解严格分开**：
  `SearchBudgetExhaustedError`（结果未知，绝不冒充最优或无解）
  ≠ `NoFeasibleCandidateError`（穷举证明不可行）。

球接口另有：

- 球数量 1..6；`dim` 仅接受整数 `1` 或 `2`；半径必须非负；
  `tol` 必须是非负有限数。
- 枚举后无可行候选时抛 `NoFeasibleCandidateError`（非空输入的正常配置
  下数学上不会发生；病态配置在给定 `tol` 下可能触发，错误信息照实说明，
  不返回近似结果冒充成功）。

## 浮点结果的误差口径

- **点接口对外的几何结果只有精确量**：`center`、`radius_squared`、
  `weight` 全部是 `Fraction`，判定与证书均精确，误差为 0。
  `center_float` / `radius_float` 仅供展示，不参与任何判定。
- **球接口是 float64 数值求解**：输入的 int/Fraction 经一次正确舍入转
  float64；Gram 方程用浮点高斯消元（规模 ≤ 2）。输出精度量级约为
  `O(eps_mach · 条件数 · 坐标尺度)`，默认 `tol=1e-10`；
  可行性谓词按上述 `tol·scale` 口径接受边界误差，支撑紧贴与法向平衡的
  残差可通过 `slacks`、`balance_residual()`、`max_violation()` 直接查看。
  若坐标尺度过大（如 1e15 以上）或球心几乎共线导致条件恶劣，应放大
  `tol` 或做点接口那样的精确建模；本入口不为病态配置提供精确证书。

## 已知限制

- 点接口：仅三维（恰好 3 个坐标）、1..30 个有理点；枚举 O(n⁴)。
  Fraction 分子分母随坐标位宽增长，输入极大整数时会变慢。
  浮点输入按其精确二进制值参与计算（如 `0.1` 即
  `3602879701896397/2^55`），这是有意的精确语义。
- 球接口（本次迭代范围）：仅 **dim ∈ {1,2}、1..6 个球、float64 数值解**；
  不做三维及以上输入球、不做动态删点/增点、不设外点预算、不提供高维
  精确（Fraction）证书。候选数 Σ_{k≤d+1} C(m,k) = O(m^{d+1})，
  m ≤ 6 时至多 41 个，不扩展到题面限定之外的规模。
- 两个入口都不宣称是任意维度 / 任意规模 / 流式 / 行业标准实现。

## 测试与独立参考说明

`tests/test_miniball3d.py`（68 个用例）：

- **原 30 个点集用例全部保留未改**：单点/两点、共线、共面、钝/直角
  退化、正四面体、重复点多 ID、30 点上限、Fraction/float 输入、全部
  校验、budget 语义、不可变性；文件内的独立参考实现（叉积 + Cramer
  法则，不 import 被测核心）继续对点集结果逐字段对照。
- **40 个输入球用例**，期望值均独立得出，不调用被测实现生成：
  - 手算解析例子：单球、两点中点、不等半径两球的
    `R=(d+a+b)/2` 公式（1D 与 2D）、大球含小球、同心球；
  - 正三角形几何关系（点与等球）、**解析构造的三不等半径支撑**
    （给定 c\*、R\* 与三条 120° 法向反推球心和半径）；
  - 零半径退化：40 组确定性随机点集与三维精确 Fraction 求解器
    （`minimum_enclosing_sphere`）逐字段对照半径、中心、支撑数；
  - 一维 100 组确定性随机配置对照独立的“最左/最右触及区间”参考；
  - 二维 60 组网格枚举配置用 61×61 细网格中心搜索给出独立最优下界；
  - 通用证书核验器独立重算逐球约束、支撑紧贴、权重凸组合、法向平衡，
    并用 24 方向数值探针验证局部（由凸性即全局）最优；
  - 输入校验（空/超上限/负半径/NaN/维度不符/bool/dim/tol/ids）、
    不可变、不修改输入、结果快照独立性、`contains_*` 边界。

## 文件清单

- `miniball3d.py` — 核心库（点集精确解 + 输入球数值解；可直接 import）。
- `tests/test_miniball3d.py` — 必要测试 + 独立参考/独立期望值。
- `demo.py` — 固定输入演示：三维点集证书；二维三球解析例（含逐球核验）；
  包含退化与零半径边界；超上限/负半径/NaN 三类真实拒绝。
- `README.md`、`requirements.txt`（无第三方依赖）。
