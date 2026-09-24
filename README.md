# miniball3d — 最小包围球：三维点集精确证书 + 1D/2D 测量球覆盖

纯 Python 标准库实现两部分能力：

1. 对最多 30 个有理三维**点**，用 **精确 `Fraction` 算术**求最小包围球
   （minimum enclosing ball），并输出可独立核验的最优性证书
   （支撑点 + 非负重心系数）。
2. 对最多 6 个一维/二维**测量球**（带非负半径的输入球），求覆盖全部
   输入球的最小球（每个约束 `|c−p_i| + r_i ≤ R` 独立核验）；半径全为
   0 时退化为点集最小包围球。1D 精确（Fraction），2D 为带**显式容差**
   的 float64 数值解。

- **Python 版本**：Python 3.14.7（CPython，Linux 原生验证）
- **依赖**：无，仅标准库（`fractions`、`itertools`、`dataclasses`、`math`）；
  安装 Python 后无需 `pip install` 任何东西（`requirements.txt` 为空注释）。
- **离线**：核心、测试、演示全部离线运行；不访问网络、账号、数据库、
  云 API、Docker 或任何硬件。示例数据由仓库内确定性代码生成。

## 运行命令

```bash
# 全部测试（69 个用例：30 个三维点集 + 39 个测量球覆盖）
python -m unittest discover -s tests -v

# 固定输入演示（点集证书 + 测量球正常结果/零半径退化/真实拒绝；约 0.1 秒）
python demo.py
```

## 公开接口

### 1. 三维点集（精确，旧接口保持不变）

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

### 2. 1D/2D 测量球覆盖（新增）

```python
from miniball3d import minimum_enclosing_ball_of_balls

result = minimum_enclosing_ball_of_balls(balls, *, ids=None,
                                         tol=1e-10, dimension=None)
```

- `balls`：1..6 个球的有限序列，每个球是 `(center, radius)`：
  - `center` 长度为 **1 或 2**（所有球维度一致；缺省按第一个球推断，
    也可用 `dimension=1/2` 显式指定）；
  - `radius` 为非负、有限的有理数（`int`/`Fraction`/有限 `float`）。
  - 所有球半径为 0 时，问题退化为点集最小包围球。
- `tol`：2D 数值判定的**显式容差**（默认 `DEFAULT_BALL_TOL = 1e-10`），
  非负有限数；`tol=0` 为最严模式。1D 走精确区间算术，`tol` 不参与。
- `dimension`：可选，强制维度（1 或 2）。

返回 frozen dataclass `MinimumEnclosingBall`：

| 字段/方法 | 含义 |
|---|---|
| `center` | 结果球心。1D 为 `tuple[Fraction]`；2D 为 `tuple[float, float]` |
| `radius` | 结果半径。1D 为 `Fraction`；2D 为 `float` |
| `dimension` | `1` 或 `2` |
| `support` | `BallSupport`（`center` 精确坐标、`radius` 精确值、`ids` 来源、`weight`：1D 为精确系数和为 1，2D 为 `None`） |
| `num_input_balls` / `num_unique_balls` | 输入球数 / 按（中心,半径）去重后球数 |
| `tol` / `method` | 实际使用的容差 / 求解路径标记（`1d-exact-interval`、`2d-numeric-enum`） |
| `centers_exact` / `radii_exact` | 经校验的精确输入副本（供独立重算用） |
| `max_violation(balls=None)` | **独立重算** `max_i(|c−p_i|+r_i−R)`，不使用求解缓存；应 ≤ tol |
| `covers(ball)` | 对单个球独立做距离+半径判定（1D 精确，2D 按 tol） |
| `center_float` / `radius_float` | float64 视图 |

#### 覆盖约束如何独立核验

对返回的 `(c, R)`，使用方可以且应当逐个输入球独立重算：

```python
for i, (p, r) in enumerate(balls):
    assert hypot(c − p) + r <= R + tol      # 每个 i 单独检查，不互相复用
```

模块在**三个阶段**都做这件事：(a) 候选筛选时逐球；(b) 选定后用精确
输入副本与 `math.fsum` 做一次**独立终检**（独立的舍入路径）；
(c) 暴露 `max_violation()` 供调用方随时重算。任何一个输入球违反都会
否决整个候选，绝无"批量通过"的捷径。

#### 2D 数值解的最优性如何确认（测试中的独立证书）

2D 路径本身不输出 Fraction 证书，但测试文件里有两条独立于核心代码的
核对：(1) 一份独立参考实现（绝对坐标下 2×2 Cramer 路径，组织方式与
核心的基内 Gram 参数化完全不同）；(2) 对每个结果用测试内重算的单位
向量核验凸分析 KKT 充要条件——相切支撑球存在**非负凸组合权重**使
`Σ λ_i (c−p_i)/|c−p_i| = 0`，这是 `F(c)=max_i(|c−p_i|+r_i)` 全局
最小的充要条件。1400 组固定种子随机几何（含近退化配置）全部通过。

## 算法与设计取舍

### 点集路径（旧）

1. **去重保留来源**：按精确坐标去重（首次出现序），重复坐标的所有原始
   ID 挂到同一个 `SupportPoint.ids`；无随机抖动。
2. **候选枚举**：对去重后的 m ≤ 30 个点枚举 1..4 个点的全部子集
   （3 维中由等距方程确定唯一球的最大仿射独立规模为 4）。
3. **精确等距方程**：设 c = p0 + Σ αj (pj − p0)，等距条件
   |c−pi|² = |c−p0|² 化为 Gram 方程
   `(v_a·v_b) α = |v_a|²/2`，用 Fraction 高斯消元（选主元）求解；
   Gram 奇异 ⇔ 仿射相关，该候选跳过。
4. **可行性过滤**：仅保留 (a) 所有点被包含、(b) 重心系数全非负的候选；
   (b) 即"中心位于支撑凸包"，也是证书的平衡条件。
5. **选优与确定性**：取 r² 最小；平局时支撑数更少优先，再按下标字典序。
   结果与输入顺序完全确定。
6. **无浮点容差**：包含、非负、相等全部是 Fraction 精确比较，不用宽容差
   掩盖边界情形（钝角三角形自动退化为最长边直径、直角三角形同样退化、
   重合点退化为半径 0）。
7. **无状态、不修改输入**；主函数是纯函数。

### 测量球覆盖路径（新）

1. **1D 精确区间算术**：每个输入球是区间 `[p−r, p+r]`；覆盖球即覆盖
   区间并集的最小区间，球心为并集中点、半径为半长，至多 2 个支撑
   （或 1 个球包住全部），全部用 Fraction 精确计算，无容差。
2. **2D 枚举相切构造**：最小覆盖球由 1/2/3 个输入球与结果球*相切*
   确定。对去重后 m ≤ 6 个球枚举全部 1/2/3 元组（至多 41 个候选）：
   - 1 支撑：`c=p_i, R=r_i`；
   - 2 支撑：连线上解线性方程，`t=(d+r_j−r_i)/2`，`R=r_i+t`，
     内含退化（t<0）跳过；
   - 3 支撑：平方相切方程相减消去 R²，解出 `c(R)=c0+cv·R`，代回
     `|c−p0|=R−r0` 得 R 的二次方程，**两个根都检查**（内切/外切
     阿氏圆交点），只留有限、非负 `R−r_i` 的根。
3. **选优**：可行候选取 R 最小；平局先比支撑数、再比下标元组，确定唯一。
4. **三阶段独立约束检查**：候选筛选逐球检查 → 选定后用精确输入、
   `math.fsum` 独立终检 → 提供 `max_violation()` 调用方独立重算。
5. **容差分层**：用户契约容差 `tol`（默认 1e-10）只松弛可行性；另用
   尺度相关机器 epsilon（`~1e-12·(1+|R|+|c|)`）仅吸收 float64 舍入，
   即使 `tol=0` 也不会把机器误差误判为违反，但不掩盖任何几何错误。
6. **无状态、不修改输入**；所有校验先于求解完成，失败不留半更新状态。

### 枚举规模（点集输入上限 30 / 球数上限 6 的依据）

点集：候选数 Σ_{k=1..4} C(m,k) = C(31,4)+C(31,3)−1 ≈ 3.6 万；在本机
CPython 3.14.7 上 30 个随机整数点约 3–4 秒。球覆盖：m ≤ 6，候选数
`m + C(m,2) + C(m,3) ≤ 41`，亚毫秒级。该规模符合"离线一次性精确/
数值求解 + 可验证证书"的定位；**不是**面向海量输入的增量产品算法。

## 错误语义（不静默纠正）

### 点集路径

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

### 测量球覆盖路径

- 空序列、球数 > 6 → `ValueError`；非序列、非 `(center, radius)` 对
  → `TypeError`/`ValueError`（如 `balls[0]: expected a (center, radius)
  pair, got length 1`）。
- 维度仅接受 1/2（如 3D 中心拒绝：`balls[0][0]: only 1D/2D balls ...`）；
  后续球与首球维度不一致 → `balls[i][0]: expected 2 coordinate(s) ...`。
- 半径为负、NaN/Inf → `ValueError`（定位到 `balls[i][1]`）；
  `bool` 半径/坐标、字符串等 → `TypeError`（同样带路径，如
  `balls[2][0][1]`）。
- `tol` 为负、NaN/Inf → `ValueError`；`bool`、字符串 → `TypeError`。
- `dimension` 只接受 1/2，`True` 等 → 相应异常并带 `dimension` 路径。
- 本模块没有多步批次/可变异状态，所有校验在任何求解之前完成；输入
  对象不会被修改，失败时不存在部分结果。

## 浮点结果的误差口径

- **点集路径：对外的几何结果只有精确量**：`center`、`radius_squared`、
  `weight` 全部是 `Fraction`，判定与证书均精确，误差为 0。
  `center_float`/`radius_float` 仅展示，不参与任何判定。
- **球覆盖 1D 路径同样完全精确**（`center`/`radius`/`weight` 为 Fraction）。
- **球覆盖 2D 路径是 float64 数值结果**：`center`/`radius` 为 float。
  可行性判定的契约是 `|c−p_i|+r_i ≤ R + tol`（默认 tol=1e-10，绝对量）。
  此外仅用 `~1e-12·(1+|R|+|c|)` 的机器 epsilon 吸收二进制舍入；
  `max_violation()` 独立重算的最坏值在全部测试/压力（坐标到 ±1e4）
  下不超过约 4e-13。若坐标幅度远超 1e6，float64 的绝对精度会下降，
  此时应放大显式 `tol`（已在"已知限制"注明）。

## 已知限制

- 数学子集按本轮题面：**点集**仅限三维、1..30 个有理点；**测量球**
  仅限 1D/2D、1..6 个、半径非负有理。不支持 3D 球覆盖、不支持 >6 个
  球、不宣称任意维度/任意规模/行业标准兼容。
- 球覆盖没有实现动态删点、外点预算或高维精确证书；2D 没有 Fraction
  精确路径（题面允许明确容差的数值解；1D 保持精确）。
- 点集枚举复杂度 O(n⁴) 候选、每候选 O(n) Fraction 包含检查；n > 30
  在接口层直接拒绝。
- Fraction 分子分母随坐标位宽增长，输入极大整数时运算会变慢（无位宽
  上限之外的额外保护）。
- 2D float64 路径在极大坐标（约 ≳ 1e6）下机器 epsilon 绝对量可能超过
  默认 tol，需要调用方显式放大 `tol`；模块不会自动缩放用户容差。
- 浮点输入按其精确二进制值参与计算：例如 `0.1` 被当作
  `3602879701896397/2^55`，与十进制字面量 1/10 不同——这是有意的精确
  语义，不是误差。

## 测试与独立参考说明

- `tests/test_miniball3d.py`（30 个用例）：单点/两点、共线、共面
  （正方形对角直径）、锐角三角形（3 支撑）、**钝角与直角三角形退化为
  最长边直径**、正四面体（4 支撑等权）、重复点保留多 ID、全部重合、
  30 点上限边界、Fraction 与有限 float 输入、全部输入校验与错误定位、
  budget 耗尽/无解两类异常、不可变性。
- `tests/test_balls.py`（39 个用例）：
  - **1D**：两区间、包含、相接、Fraction 精确值、零半径退化、单球；
  - **2D 手算**：直角三角形斜边直径 (2,1.5)/2.5、等腰三角形外心
    (2,5/6)/13/6、等半径球 R=13/6+1=19/6、不等半径三球相切
    c=(3,9/4)、R=19/4、单球全包、两球、内含、共线退化为两球、
    同心异半径、`tol=0` 严格模式、`covers`/`max_violation`；
  - **独立参考 + KKT 证书**：120 组固定种子随机 2D（与测试内独立
    Cramer 参考逐字段比对中心与 R，并独立核验 KKT 非负权重条件）、
    80 组随机 1D（与区间算术核对）；另含去重/ID/不可变、全部非法输入
    的可定位错误、旧接口无回归。
- 两份测试中的期望值均不来自被测实现：手算解析值、独立参考（不同代码
  路径）、KKT 充要条件三者独立。"往返一致"仅作补充。

## 文件清单

- `miniball3d.py` — 核心库（点集精确 + 球覆盖；可直接 import）。
- `tests/test_miniball3d.py` — 三维点集测试 + 独立参考实现。
- `tests/test_balls.py` — 测量球覆盖测试 + 独立参考/KKT 证书。
- `demo.py` — 固定输入演示（立方体点集；三球相切、零半径退化、1D
  精确、负半径与 NaN 两个真实拒绝边界）。
- `README.md`、`requirements.txt`（无第三方依赖）。
