# meb3d — 三维有理点集的最小包围球证书

离线、纯 Python 3.14.7 标准库实现（只用 `fractions` / `itertools` / `math` /
`dataclasses` / `typing`），无需安装任何第三方依赖，不需要网络、账号、
数据库或 Docker。

对最多 30 个有理三维点求精确最小包围球（minimum enclosing ball），
并输出可独立核验的最优性证书：支撑点（1..4 个，仿射独立）及非负、
和为 1 的重心系数。全部计算使用 `Fraction` 精确有理数运算，不输出
开根（半径以平方给出），不使用随机抖动，结果对输入顺序确定。

## 运行

```bash
python -m unittest discover -s tests -v   # 单元测试（约 5 秒）
python demo.py                            # 演示（约 1 秒）
```

## 接口

```python
from meb3d import min_enclosing_ball, verify_certificate

cert = min_enclosing_ball(points, ids=None)   # -> BallCertificate
ok   = verify_certificate(points, cert, ids=None)  # -> True 或抛异常
```

- `points`：1..30 个点的可迭代序列，每个点是 3 个坐标，坐标接受
  `int` / `Fraction` / 有限 `float`（float 按其二进制值精确转换为
  有理数，不做四舍五入；要完全可控请用 `int`/`Fraction`）。
- `ids`：可选，与 `points` 等长的唯一可哈希 ID 序列，缺省为 `0..n-1`。
  坐标重复的点会被去重，但其全部原始 ID 保留在证书支撑点的
  `SupportPoint.ids` 中。

`BallCertificate`（frozen dataclass）：

| 字段 | 含义 |
|---|---|
| `center` | 球心，3 个 `Fraction` |
| `radius_squared` | 半径平方，`Fraction`（不开根） |
| `support` | `SupportPoint` 元组，每项含 `ids`（来源 ID 元组）、`coordinates`、`weight`（重心系数 λ ≥ 0，Σλ = 1） |

`verify_certificate` 独立核验：支撑坐标属于输入、λ ≥ 0 且和为 1、
球心等于支撑加权组合、支撑点在边界上、所有输入点被包含、对偶恒等式
r² = Σλᵢ|sᵢ|² − |c|² 成立。其中“包含性 + λ 在单纯形上 + 对偶目标值
等于 r²”构成不依赖搜索算法的最优性完整初等证明。

## 算法与设计取舍

- 3 维中最小包围球由边界上至多 4 个仿射独立支撑点决定，且球心落在
  支撑凸包内。模块枚举去重后点集的全部 1..4 点子集，对每个仿射独立
  子集用等距方程（重心坐标下的精确线性方程组，高斯-若尔当消元）
  解出候选球心，只保留“λ 全非负 且 球包含所有点”的候选，取半径
  平方最小者。枚举是穷举，因此结果是精确全局最优，不是启发式。
- 并列时保留先枚举到的（更小支撑尺寸优先），行为确定。
- 精确 `Fraction` 运算：结果无舍入误差，不存在浮点误差口径问题；
  若调用方自行 `float()` 转换，该舍入由调用方负责。

## 错误语义

所有异常继承 `meb3d.MebError`：

- `InvalidInputError`(= `ValueError`)：非法输入，消息含可定位位置
  （如 `points[3][1]`）。拒绝：NaN/Infinity、bool 坐标、非数值坐标、
  维数 ≠ 3、空输入、超过 30 点、ID 长度不符/重复/不可哈希。
  不做任何静默纠正。
- `SearchBudgetExhaustedError`：候选枚举超出防御性搜索预算
  （正常输入不可达的保险丝），表示搜索未完成，与“无解”严格区分。
- `NoFeasibleCandidateError`：完整枚举后无可行候选（理论上不可能，
  出现即实现缺陷信号）。
- `CertificateVerificationError`：证书核验失败，消息指明被破坏的
  不变式。

## 输入上限与已知限制

- 接口上限：30 个输入点（去重前计）。枚举规模为
  C(n,1)+…+C(n,4) ≤ 31930 个候选，n=30 的整数坐标用例在本机约
  数秒完成；坐标分母巨大时 `Fraction` 运算会变慢，这是精确算术的
  固有代价。
- 仅实现本题契约的三维、有理数、最小包围球证书子集，不宣称兼容
  任何通用计算几何标准或支持更高维。

## 文件

- `meb3d.py` — 核心模块（算法 + 数据结构 + 错误 + 证书核验）
- `tests/test_meb3d.py` — 单元测试；内含独立编写的暴力参考实现
  （不调用被测核心）用于 ≤ 8 点对照，及对偶恒等式独立最优性核验
- `demo.py` — 固定种子确定性数据的真实计算演示 + 一次实际触发的
  NaN 输入拒绝
