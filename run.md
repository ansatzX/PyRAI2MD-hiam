# PyRAI2MD kTDC 使用指南

## 目录
1. [程序运行基础](#程序运行基础)
2. [输入文件参数详解](#输入文件参数详解)
3. [data.json生成方法](#datajson生成方法)
4. [kTDC方法详解](#ktdc方法详解)
5. [结果分析](#结果分析)

---

## 程序运行基础

### 基本运行命令

```bash
# 进入工作目录
cd /path/to/working/directory

# 运行PyRAI2MD
PyRAI2MD.x input > output.log
```

### 运行环境要求

- Python 3.x
- PyRAI2MD 已安装（开发版建议用 `pip install -e .`）
- xTB 可用（GFN-FF或GFN2-xTB）
- 神经网络模型文件（如需要）

---

## 输入文件参数详解

### 完整输入文件示例

```
&CONTROL
title tpsagg
jobtype md
qm nn xtb
ml_ncpu 1
qc_ncpu 10
maxenergy 0.06
minenergy 0.06
maxgrad 0.18
mingrad 0.18

&MOLECULE
ci 3
spin 0
coupling 1 2,2 3
shape ellipsoid
highlevel 1-51
midlevel 52-612
embedding 0
cavity 19.20
factor 40
center 1 2 4 5 6

&xtb
xtb  /path/to/conda/env
xtb_nproc 10
gfnver 0
mem 4000
gfnff_topo 0

&MD
reset 1
resetstep 10
randvelo 1
temp 298.15
step 100
size 20.67
root 2
thermo nvt
direct 1
buffer 40
silent 0
verbose 2
nactype ktdc
sfhp fssh
gap 2.0

&NN
modeldir NN-tps
train_data data4880.json
nn_eg_type 2

&EG
invd_index tpss
depth 4
nn_size 700
batch_size 64
loss_weights 1 1
use_reg_activ l2
use_reg_weight l2
use_reg_bias l2
reg_l2 1e-9
epo 600
epostep 10
learning_rate 1e-3
learning_rate_step 1e-3 1e-4 1e-5
epoch_step_reduction 350 200 50

&EG2
invd_index  tpss
depth 4
nn_size 600
batch_size 64
loss_weights 1 1
use_reg_activ l2
use_reg_weight l2
use_reg_bias l2
reg_l2 1e-7
epo 600
epostep 10
learning_rate 1e-3
learning_rate_step 1e-3 1e-4 1e-5
epoch_step_reduction 350 200 50
```

### CONTROL 模块

| 参数 | 说明 | 示例 |
|------|------|------|
| title | 作业标题 | tpsagg |
| jobtype | 作业类型 | md (分子动力学) |
| qm | QM方法组合 | nn xtb (神经网络+xTB) |
| ml_ncpu | ML计算CPU数 | 1 |
| qc_ncpu | QC计算CPU数 | 10 |
| maxenergy/minenergy | 能量自适应阈值 | 0.06 |
| maxgrad/mingrad | 梯度自适应阈值 | 0.18 |

### MOLECULE 模块

| 参数 | 说明 | 示例 |
|------|------|------|
| ci | 电子态数目 | 3 |
| spin | 自旋多重度 | 0 (闭壳层) |
| coupling | 耦合的态对 | 1 2,2 3 (态1-2和态2-3耦合) |
| highlevel | 高层QM区域原子 | 1-51 |
| midlevel | 中层QM区域原子 | 52-612 |
| cavity | 椭球空腔半径 | 19.20 |

### xtb 模块

| 参数 | 说明 | 示例 |
|------|------|------|
| xtb | xTB环境路径 | /path/to/conda/env |
| xtb_nproc | xTB并行核数 | 10 |
| gfnver | xTB版本 | 0 (GFN-FF), 1 (GFN1-xTB), 2 (GFN2-xTB) |
| mem | 内存(MB) | 4000 |

### MD 模块（核心）

| 参数 | 说明 | 示例 |
|------|------|------|
| step | MD步数 | 100 |
| size | 时间步长(fs) | 20.67 |
| temp | 温度(K) | 298.15 |
| thermo | 系综 | nvt (正则系综) |
| root | 初始电子态 | 2 (从第2态开始) |
| **nactype** | **NAC计算方法** | **ktdc** (关键参数!) |
| **sfhp** | **表面hop方法** | **fssh** (Fewest Switches) |
| **gap** | **kTDC能隙阈值(eV)** | **2.0** (关键参数!) |
| randvelo | 初始速度 | 0 (读取文件), 1 (随机) |
| silent | 静默模式 | 0 (显示输出) |
| verbose | 详细程度 | 0-2 (越大越详细) |

### NN 模块

| 参数 | 说明 | 示例 |
|------|------|------|
| modeldir | 模型目录 | NN-tps |
| train_data | 训练数据文件 | data4880.json |

---

## data.json生成方法

### 为什么需要data.json

PyRAI2MD的神经网络模块需要一个初始化的数据文件，即使不进行训练也需要这个文件来确定系统的基本信息（原子数、态数等）。

### 自动生成脚本

使用 `generate_data_json.py` 脚本自动生成：

```python
#!/usr/bin/env python3
import argparse
import json
import os
import re

def read_hyper_json(hyper_file):
    """从hyper.json读取系统参数"""
    with open(hyper_file, 'r') as f:
        hyper = json.load(f)

    # 获取原子数和态数
    if 'model' in hyper:
        natom = hyper['model'].get('atoms', 0)
        nstate = hyper['model'].get('states', 0)
    else:
        # 尝试其他可能的结构
        natom = hyper.get('atoms', 0)
        nstate = hyper.get('states', 0)

    return natom, nstate

def read_xyz(xyz_file):
    """从XYZ文件读取原子类型和坐标"""
    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    natom = int(lines[0].strip())
    atoms = []
    coords = []

    for line in lines[2:2+natom]:
        parts = line.strip().split()
        if len(parts) >= 4:
            atoms.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])

    return atoms, coords

def generate_data_json(natom, nstate, atoms=None, coords=None, use_real_coords=False):
    """生成data.json数据结构"""
    nnac = nstate * (nstate - 1) // 2

    if atoms is None:
        atoms = ['C'] * natom

    # 坐标
    if use_real_coords and coords is not None:
        xyz = coords
    else:
        xyz = [[0.0, 0.0, 0.0] for _ in range(natom)]

    # 能量占位符
    energy = [0.0 for _ in range(nstate)]

    # 梯度占位符
    grad = [[[0.0, 0.0, 0.0] for _ in range(natom)] for _ in range(nstate)]

    # NAC占位符
    if nnac > 0:
        nac = [[[0.0, 0.0, 0.0] for _ in range(natom)] for _ in range(nnac)]
    else:
        nac = []

    # 构建数据结构
    data = {
        'Atoms': atoms,
        'Coords': xyz,
        'Energy': energy,
        'Grad': grad,
        'Nac': nac
    }

    return data

def main():
    parser = argparse.ArgumentParser(description='Generate data.json for PyRAI2MD')
    parser.add_argument('--nn-dir', default='NN-tps', help='Neural network directory')
    parser.add_argument('--xyz', help='XYZ file for atomic coordinates')
    parser.add_argument('-o', '--output', default='data4880.json', help='Output JSON file')
    parser.add_argument('--use-real-coords', action='store_true',
                        help='Use real coordinates from XYZ file')

    args = parser.parse_args()

    # 查找hyper.json
    hyper_file = os.path.join(args.nn_dir, 'energy_gradient', 'hyper_v0.json')
    if not os.path.exists(hyper_file):
        hyper_file = os.path.join(args.nn_dir, 'hyper_v0.json')
    if not os.path.exists(hyper_file):
        hyper_file = os.path.join(args.nn_dir, 'hyper.json')

    if not os.path.exists(hyper_file):
        raise FileNotFoundError(f"Cannot find hyper.json in {args.nn_dir}")

    print(f"Reading from: {hyper_file}")
    natom, nstate = read_hyper_json(hyper_file)
    print(f"  natom = {natom}")
    print(f"  nstate = {nstate}")

    # 读取XYZ文件
    atoms = None
    coords = None
    if args.xyz and os.path.exists(args.xyz):
        print(f"Reading from: {args.xyz}")
        atoms, coords = read_xyz(args.xyz)
        if len(atoms) != natom:
            print(f"  Warning: XYZ has {len(atoms)} atoms, but hyper.json says {natom}")

    # 生成data.json
    print(f"Generating: {args.output}")
    data = generate_data_json(natom, nstate, atoms, coords, args.use_real_coords)

    # 写入文件
    with open(args.output, 'w') as f:
        json.dump([data], f, indent=2)

    print(f"Done!")
    print(f"  Atoms: {len(data['Atoms'])}")
    print(f"  States: {len(data['Energy'])}")
    print(f"  NAC pairs: {len(data['Nac'])}")

if __name__ == '__main__':
    main()
```

### 使用方法

```bash
# 基本使用（自动检测）
python generate_data_json.py

# 指定输出文件名
python generate_data_json.py -o data4880.json

# 指定力场目录和XYZ文件
python generate_data_json.py --nn-dir NN-tps --xyz tps.init.xyz

# 使用XYZ文件中的真实坐标
python generate_data_json.py --xyz tps.init.xyz --use-real-coords
```

### data.json文件格式

```json
[
  {
    "Atoms": ["C", "C", "H", ...],
    "Coords": [
      [0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0],
      ...
    ],
    "Energy": [0.0, 0.0, 0.0],
    "Grad": [
      [[0.0, 0.0, 0.0], ...],
      [[0.0, 0.0, 0.0], ...],
      [[0.0, 0.0, 0.0], ...]
    ],
    "Nac": [
      [[0.0, 0.0, 0.0], ...],
      [[0.0, 0.0, 0.0], ...]
    ]
  }
]
```

---

## kTDC方法详解

### 什么是kTDC

kTDC（kurtotic time-dependent curvature）是一种从能量历史计算非绝热耦合（NAC）的方法，不需要直接计算NAC向量。

### kTDC原理

kTDC使用Baeck-An近似：

1. 需要当前步和前两步的能量
2. 计算能量差的二阶导数
3. 检查两个条件：
   - **能隙条件**：|gap| ≤ threshold
   - **曲率条件**：d²V/dt² / dV/dt > 0

kTDC公式：
```
dVt = E_s1(t) - E_s2(t)
dVt_dt = E_s1(t-Δt) - E_s2(t-Δt)
dVt_2dt = E_s1(t-2Δt) - E_s2(t-2Δt)

d²Vdt2 = (dVt - 2*dVt_dt + dVt_2dt) / (Δt²)

if |dVt| > gap_threshold or |dVt_dt| > gap_threshold:
    NAC = 0
elif d²Vdt2 / dVt > 0:
    NAC = -sqrt(d²Vdt2 / dVt) / 2
else:
    NAC = 0
```

### kTDC参数设置

#### 1. 选择kTDC方法

```
&MD
nactype ktdc
```

#### 2. 启用表面hopping

```
&MD
sfhp fssh
```

#### 3. 设置能隙阈值

```
&MD
gap 2.0
```

**注意**：
- 单位是eV
- 代码内部会自动转换为Hartree（除以27.2114）
- 参考值：TPS分子用2.0 eV

### kTDC代码实现（fssh.pyx）

```cython
cdef kTDC(int s1, int s2, np.ndarray E, np.ndarray Ep, np.ndarray Epp, float dt, float gap):
    """
    kTDC method for nonadiabatic coupling

    Parameters
    ----------
    s1, s2 : state indices
    E : energies at current step (t)
    Ep : energies at previous step (t-Δt)
    Epp : energies at step before previous (t-2Δt)
    dt : time step in a.u.
    gap : gap threshold in eV (will be converted to Hartree)

    Returns
    -------
    nacme : nonadiabatic coupling matrix element
    """
    # Convert gap from eV to Hartree
    gap = gap / 27.211396132

    # Ensure s1 < s2
    if s1 > s2:
        s1, s2 = s2, s1

    # Energy differences
    dVt = E[s1] - E[s2]
    dVt_dt = Ep[s1] - Ep[s2]
    dVt_2dt = Epp[s1] - Epp[s2]

    # Check gap threshold condition
    if abs(dVt) > gap or abs(dVt_dt) > gap:
        return 0.0

    # Second derivative
    d2Vdt2 = (dVt - 2 * dVt_dt + dVt_2dt) / (dt ** 2)

    # Check curvature condition
    if d2Vdt2 / dVt > 0:
        nacme = -(d2Vdt2 / dVt) ** 0.5 / 2
    else:
        nacme = 0.0

    return nacme
```

### kTDC优缺点

| 优点 | 缺点 |
|------|------|
| 不需要训练NAC的神经网络 | 需要3步能量历史 |
| 计算速度快 | 能隙太大时返回0 |
| 内存占用小 | 曲率条件不满足时返回0 |
| 适用于任何能计算能量的方法 | NAC值可能被高估（2-10倍） |

### kTDC参数调优建议

#### 能隙阈值（gap）

| 分子 | 推荐值 | 文献参考 |
|------|--------|----------|
| HPS | 2.05 eV | J. Phys. Chem. Lett. 2023 |
| TPS | 2.0 eV | J. Phys. Chem. Lett. 2023 |
| COTh | 1.0 eV | J. Phys. Chem. Lett. 2023 |

**注意**：阈值越大，kTDC越常计算，但可能越不准确。

#### 初始速度（randvelo）

```
&MD
randvelo 1  ! 随机速度，适合探索
! 或
randvelo 0  ! 从文件读取，适合重现
```

#### 详细程度（verbose）

```
&MD
verbose 0  ! 最少输出
verbose 1  ! 标准输出
verbose 2  ! 详细输出（含NAC矩阵）
```

---

## 结果分析

### 输出文件

| 文件 | 内容 |
|------|------|
| tpsagg.log | 主日志文件 |
| tpsagg.md.energies | 每步能量 |
| tpsagg.md.xyz | 每步坐标 |
| tpsagg.md.velo | 每步速度 |

### 分析脚本

#### 1. 分析kTDC和能隙

```python
#!/usr/bin/env python3
import re
import numpy as np

# Read log file
with open('tpsagg.log', 'r') as f:
    log_content = f.read()

# Find all Gnuplot lines for energies
gnuplot_lines = re.findall(r'Gnuplot:\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', log_content)

print(f"Found {len(gnuplot_lines)} Gnuplot lines\n")

# Extract state energies (in Hartree)
if len(gnuplot_lines) > 0:
    steps = np.arange(1, len(gnuplot_lines) + 1)
    e0 = np.array([float(line[4]) for line in gnuplot_lines])  # state 1
    e1 = np.array([float(line[5]) for line in gnuplot_lines])  # state 2
    e2 = np.array([float(line[6]) for line in gnuplot_lines])  # state 3

    # Calculate gaps in eV
    gap01 = np.abs(e1 - e0) * 27.2114
    gap12 = np.abs(e2 - e1) * 27.2114

    print("=== Energy Gaps (eV) ===")
    print(f"Gap 1-2: min={gap01.min():.4f}, max={gap01.max():.4f}, mean={gap01.mean():.4f}")
    print(f"Gap 2-3: min={gap12.min():.4f}, max={gap12.max():.4f}, mean={gap12.mean():.4f}")

    # Simulate kTDC calculation
    dt = 20.67  # from input file
    gap_threshold = 2.0 / 27.2114  # 2.0 eV in Hartree

    ktdc_values = []
    gap_skipped = 0
    curv_skipped = 0
    calculated = 0

    for i in range(3, len(steps)):
        dVt = e0[i] - e1[i]
        dVt_dt = e0[i-1] - e1[i-1]
        dVt_2dt = e0[i-2] - e1[i-2]

        if abs(dVt) > gap_threshold or abs(dVt_dt) > gap_threshold:
            gap_skipped += 1
        else:
            d2Vdt2 = (dVt - 2 * dVt_dt + dVt_2dt) / (dt ** 2)
            if d2Vdt2 / dVt > 0:
                nacme = -(d2Vdt2 / dVt) ** 0.5 / 2
                ktdc_values.append(nacme)
                calculated += 1
            else:
                curv_skipped += 1

    print("\n=== kTDC Statistics ===")
    print(f"Total steps: {len(steps)-3}")
    print(f"Calculated: {calculated}")
    print(f"Skipped (gap > threshold): {gap_skipped}")
    print(f"Skipped (curvature condition): {curv_skipped}")

    if calculated > 0:
        ktdc_array = np.array(ktdc_values)
        print(f"kTDC values: min={ktdc_array.min():.2e}, max={ktdc_array.max():.2e}, mean={ktdc_array.mean():.2e}")
```

#### 2. 从verbose输出提取NAC

```python
#!/usr/bin/env python3
import re
import numpy as np

# Read log file
with open('tpsagg.log', 'r') as f:
    log_content = f.read()

# Find all TEST sections (verbose output)
test_sections = re.findall(r'-------------- TEST ----------------.*?(?=-------------- TEST ----------------|$)', log_content, re.DOTALL)

print(f"Found {len(test_sections)} TEST sections\n")

# Extract NAC matrices
nac_12_values = []
nac_23_values = []

for i, section in enumerate(test_sections):
    # Find Iter number
    iter_match = re.search(r'Iter:\s*(\d+)', section)
    if not iter_match:
        continue

    step = int(iter_match.group(1))

    # Find Current NAC matrix
    nac_match = re.search(r'Current NAC\s*\[\[(.*?)\]\]', section, re.DOTALL)
    if not nac_match:
        continue

    # Parse the matrix
    matrix_content = nac_match.group(1)
    complex_numbers = re.findall(r'(-?\d+\.\d+)([+-]\d+\.\d+j)?', matrix_content)

    nac_values = []
    for num_str, imag_str in complex_numbers:
        nac_values.append(float(num_str))

    if len(nac_values) >= 9:
        # 3x3 matrix: indices 0-8
        nac_12 = abs(nac_values[1])  # Dt[0,1]
        nac_23 = abs(nac_values[5])  # Dt[1,2]

        nac_12_values.append(nac_12)
        nac_23_values.append(nac_23)

if nac_12_values:
    print("=== kTDC NAC Values ===")
    print(f"NAC(1-2): min={min(nac_12_values):.6f}, max={max(nac_12_values):.6f}, mean={np.mean(nac_12_values):.6f}")
    print(f"NAC(2-3): min={min(nac_23_values):.6f}, max={max(nac_23_values):.6f}, mean={np.mean(nac_23_values):.6f}")

# Find final hopping probability
final_match = re.search(r'&surface hopping information.*?Accumulated probability:\s*(\S+)', log_content, re.DOTALL)
if final_match:
    print(f"\nFinal accumulated probability: {final_match.group(1)}")
```

### 常见问题

#### Q: kTDC为什么经常返回0？

A: kTDC返回0有两个原因：
1. 能隙太大（> gap阈值）
2. 曲率条件不满足（d²V/dt² / dV/dt ≤ 0）

可以用分析脚本统计各自的比例。

#### Q: 如何增加hopping概率？

A: 可以尝试：
1. 增大gap阈值（但可能降低准确性）
2. 提高温度（增加动能）
3. 延长模拟时间

#### Q: kTDC和直接计算NAC有什么区别？

A:
- kTDC：从能量历史推导，速度快，需要3步历史，可能高估
- 直接NAC：从电子结构计算，准确，但训练NN更困难

---

## 快速参考

### 最小输入文件（kTDC）

```
&CONTROL
title test
jobtype md
qm nn xtb

&MOLECULE
ci 3
spin 0
coupling 1 2,2 3
highlevel 1-51
midlevel 52-612

&xtb
xtb  /path/to/xtb
xtb_nproc 10
gfnver 0

&MD
step 100
size 20.67
temp 298.15
root 2
thermo nvt
nactype ktdc
sfhp fssh
gap 2.0
verbose 2

&NN
modeldir NN-tps
train_data data4880.json
```

### 运行检查清单

- [ ] data.json已生成
- [ ] 神经网络模型目录存在
- [ ] xTB路径正确
- [ ] `nactype ktdc` 已设置
- [ ] `sfhp fssh` 已设置
- [ ] `gap` 阈值已设置
- [ ] `verbose 2` 用于调试

### 参考文献

1. J. Phys. Chem. Lett. 2023, 14, XXX-YYY (kTDC方法)
2. PyRAI2MD documentation: https://pyrai2md.readthedocs.io/
