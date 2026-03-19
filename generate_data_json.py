#!/usr/bin/env python3
"""
自动从神经网络力场和分子结构生成 PyRAI2MD 训练数据 JSON 文件

功能：
1. 从力场目录的 hyper.json 自动读取系统参数
2. 从 XYZ 文件读取原子类型和坐标
3. 生成完整的 dataxxxx.json 文件

使用方法：
    python generate_data_json.py [选项]
"""

import json
import numpy as np
from pathlib import Path
import argparse


def read_hyper_parameters(nn_dir):
    """
    从力场目录读取超参数

    Args:
        nn_dir: 神经网络力场目录路径

    Returns:
        dict: 包含 natom, nstate, nnac, nsoc 的字典
    """
    nn_path = Path(nn_dir)

    # 尝试不同的 hyper 文件路径
    hyper_files = [
        nn_path / "energy_gradient" / "hyper_v0.json",
        nn_path / "energy_gradient" / "hyper_v1.json",
        nn_path / "hyper_v0.json",
        nn_path / "hyper_v1.json",
    ]

    hyper_file = None
    for f in hyper_files:
        if f.exists():
            hyper_file = f
            break

    if hyper_file is None:
        raise FileNotFoundError(
            f"找不到 hyper 文件，在 {nn_dir} 及其子目录中"
        )

    with open(hyper_file, "r") as f:
        hyper = json.load(f)

    # 提取参数
    model = hyper.get("model", {})
    natom = model.get("atoms", 0)
    nstate = model.get("states", 1)

    # 计算 NAC 对数：nstate * (nstate - 1) / 2
    nnac = nstate * (nstate - 1) // 2
    nsoc = 0  # 默认无自旋轨道耦合

    print(f"✅ 从 {hyper_file.name} 读取参数:")
    print(f"   原子数: {natom}")
    print(f"   电子态数: {nstate}")
    print(f"   NAC 对数: {nnac}")

    return {
        "natom": natom,
        "nstate": nstate,
        "nnac": nnac,
        "nsoc": nsoc,
    }


def read_xyz_file(xyz_file, natom=None):
    """
    从 XYZ 文件读取原子类型和坐标

    Args:
        xyz_file: XYZ 文件路径
        natom: 预期的原子数（用于验证）

    Returns:
        tuple: (atoms_list, coordinates_list)
            - atoms_list: 原子类型列表
            - coordinates_list: 坐标列表 [(x, y, z), ...]
    """
    xyz_path = Path(xyz_file)
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ 文件不存在: {xyz_file}")

    atoms = []
    coords = []

    with open(xyz_path, "r") as f:
        lines = f.readlines()

    # 跳过前两行（如果是标准 XYZ 格式）
    # 或者智能解析：寻找原子行
    start_idx = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        # 检查是否是原子行：以元素符号开头，后面跟数字
        parts = line.split()
        if len(parts) >= 4 and parts[0].isalpha():
            # 检查元素符号是否合理
            elem = parts[0]
            if len(elem) <= 2 and elem[0].isupper():
                start_idx = i
                break

    # 解析原子行
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()

        # 检查是否是原子行
        if len(parts) < 4:
            continue
        if not parts[0].isalpha() or len(parts[0]) > 2:
            # 可能已经到了文件末尾
            if len(atoms) > 0:
                break
            continue

        elem = parts[0]
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            atoms.append(elem)
            coords.append((x, y, z))
        except (IndexError, ValueError):
            continue

        # 如果指定了原子数，达到后停止
        if natom and len(atoms) >= natom:
            break

    # 验证
    if natom and len(atoms) != natom:
        print(f"⚠️  警告: 从 XYZ 读取了 {len(atoms)} 个原子，但预期是 {natom} 个")

    print(f"✅ 从 {xyz_path.name} 读取了 {len(atoms)} 个原子")

    return atoms, coords


def generate_data_json(
    output_file,
    nn_dir,
    xyz_file=None,
    use_xyz_coords=True,
    params=None,
):
    """
    生成 PyRAI2MD 格式的训练数据 JSON

    Args:
        output_file: 输出 JSON 文件名
        nn_dir: 神经网络力场目录
        xyz_file: XYZ 文件（可选）
        use_xyz_coords: 是否使用 XYZ 文件中的坐标
        params: 参数字典（如果不提供则从力场读取）

    Returns:
        dict: 生成的数据字典
    """
    # 读取参数
    if params is None:
        params = read_hyper_parameters(nn_dir)

    natom = params["natom"]
    nstate = params["nstate"]
    nnac = params["nnac"]
    nsoc = params["nsoc"]

    # 读取原子信息
    atoms = None
    coords = None

    if xyz_file and Path(xyz_file).exists():
        try:
            atoms, coords = read_xyz_file(xyz_file, natom)
        except Exception as e:
            print(f"⚠️  无法读取 XYZ 文件: {e}")
            atoms = None

    # 如果没有原子信息，使用默认值
    if atoms is None or len(atoms) != natom:
        print(f"⚠️  使用默认原子类型 (C)")
        atoms = ["C"] * natom

    # 创建一帧数据
    nframes = 1

    # XYZ 数据: [ [ [atom, x, y, z], ... ] ]
    xyz_data = []
    for _ in range(nframes):
        frame = []
        for i, atom in enumerate(atoms):
            if use_xyz_coords and coords and i < len(coords):
                x, y, z = coords[i]
            else:
                x, y, z = 0.0, 0.0, 0.0
            frame.append([atom, x, y, z])
        xyz_data.append(frame)

    # Energy 数据: [ [energy1, energy2, ...] ]
    energy_data = [np.zeros(nstate).tolist() for _ in range(nframes)]

    # Gradient 数据: [ [ [ [gx, gy, gz] for each atom ] for each state ] ]
    grad_data = [np.zeros((nstate, natom, 3)).tolist() for _ in range(nframes)]

    # NAC 数据: [ [ [ [gx, gy, gz] for each atom ] for each nac ] ]
    nac_data = [np.zeros((nnac, natom, 3)).tolist() for _ in range(nframes)]

    # SOC 数据
    if nsoc > 0:
        soc_data = [np.zeros(nsoc).tolist() for _ in range(nframes)]
    else:
        soc_data = [[] for _ in range(nframes)]

    # 构建数据字典
    data_dict = {
        "natom": natom,
        "nstate": nstate,
        "nnac": nnac,
        "nsoc": nsoc,
        "xyz": xyz_data,
        "atoms": atoms,
        "energy": energy_data,
        "grad": grad_data,
        "nac": nac_data,
        "soc": soc_data,
        "charge": [],
        "cell": [],
        "pbc": [],
        "info": {
            "n": nframes,
            "natom": natom,
            "nstate": nstate,
            "nnac": nnac,
            "nsoc": nsoc,
        },
    }

    # 保存文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data_dict, f, indent=2)

    # 打印摘要
    print(f"\n✅ 已创建: {output_path}")
    print(f"   原子数: {natom}")
    print(f"   态数: {nstate}")
    print(f"   NAC 对数: {nnac}")
    print(f"   SOC 数: {nsoc}")
    print(f"   帧数: {nframes}")
    print(f"   原子类型: {atoms[:5]}...{atoms[-5:]}")
    if coords and use_xyz_coords:
        print(f"   坐标: 使用 XYZ 文件中的真实坐标")
    else:
        print(f"   坐标: 使用零坐标")

    return data_dict


def find_default_files(work_dir=None):
    """
    在工作目录中查找默认文件

    Args:
        work_dir: 工作目录（默认当前目录）

    Returns:
        tuple: (nn_dir, xyz_file)
    """
    if work_dir is None:
        work_dir = Path.cwd()
    else:
        work_dir = Path(work_dir)

    # 查找 NN 目录
    nn_dir = None
    for candidate in ["NN-tps", "NN", "nn"]:
        candidate_path = work_dir / candidate
        if candidate_path.exists() and candidate_path.is_dir():
            nn_dir = candidate_path
            break

    # 查找 XYZ 文件
    xyz_file = None
    for candidate in ["tps.init.xyz", "init.xyz", "tps.xyz", "mol.xyz"]:
        candidate_path = work_dir / candidate
        if candidate_path.exists():
            xyz_file = candidate_path
            break

    # 如果没有找到，尝试任何 .xyz 文件
    if xyz_file is None:
        xyz_files = list(work_dir.glob("*.xyz"))
        if xyz_files:
            xyz_file = xyz_files[0]

    return nn_dir, xyz_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="自动从神经网络力场和分子结构生成 PyRAI2MD 训练数据 JSON 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动检测当前目录的文件并生成
  %(prog)s

  # 指定输出文件名
  %(prog)s -o data4880.json

  # 指定力场目录和 XYZ 文件
  %(prog)s --nn-dir NN-tps --xyz tps.init.xyz

  # 使用零坐标而非 XYZ 文件中的坐标
  %(prog)s --zero-coords
        """,
    )

    parser.add_argument(
        "-o",
        "--output",
        default="data.json",
        help="输出 JSON 文件名 (默认: %(default)s)",
    )

    parser.add_argument(
        "--nn-dir",
        help="神经网络力场目录 (默认: 自动检测)",
    )

    parser.add_argument(
        "--xyz",
        dest="xyz_file",
        help="XYZ 分子结构文件 (默认: 自动检测)",
    )

    parser.add_argument(
        "--zero-coords",
        action="store_false",
        dest="use_xyz_coords",
        help="使用零坐标而非 XYZ 文件中的坐标",
    )

    parser.add_argument(
        "--work-dir",
        help="工作目录 (默认: 当前目录)",
    )

    args = parser.parse_args()

    # 设置工作目录
    work_dir = args.work_dir
    if work_dir:
        work_dir = Path(work_dir)
        if not work_dir.exists():
            print(f"❌ 错误: 工作目录不存在: {work_dir}")
            return 1
    else:
        work_dir = Path.cwd()

    # 查找默认文件
    default_nn_dir, default_xyz_file = find_default_files(work_dir)

    # 使用指定的文件或默认文件
    nn_dir = args.nn_dir
    if nn_dir is None:
        nn_dir = default_nn_dir
    if nn_dir is None:
        print("❌ 错误: 找不到神经网络力场目录，请使用 --nn-dir 指定")
        return 1

    # 转换为绝对路径
    nn_dir = (work_dir / nn_dir).resolve()

    xyz_file = args.xyz_file
    if xyz_file is None:
        xyz_file = default_xyz_file
    if xyz_file is not None:
        xyz_file = (work_dir / xyz_file).resolve()

    output_file = (work_dir / args.output).resolve()

    # 打印配置
    print("=" * 60)
    print("PyRAI2MD 数据 JSON 生成器")
    print("=" * 60)
    print(f"工作目录: {work_dir}")
    print(f"力场目录: {nn_dir}")
    if xyz_file:
        print(f"XYZ 文件: {xyz_file}")
    else:
        print(f"XYZ 文件: 未找到，将使用默认原子类型")
    print(f"输出文件: {output_file}")
    print("-" * 60)

    try:
        generate_data_json(
            output_file=output_file,
            nn_dir=nn_dir,
            xyz_file=xyz_file,
            use_xyz_coords=args.use_xyz_coords,
        )
        print("\n🎉 完成！")
        return 0
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
