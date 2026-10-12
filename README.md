# xjtu-sy-rul-benchmark

用于滚动轴承剩余寿命（RUL）预测的可复现实验基准框架。仓库只保存特征级数据和实验代码；约 7 GB 的原始振动波形留在用户本地，不进入仓库。

## 项目目标

统一 XJTU-SY 数据的本地特征提取、标签、轴承级划分和指标接口，建立可审计、无数据泄漏的实验流程。最高规范见 [PROTOCOL.md](PROTOCOL.md)。

## XJTU-SY 数据集

XJTU-SY 是滚动轴承全寿命振动数据集，包含 35 Hz/12 kN、37.5 Hz/11 kN、40 Hz/10 kN 三类工况。每个轴承由按分钟采集的 CSV 文件组成，每个文件包含水平和垂直两通道波形。

## 本地特征提取

在 Windows 本地执行，原始目录默认探测 `D:/RUL/XTJU`、`D:/RUL/XJTU` 及其最多两层内名称含 `XJTU` 的目录，也可显式指定：

```bash
python scripts/extract_features.py --raw-dir D:/RUL/XTJU
```

输出为 `data/features/Bearing{condition}_{index}.csv`、`meta.csv` 和坏文件清单 `bad_files.csv`。云端实验直接消费 `data/features/`。原始波形不会被复制到仓库。

## 数据目录

```text
data/features/
├── Bearing1_1.csv
├── Bearing2_3.csv
├── Bearing3_5.csv
└── meta.csv
```

## 运行与复现

```bash
python -m pytest
python scripts/extract_features.py --help
```

实验必须保存数据版本或校验值、轴承级划分、5 个随机种子（0–4）、配置和预测结果，并追加写入 `results/results.csv`。所有处理和评估遵守 `PROTOCOL.md`。
