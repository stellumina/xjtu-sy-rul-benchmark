# xjtu-sy-rul-benchmark

用于滚动轴承剩余寿命（RUL）预测的可复现实验基准框架。当前版本建立工程骨架和无数据泄漏实验协议，不包含模型实现。

## 项目目标

统一 XJTU-SY 数据的读取、特征、标签、轴承级划分和指标接口，为后续研究代码提供可审计、可复现的基础。

## XJTU-SY 数据集

XJTU-SY 是滚动轴承全寿命振动数据集，包含 35 Hz/12 kN、37.5 Hz/11 kN、40 Hz/10 kN 三类工况，每个工况下包含多个轴承及按时间采集的 CSV 文件。

## 数据下载

优先来源为 [WangBiaoXJTU/xjtu-sy-bearing-datasets](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets)。运行 `python scripts/download.py --help` 查看选项。由于当前任务没有确认 biaowang.tech 的实际下载 URL，代码保留 TODO，不编造该地址；完整下载由用户显式指定 URL 后执行。

## 数据目录

```text
data/raw/XJTU-SY_Bearing_Datasets/
├── 35Hz12kN/Bearing1_1/1.csv
├── 37.5Hz11kN/Bearing2_3/1.csv
└── 40Hz10kN/Bearing3_5/1.csv
```

## 运行 pipeline

```bash
python -m pytest
python scripts/download.py --bearings Bearing1_1 Bearing2_3
```

后续实现数据处理和实验脚本后，应从仓库根目录运行 pipeline，并使用 `PROTOCOL.md` 中规定的轴承级划分和训练集拟合流程。

## 复现实验

保存依赖版本、数据版本或校验值、划分清单、随机种子、配置和测试预测，再运行相同配置生成指标。任何实验报告都必须说明训练/验证/测试轴承列表及协议版本。
