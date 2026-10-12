# xjtu-sy-rul-benchmark

本项目为 XJTU-SY 滚动轴承剩余寿命（RUL）预测建立可复现实验基准。实验的数据切分、标签和评估必须遵守 [PROTOCOL.md](PROTOCOL.md)。

## 特征数据

`data/features/` 包含 15 个轴承的逐文件特征 CSV 和 `meta.csv`。每个原始数据文件代表 1 分钟；每行特征对应一个文件。特征表的实际列名以 CSV 表头为准，当前为：

`file_idx, rms_h, rms_v, kurt_h, kurt_v, peak_h, peak_v, crest_h, crest_v`

`meta.csv` 记录 `bearing, condition, n_files`。`bad_files.csv` 是本地提取过程的坏文件清单。原始波形约 7 GB，保存在用户本地，不进入仓库。

## 项目结构与后续运行

`src/` 放置数据、特征、标签、划分、指标与训练代码；`scripts/` 放置运行和误差分析入口；`tests/` 存放合成数据测试；`reports/` 与 `results/` 分别保存报告和实验输出。

先安装依赖并运行测试：

```bash
python -m pip install -r requirements.txt
python -m pytest
```

后续实验按照 [PROTOCOL.md](PROTOCOL.md) 固定轴承级切分和评估规则，运行相应脚本，并在 `results/results.csv` 保存七列结果。当前任务只校准协议、数据目录和工程骨架，不执行模型训练。
