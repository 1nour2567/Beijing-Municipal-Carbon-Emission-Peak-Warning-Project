
# 北京市碳排放月度数据拆分与预警体系

## 项目概述

这是一个专为统计建模赛事设计的北京市碳排放月度高频预测与动态分级预警体系的数据预处理模块。该体系将北京市2005-2025年年度碳排放数据拆分为月度时序数据，并完成数据校准、建模适配、质量控制与溯源等全流程处理，为后续的XGBoost时序预测、马尔可夫状态转换风险预警、SHAP归因分析提供高质量的数据基础。

## 核心功能

### 1. STL季节分解权重拆分法
- 基于月度能源消费数据的STL分解提取季节性波动先验特征
- 结合年度碳排放总量完成月度数据拆分
- 支持全口径碳排放总量及分领域（建筑/交通/工业）数据拆分

### 2. 数据校准与极端情况处理
- 匹配月度极端气候、出行需求等外生协变量的波动特征
- 确保极端月份（如极端高温月、春节月）的拆分结果符合现实逻辑

### 3. 建模适配预处理
- 完成平稳性预处理，输出趋势项、季节项、残差项
- 异常值标记与稳健化处理
- 时间维度对齐，确保所有特征的时间粒度、起止时间完全匹配

### 4. 质量控制与溯源机制
- 完整保留拆分全流程的中间数据与权重计算过程
- 量化不同拆分方法对后续模型预测精度的影响
- 可实现单月碳排放异动的拆分环节回溯校验

## 项目结构

```
.
├── config.py                      # 项目配置文件
├── stl_disaggregation.py         # STL权重拆分核心模块
├── data_preprocessing.py          # 数据预处理与建模适配模块
├── quality_control.py             # 质量控制与溯源模块
├── sample_data_generator.py       # 示例数据生成模块
├── main.py                        # 主程序入口
├── requirements.txt               # 依赖包清单
├── data/
│   ├── raw/                       # 原始数据目录
│   ├── processed/                 # 处理后数据目录
│   └── models/                    # 模型保存目录
├── results/                       # 结果输出目录
└── logs/                          # 日志目录
```

## 安装依赖

```bash
py -m pip install -r requirements.txt
```

## 使用方法

### 快速开始

运行主程序，使用示例数据演示完整流程：

```bash
py main.py
```

### 使用真实数据

1. 准备年度碳排放数据，格式为CSV，包含以下列：
   - year: 年份（2005-2025）
   - total: 全口径碳排放总量
   - building: 建筑领域碳排放
   - transport: 交通领域碳排放
   - industry: 工业领域碳排放

2. 准备月度校准数据（可选），格式为CSV，包含：
   - date: 日期（YYYY-MM-01格式）
   - temperature: 月度平均温度
   - precipitation: 月度降水量
   - holiday_flag: 节假日标识（1为有节假日，0为无）
   - travel_index: 出行指数

3. 准备月度能源消费数据（用于提取季节性模式），格式为CSV，包含：
   - date: 日期
   - value: 能源消费量

4. 修改main.py中的数据加载部分，使用真实数据替换示例数据

5. 运行主程序：
   ```bash
   py main.py
   ```

## 输出文件

### 数据文件
- `data/processed/monthly_emission_data.csv` - 月度碳排放数据
- `data/processed/stl_decomposition_data.csv` - STL分解后的趋势、季节、残差项
- `data/processed/stationary_emission_data.csv` - 平稳化处理后的数据
- `data/processed/cleaned_emission_data.csv` - 异常值处理后的数据
- `data/processed/traceability_*.csv` - 各领域数据拆分溯源文件

### 报告文件
- `results/validation_*.json` - 数据质量验证报告
- `results/method_assessment_*.json` - 方法评估报告
- `logs/traceability_*.json` - 溯源日志

## 核心模块说明

### STLDisaggregator (stl_disaggregation.py)
主要类：`STLDisaggregator`

核心方法：
- `extract_seasonal_pattern()` - 从月度数据提取季节性模式
- `disaggregate_annual_to_monthly()` - 年度数据到月度数据拆分
- `get_traceability_info()` - 获取指定日期的溯源信息

### DataPreprocessor (data_preprocessing.py)
主要类：`DataPreprocessor`

核心方法：
- `decompose_with_stl()` - STL分解
- `test_stationarity()` - 平稳性检验
- `make_stationary()` - 平稳化处理
- `detect_outliers()` - 异常值检测
- `handle_outliers()` - 异常值处理

### QualityController (quality_control.py)
主要类：`QualityController`

核心方法：
- `validate_disaggregated_data()` - 验证拆分数据质量
- `generate_method_assessment_report()` - 生成方法评估报告
- `log_traceability()` - 记录溯源日志

## 适配后续建模

处理后的数据可直接用于：
1. **XGBoost时序预测模型** - 使用`monthly_emission_data.csv`或`cleaned_emission_data.csv`
2. **马尔可夫状态转换模型** - 使用`stl_decomposition_data.csv`中的趋势项
3. **SHAP归因分析** - 使用完整的特征面板数据
4. **CausalImpact政策评估** - 结合外生协变量进行因果推断

## 赛事论文创新点表述建议

### 数据拆分方法创新
> 采用「基于STL季节分解先验的权重拆分法」，通过北京市月度用电量、能源消费数据的STL分解提取季节性波动特征，结合极端气候、出行需求等外生协变量完成校准，有效解决了年度碳排放数据月度拆分的季节性偏差问题，为高频时序预测提供了高质量的数据基础。

### 质量控制与溯源机制
> 建立了全流程数据溯源机制，完整保留拆分权重、校准因子等中间变量，可实现单月碳排放异动的回溯校验；同时通过多方法对比验证，量化了不同拆分方案对预测精度的影响，确保了数据预处理的科学性与稳健性。

## 注意事项

1. 本体系包含示例数据生成器，便于演示和测试，实际使用时请替换为真实数据
2. 可通过修改`config.py`中的参数调整STL分解、异常值检测等算法的行为
3. 所有输出文件均包含时间戳，便于版本管理和结果对比

## 技术栈

- Python 3.13+
- pandas - 数据处理
- numpy - 数值计算
- statsmodels - STL分解、平稳性检验
- scikit-learn - 数据预处理
- scipy - 统计分析

## 许可证

本项目为统计建模赛事专用工具，仅供学习和科研使用。
