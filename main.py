
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import config
from stl_disaggregation import STLDisaggregator
from data_preprocessing import DataPreprocessor
from quality_control import QualityController
from sample_data_generator import SampleDataGenerator


def main():
    print("=" * 80)
    print("北京市碳排放月度数据拆分与预警体系 - 数据处理流程")
    print("=" * 80)
    
    print("\n[1/7] 生成示例数据...")
    data_gen = SampleDataGenerator(config)
    sample_data = data_gen.save_sample_data()
    print("   ✓ 示例数据已生成并保存到 data/raw/")
    
    print("\n[2/7] 初始化处理模块...")
    disaggregator = STLDisaggregator(config)
    preprocessor = DataPreprocessor(config)
    quality_controller = QualityController(config)
    print("   ✓ 所有处理模块已初始化")
    
    annual_data = sample_data['annual_emission']
    calibration_data = sample_data['calibration']
    energy_data = sample_data['energy']
    
    calibration_data['date'] = pd.to_datetime(calibration_data['date'])
    energy_data['date'] = pd.to_datetime(energy_data['date'])
    
    print("\n[3/7] 从月度能源数据提取季节性模式...")
    seasonal_pattern = disaggregator.extract_seasonal_pattern(energy_data, 'energy_consumption')
    print("   ✓ 季节性模式已提取")
    print(f"   月度权重: {seasonal_pattern.to_dict()}")
    
    print("\n[4/7] 执行年度数据到月度数据的STL权重拆分...")
    sectoral_dfs = []
    
    for sector in config.SECTORS:
        print(f"   处理领域: {sector}")
        sector_df = disaggregator.disaggregate_annual_to_monthly(
            annual_data,
            sector=sector,
            prior_seasonal_pattern=seasonal_pattern.to_dict(),
            calibration_data=calibration_data
        )
        sectoral_dfs.append(sector_df)
    
    monthly_emission = disaggregator.combine_sectoral_data(sectoral_dfs)
    print("   ✓ 所有领域月度拆分完成")
    
    print("\n[5/7] 数据预处理与建模适配...")
    print("   a. STL分解与趋势-季节-残差项提取...")
    stl_decomp_df = preprocessor.get_stl_decomposition_df(monthly_emission)
    print("      ✓ STL分解完成")
    
    print("   b. 平稳性检验与处理...")
    stationary_df = preprocessor.make_stationary(monthly_emission)
    print("      ✓ 平稳性检验完成")
    
    print("   c. 异常值检测与处理...")
    cleaned_df = preprocessor.handle_outliers(monthly_emission, method='winsorize')
    outlier_summary = preprocessor.outlier_flags.groupby('sector')['is_outlier'].sum()
    print(f"      ✓ 异常值处理完成，检测到 {outlier_summary.sum()} 个异常值")
    
    print("\n[6/7] 质量控制与溯源...")
    print("   a. 拆分数据验证...")
    validation_report = quality_controller.validate_disaggregated_data(monthly_emission, annual_data)
    print(f"      ✓ 验证完成，总体得分: {validation_report['overall_score']:.2%}")
    
    print("   b. 生成方法评估报告...")
    assessment_report = quality_controller.generate_method_assessment_report(disaggregator, preprocessor)
    print("      ✓ 评估报告已生成")
    
    print("   c. 溯源日志记录...")
    quality_controller.log_traceability('data_generation', {'source': 'sample_generator'})
    quality_controller.log_traceability('stl_disaggregation', {'sectors': config.SECTORS})
    quality_controller.log_traceability('preprocessing', {'outlier_method': 'winsorize'})
    print("      ✓ 溯源日志已记录")
    
    print("\n[7/7] 保存处理结果...")
    monthly_emission.to_csv(config.PROCESSED_DATA_DIR / 'monthly_emission_data.csv', index=False)
    stl_decomp_df.to_csv(config.PROCESSED_DATA_DIR / 'stl_decomposition_data.csv', index=False)
    stationary_df.to_csv(config.PROCESSED_DATA_DIR / 'stationary_emission_data.csv', index=False)
    cleaned_df.to_csv(config.PROCESSED_DATA_DIR / 'cleaned_emission_data.csv', index=False)
    
    quality_controller.save_quality_report('validation')
    quality_controller.save_quality_report('method_assessment')
    quality_controller.save_traceability_logs()
    
    for sector in config.SECTORS:
        if sector in disaggregator.disaggregation_trace:
            trace_path = config.PROCESSED_DATA_DIR / f'traceability_{sector}.csv'
            disaggregator.disaggregation_trace[sector].to_csv(trace_path, index=False)
    
    print("   ✓ 所有结果已保存到 data/processed/ 和 results/")
    
    print("\n" + "=" * 80)
    print("数据处理流程完成！")
    print("=" * 80)
    
    print("\n【关键输出文件】")
    print("1. 月度碳排放数据: data/processed/monthly_emission_data.csv")
    print("2. STL分解数据: data/processed/stl_decomposition_data.csv")
    print("3. 平稳化数据: data/processed/stationary_emission_data.csv")
    print("4. 异常值处理数据: data/processed/cleaned_emission_data.csv")
    print("5. 质量验证报告: results/validation_*.json")
    print("6. 方法评估报告: results/method_assessment_*.json")
    print("7. 溯源日志: logs/traceability_*.json")
    
    print("\n【溯源示例】")
    sample_date = monthly_emission.iloc[0]['date']
    sample_sector = monthly_emission.iloc[0]['sector']
    trace_info = disaggregator.get_traceability_info(sample_date, sample_sector)
    if trace_info:
        print(f"日期: {trace_info['date'].strftime('%Y-%m')}")
        print(f"领域: {trace_info['sector']}")
        print(f"年度总量: {trace_info['annual_total']:.2f}")
        print(f"月度排放: {trace_info['monthly_emission']:.2f}")
        print(f"最终权重: {trace_info['final_weight']:.4f}")
    
    print("\n【下一步】")
    print("- 替换示例数据为真实年度碳排放数据")
    print("- 调整 config.py 中的参数以优化拆分效果")
    print("- 使用处理后的数据进行XGBoost预测、SHAP归因等后续建模")


if __name__ == "__main__":
    main()
