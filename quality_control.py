
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        return super().default(obj)


class QualityController:
    def __init__(self, config):
        self.config = config
        self.quality_reports = {}
        self.traceability_logs = []

    def validate_disaggregated_data(self, monthly_df, annual_df):
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'overall_score': 0.0
        }
        
        checks = []
        
        for sector in self.config.SECTORS:
            if sector in annual_df.columns:
                sector_monthly = monthly_df[monthly_df['sector'] == sector]
                
                for year in self.config.YEARS:
                    if year in annual_df.index:
                        monthly_sum = sector_monthly[sector_monthly['year'] == year]['emission'].sum()
                        annual_value = annual_df.loc[year, sector]
                        relative_error = abs(monthly_sum - annual_value) / annual_value
                        
                        check_name = f'annual_sum_match_{sector}_{year}'
                        validation_report['checks'][check_name] = {
                            'passed': relative_error < 0.01,
                            'error': relative_error,
                            'monthly_sum': monthly_sum,
                            'annual_value': annual_value
                        }
                        checks.append(relative_error < 0.01)
        
        validation_report['checks']['no_missing_values'] = {
            'passed': monthly_df['emission'].notna().all(),
            'missing_count': monthly_df['emission'].isna().sum()
        }
        checks.append(monthly_df['emission'].notna().all())
        
        validation_report['checks']['positive_values'] = {
            'passed': (monthly_df['emission'] > 0).all(),
            'negative_count': (monthly_df['emission'] <= 0).sum()
        }
        checks.append((monthly_df['emission'] > 0).all())
        
        validation_report['overall_score'] = sum(checks) / len(checks) if checks else 0.0
        
        self.quality_reports['validation'] = validation_report
        return validation_report

    def compare_disaggregation_methods(self, monthly_dfs, method_names):
        comparison_report = {
            'timestamp': datetime.now().isoformat(),
            'methods': method_names,
            'metrics': {}
        }
        
        metrics = ['mean', 'std', 'min', 'max', 'iqr']
        
        for sector in self.config.SECTORS:
            comparison_report['metrics'][sector] = {}
            
            for metric in metrics:
                comparison_report['metrics'][sector][metric] = []
            
            for i, df in enumerate(monthly_dfs):
                sector_data = df[df['sector'] == sector]['emission']
                
                comparison_report['metrics'][sector]['mean'].append(sector_data.mean())
                comparison_report['metrics'][sector]['std'].append(sector_data.std())
                comparison_report['metrics'][sector]['min'].append(sector_data.min())
                comparison_report['metrics'][sector]['max'].append(sector_data.max())
                comparison_report['metrics'][sector]['iqr'].append(
                    sector_data.quantile(0.75) - sector_data.quantile(0.25)
                )
        
        self.quality_reports['method_comparison'] = comparison_report
        return comparison_report

    def log_traceability(self, operation, details):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'details': details
        }
        self.traceability_logs.append(log_entry)
        return log_entry

    def save_traceability_logs(self, filepath=None):
        if filepath is None:
            filepath = self.config.LOGS_DIR / f'traceability_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.traceability_logs, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        return filepath

    def save_quality_report(self, report_name, filepath=None):
        if filepath is None:
            filepath = self.config.RESULTS_DIR / f'{report_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        if report_name in self.quality_reports:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.quality_reports[report_name], f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        return filepath

    def generate_method_assessment_report(self, disaggregator, preprocessor):
        report = {
            'timestamp': datetime.now().isoformat(),
            'disaggregation_trace': {},
            'stl_decomposition_summary': {},
            'stationarity_results': {},
            'outlier_summary': {}
        }
        
        for sector in self.config.SECTORS:
            if sector in disaggregator.disaggregation_trace:
                trace_df = disaggregator.disaggregation_trace[sector]
                report['disaggregation_trace'][sector] = {
                    'weight_variation': trace_df['final_weight'].std(),
                    'annual_match': (trace_df.groupby(trace_df['date'].dt.year)
                                     ['monthly_emission'].sum() - 
                                     trace_df.groupby(trace_df['date'].dt.year)
                                     ['annual_total'].first()).abs().mean()
                }
        
        for sector, stl_data in preprocessor.stl_results.items():
            report['stl_decomposition_summary'][sector] = {
                'trend_variance': stl_data['trend'].var(),
                'seasonal_variance': stl_data['seasonal'].var(),
                'residual_variance': stl_data['resid'].var(),
                'explained_variance_ratio': 1 - stl_data['resid'].var() / (
                    stl_data['trend'].var() + stl_data['seasonal'].var() + stl_data['resid'].var()
                )
            }
        
        report['stationarity_results'] = preprocessor.stationarity_tests
        
        if isinstance(preprocessor.outlier_flags, pd.DataFrame):
            for sector in preprocessor.outlier_flags['sector'].unique():
                sector_outliers = preprocessor.outlier_flags[preprocessor.outlier_flags['sector'] == sector]
                report['outlier_summary'][sector] = {
                    'total_observations': len(sector_outliers),
                    'outlier_count': sector_outliers['is_outlier'].sum(),
                    'outlier_ratio': sector_outliers['is_outlier'].mean()
                }
        
        self.quality_reports['method_assessment'] = report
        return report
