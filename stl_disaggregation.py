
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class STLDisaggregator:
    def __init__(self, config):
        self.config = config
        self.seasonal_patterns = {}
        self.calibration_weights = {}
        self.disaggregation_trace = {}

    def extract_seasonal_pattern(self, monthly_data, variable_name):
        monthly_ts = monthly_data.set_index('date')['value']
        
        monthly_ts_grouped = monthly_ts.groupby(monthly_ts.index.month).mean()
        normalized_weights = monthly_ts_grouped / monthly_ts_grouped.sum()
        
        stl_result = STL(
            monthly_ts,
            period=self.config.STL_PARAMS['period'],
            seasonal=self.config.STL_PARAMS['seasonal'],
            trend=self.config.STL_PARAMS['trend'],
            robust=self.config.STL_PARAMS['robust']
        ).fit()
        
        self.seasonal_patterns[variable_name] = {
            'stl_result': stl_result,
            'seasonal': stl_result.seasonal,
            'trend': stl_result.trend,
            'resid': stl_result.resid,
            'monthly_weights': normalized_weights
        }
        
        return normalized_weights

    def disaggregate_annual_to_monthly(self, annual_data, sector='total', 
                                         prior_seasonal_pattern=None, 
                                         calibration_data=None):
        if prior_seasonal_pattern is None:
            prior_seasonal_pattern = self.config.SEASONAL_WEIGHTS_PRIOR
        
        monthly_data = []
        disagg_trace = []
        
        for year in annual_data.index:
            annual_value = annual_data.loc[year, sector]
            
            weights = pd.Series([prior_seasonal_pattern[m] for m in self.config.MONTHS],
                                index=self.config.MONTHS)
            
            if calibration_data is not None:
                year_calib = calibration_data[calibration_data['date'].dt.year == year]
                weights = self._apply_calibration(weights, year_calib)
            
            weights = weights / weights.sum()
            monthly_values = annual_value * weights
            
            for month in self.config.MONTHS:
                date = pd.Timestamp(year=year, month=month, day=1)
                value = monthly_values[month]
                
                monthly_data.append({
                    'date': date,
                    'year': year,
                    'month': month,
                    'sector': sector,
                    'emission': value
                })
                
                disagg_trace.append({
                    'date': date,
                    'sector': sector,
                    'annual_total': annual_value,
                    'seasonal_weight': prior_seasonal_pattern[month],
                    'final_weight': weights[month],
                    'monthly_emission': value
                })
        
        monthly_df = pd.DataFrame(monthly_data)
        self.disaggregation_trace[sector] = pd.DataFrame(disagg_trace)
        
        return monthly_df

    def _apply_calibration(self, base_weights, calibration_data):
        calibrated_weights = base_weights.copy()
        
        if 'temperature' in calibration_data.columns:
            temp_effect = self._calculate_temperature_effect(calibration_data)
            calibrated_weights *= temp_effect
        
        if 'holiday_flag' in calibration_data.columns:
            holiday_effect = self._calculate_holiday_effect(calibration_data)
            calibrated_weights *= holiday_effect
        
        return calibrated_weights

    def _calculate_temperature_effect(self, calibration_data):
        temps = calibration_data.set_index('date')['temperature']
        monthly_temps = temps.groupby(temps.index.month).mean()
        temp_deviation = monthly_temps - monthly_temps.mean()
        effect = 1 + 0.05 * temp_deviation / temp_deviation.std()
        return effect.fillna(1)

    def _calculate_holiday_effect(self, calibration_data):
        holidays = calibration_data.set_index('date')['holiday_flag']
        monthly_holiday_ratio = holidays.groupby(holidays.index.month).mean()
        effect = 1 - 0.15 * monthly_holiday_ratio
        return effect.fillna(1)

    def combine_sectoral_data(self, sectoral_dfs):
        combined = pd.concat(sectoral_dfs, axis=0, ignore_index=True)
        return combined

    def get_traceability_info(self, date, sector):
        if sector not in self.disaggregation_trace:
            return None
        
        trace = self.disaggregation_trace[sector]
        target_trace = trace[(trace['date'] == pd.Timestamp(date)) & 
                            (trace['sector'] == sector)]
        
        if len(target_trace) == 0:
            return None
        
        return target_trace.iloc[0].to_dict()
