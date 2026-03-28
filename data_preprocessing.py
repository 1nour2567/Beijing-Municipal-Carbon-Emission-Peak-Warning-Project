
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.stl_results = {}
        self.outlier_flags = {}
        self.stationarity_tests = {}

    def decompose_with_stl(self, df, value_col='emission'):
        sector_data = df.pivot(index='date', columns='sector', values=value_col)
        
        for sector in sector_data.columns:
            ts = sector_data[sector].dropna()
            stl_result = STL(
                ts,
                period=self.config.STL_PARAMS['period'],
                seasonal=self.config.STL_PARAMS['seasonal'],
                trend=self.config.STL_PARAMS['trend'],
                robust=self.config.STL_PARAMS['robust']
            ).fit()
            
            self.stl_results[sector] = {
                'trend': stl_result.trend,
                'seasonal': stl_result.seasonal,
                'resid': stl_result.resid,
                'stl_object': stl_result
            }
        
        return self.stl_results

    def test_stationarity(self, series, name='series'):
        adf_result = adfuller(series.dropna())
        kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
        
        stationarity_info = {
            'ADF_statistic': adf_result[0],
            'ADF_pvalue': adf_result[1],
            'ADF_critical_values': adf_result[4],
            'KPSS_statistic': kpss_result[0],
            'KPSS_pvalue': kpss_result[1],
            'KPSS_critical_values': kpss_result[3],
            'is_stationary_adf': adf_result[1] < 0.05,
            'is_stationary_kpss': kpss_result[1] >= 0.05
        }
        
        self.stationarity_tests[name] = stationarity_info
        return stationarity_info

    def make_stationary(self, df, value_col='emission'):
        df_stationary = df.copy()
        sector_data = df_stationary.pivot(index='date', columns='sector', values=value_col)
        
        for sector in sector_data.columns:
            ts = sector_data[sector]
            
            test_result = self.test_stationarity(ts, f'{sector}_original')
            
            if not test_result['is_stationary_adf']:
                ts_diff = ts.diff().dropna()
                sector_data[sector] = ts_diff
                self.test_stationarity(ts_diff, f'{sector}_diff1')
        
        df_stationary = sector_data.stack().reset_index()
        df_stationary.columns = ['date', 'sector', f'{value_col}_stationary']
        
        return df_stationary

    def detect_outliers(self, df, value_col='emission', method='iqr'):
        df_outliers = df.copy()
        outlier_flags = []
        
        for sector in df['sector'].unique():
            sector_df = df[df['sector'] == sector]
            
            if method == 'iqr':
                Q1 = sector_df[value_col].quantile(0.25)
                Q3 = sector_df[value_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                is_outlier = (sector_df[value_col] < lower_bound) | (sector_df[value_col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(sector_df[value_col].dropna()))
                is_outlier = pd.Series(z_scores > 3, index=sector_df.index[sector_df[value_col].notna()])
                
            elif method == 'stl':
                if sector in self.stl_results:
                    resid = self.stl_results[sector]['resid']
                    mad = stats.median_abs_deviation(resid.dropna())
                    threshold = 3 * mad
                    is_outlier = np.abs(resid) > threshold
                    is_outlier = is_outlier.reindex(sector_df.index)
                
            for idx, row in sector_df.iterrows():
                flag = is_outlier.get(idx, False) if 'is_outlier' in locals() else False
                outlier_flags.append({
                    'date': row['date'],
                    'sector': sector,
                    'is_outlier': flag,
                    'value': row[value_col]
                })
        
        self.outlier_flags = pd.DataFrame(outlier_flags)
        return self.outlier_flags

    def handle_outliers(self, df, value_col='emission', method='winsorize'):
        df_cleaned = df.copy()
        outlier_flags = self.detect_outliers(df, value_col)
        
        for sector in df['sector'].unique():
            sector_mask = df_cleaned['sector'] == sector
            outlier_mask = outlier_flags[(outlier_flags['sector'] == sector) & 
                                         (outlier_flags['is_outlier'])]['date'].values
            
            if method == 'winsorize':
                values = df_cleaned.loc[sector_mask, value_col]
                lower = values.quantile(0.05)
                upper = values.quantile(0.95)
                df_cleaned.loc[sector_mask, value_col] = np.clip(values, lower, upper)
            
            elif method == 'interpolate':
                df_cleaned.loc[sector_mask & df_cleaned['date'].isin(outlier_mask), value_col] = np.nan
                df_cleaned.loc[sector_mask, value_col] = df_cleaned.loc[sector_mask, value_col].interpolate(method='linear')
            
            elif method == 'stl_adjust':
                if sector in self.stl_results:
                    trend = self.stl_results[sector]['trend']
                    seasonal = self.stl_results[sector]['seasonal']
                    df_cleaned.loc[sector_mask, value_col] = trend + seasonal
        
        df_cleaned['is_outlier'] = outlier_flags.set_index(['date', 'sector']).loc[
            pd.MultiIndex.from_frame(df_cleaned[['date', 'sector']]), 'is_outlier'
        ].values
        
        return df_cleaned

    def get_stl_decomposition_df(self, df, value_col='emission'):
        self.decompose_with_stl(df, value_col)
        
        decomp_dfs = []
        for sector, stl_data in self.stl_results.items():
            decomp_df = pd.DataFrame({
                'date': stl_data['trend'].index,
                'sector': sector,
                f'{value_col}_trend': stl_data['trend'],
                f'{value_col}_seasonal': stl_data['seasonal'],
                f'{value_col}_resid': stl_data['resid'],
                f'{value_col}_original': stl_data['trend'] + stl_data['seasonal'] + stl_data['resid']
            })
            decomp_dfs.append(decomp_df)
        
        return pd.concat(decomp_dfs, axis=0, ignore_index=True)
