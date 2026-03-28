
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SampleDataGenerator:
    def __init__(self, config):
        self.config = config
        np.random.seed(42)

    def generate_annual_emission_data(self):
        years = self.config.YEARS
        
        data = {
            'year': years,
            'total': np.zeros(len(years)),
            'building': np.zeros(len(years)),
            'transport': np.zeros(len(years)),
            'industry': np.zeros(len(years))
        }
        
        base_total = 150.0
        building_ratio = 0.35
        transport_ratio = 0.25
        industry_ratio = 0.40
        
        for i, year in enumerate(years):
            growth_factor = 1 + 0.03 * np.sin((year - 2005) / 10)
            annual_total = base_total * (1 + 0.02 * i) * growth_factor
            
            data['total'][i] = annual_total
            data['building'][i] = annual_total * building_ratio * (1 + 0.01 * np.sin(year / 5))
            data['transport'][i] = annual_total * transport_ratio * (1 + 0.02 * np.cos(year / 4))
            data['industry'][i] = annual_total * industry_ratio * (1 - 0.005 * i)
        
        df = pd.DataFrame(data).set_index('year')
        return df

    def generate_calibration_data(self):
        dates = []
        for year in self.config.YEARS:
            for month in self.config.MONTHS:
                dates.append(pd.Timestamp(year=year, month=month, day=1))
        
        data = {
            'date': dates,
            'temperature': np.zeros(len(dates)),
            'precipitation': np.zeros(len(dates)),
            'holiday_flag': np.zeros(len(dates), dtype=int),
            'travel_index': np.zeros(len(dates))
        }
        
        for i, date in enumerate(dates):
            month = date.month
            year = date.year
            
            temp_seasonal = -10 * np.cos(2 * np.pi * (month - 1) / 12)
            temp_trend = 0.02 * (year - 2005)
            data['temperature'][i] = 15 + temp_seasonal + temp_trend + np.random.normal(0, 2)
            
            data['precipitation'][i] = 50 + 30 * np.sin(2 * np.pi * (month - 7) / 12) + np.random.normal(0, 10)
            
            if month in [1, 2, 10]:
                data['holiday_flag'][i] = 1
            
            travel_base = 100
            travel_seasonal = 20 * np.sin(2 * np.pi * (month - 4) / 12)
            travel_trend = 2 * (year - 2005)
            data['travel_index'][i] = travel_base + travel_seasonal + travel_trend + np.random.normal(0, 5)
        
        df = pd.DataFrame(data)
        return df

    def generate_monthly_energy_data(self):
        dates = []
        for year in self.config.YEARS:
            for month in self.config.MONTHS:
                dates.append(pd.Timestamp(year=year, month=month, day=1))
        
        data = {
            'date': dates,
            'value': np.zeros(len(dates))
        }
        
        for i, date in enumerate(dates):
            month = date.month
            year = date.year
            
            seasonal_effect = np.array([
                self.config.SEASONAL_WEIGHTS_PRIOR[m] for m in range(1, 13)
            ])[month - 1]
            trend = 0.05 * (year - 2005)
            value = 1000 * (1 + trend) * seasonal_effect * 12 * (1 + np.random.normal(0, 0.03))
            
            data['value'][i] = value
        
        df = pd.DataFrame(data)
        return df

    def save_sample_data(self):
        annual_emission = self.generate_annual_emission_data()
        calibration = self.generate_calibration_data()
        energy = self.generate_monthly_energy_data()
        
        annual_emission.to_csv(self.config.RAW_DATA_DIR / 'annual_emission_data.csv')
        calibration.to_csv(self.config.RAW_DATA_DIR / 'calibration_data.csv', index=False)
        energy.to_csv(self.config.RAW_DATA_DIR / 'monthly_energy_data.csv', index=False)
        
        return {
            'annual_emission': annual_emission,
            'calibration': calibration,
            'energy': energy
        }
