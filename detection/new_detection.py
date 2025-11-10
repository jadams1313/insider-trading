import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
import os 
import math
import scipy.spatial.distance import euclidean
class AnomalyDetection:

    def __init__(self):
        self.window = 50
        self.overlap = 10
        self.actual_data = None
        self.predicited_data_window_based = None
        self.predicted_data_day_based = None
        self.predicted_data_historical_based = None
        self.pattern_desc_arr = None
        self.pattern_mat = None
        self.result = []

    def import_data(self):
        with open('r'
    def generate_day_list(self, start_date, end_date):
    """Generate a list of days from start_date to end_date inclusive."""
        days = []
        current = start_date
        while current <= end_date:
            days.append(current)
            current += timedelta(days=1)
        return days

        
    def analyze(self):
        """Anomaly Detection Algorithm"""
        companies = self.load_data()


        for c in companies:
            for m in c.getTradingMethods():
                for c_i in m.getTradingPortfolios():
                    time_window = c_i.getWindow()
                    start, end = time_window 
                    days_in_window = self.generate_day_list(start, end)

                    event_days = c_i.getUpcomingEvents()
                    event_day_indexes = [days_in_window.index(d) + 1 for d in event_days if d in days_in_window]
                    for day_index, day in enumerate(days_in_window, start=1):
                        time_series = c_i.getTimeSeries(day)
                        baseline = self.getBaseLine(day_index)
                        p = self.getPattern(day_index)
                        distance, path = fastdtw(x,y dist=euclidean)
        return distance, path
                    
                        
    def main():
        print("hi")
if __name__ == "__main__":
    main()

