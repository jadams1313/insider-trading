
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.spatial.distance import euclidean
import os
import math
from fastdtw import fastdtw 

class TimeSeriesSimilarity:
    def __init__(self):
        # Global variables equivalent
        self.window_size = 50
        self.overlap = 10
        self.actual_data = None
        self.predicted_data_window_based = None
        self.predicted_data_day_based = None
        self.predicted_data_historical_based = None
        self.pattern_desc_arr = None
        self.pattern_mat = None
        self.result = []
        
        # Configuration
        self.num_method = 3
        self.num_pattern = 11

        self.dtw_threshold = 9.0
    
    def handle_pattern_data(self):
        with open('data/pattern.csv', 'r') as f:
            lines = f.readlines()

        data = []
        max_cols = 0

        for line in lines:
            row = [x.strip() for x in line.strip().split(',') if x.strip()]
            max_cols = max(max_cols, len(row))

        for line in lines:
            row = [x.strip() for x in line.strip().split(',')]
            row_floats = []
            for item in row:
                if item.strip() == '':
                    row_floats.append(0.0)
                else:
                    try:
                        row_floats.append(float(item))
                    except ValueError:
                        row_floats.append(0.0)
            
            # Pad with zeros to reach max_cols
            while len(row_floats) < max_cols:
                row_floats.append(0.0)
            
            data.append(row_floats)

        # Create DataFrame
        df = pd.DataFrame(data)
        return df.values
        
    def load_data(self):
        """Load data from CSV files"""
        try:
            self.actual_data = pd.read_csv('data/amsc_full_window_act.csv', header=None).values.flatten()
            self.predicted_data_window_based = pd.read_csv('data/amsc_full_window_pred.csv', header=None).values.flatten()
            self.predicted_data_day_based = pd.read_csv('data/amsc_full_point_pred.csv', header=None).values.flatten()
            self.predicted_data_historical_based = pd.read_csv('data/amsc_full_sequence_pred.csv', header=None).values.flatten()
            self.pattern_mat = self.handle_pattern_data()
            self.pattern_desc_arr = pd.read_csv('data/pattern_sizes.csv', header=None).values.flatten()
            
            print("Data loaded successfully")
            print(f"Actual data shape: {self.actual_data.shape}")
            print(f"Pattern matrix shape: {self.pattern_mat.shape}")
            
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            print("Please ensure all data CSV files are in the 'data' directory")
    
    def calculate_actual_day(self, window, day):
        return self.window_size + (window * self.window_size) + day

    def DTW(self, method, window, pattern, day):
        signal1 = []
        signal2 = []
        predicted_data = []

        if method == 1:
            predicted_data = self.predicted_data_window_based
            method_name = "Window Based Forecasting"
        elif method == 2:
            predicted_data = self.predicted_data_day_based
            method_name = "Day Ahead Forecasting"
        elif method == 3:
            predicted_data = self.predicted_data_historical_based
            method_name = "Entire History Based Forecasting"

        # Extract signals for comparison
        if pattern == 0:
            # Compare entire window
            signal1_start = (window - 1) * self.window_size
            signal1_end = signal1_start + self.window_size
            if len(predicted_data) < signal1_end:
                signal1_end = len(predicted_data)
            signal2_start = signal1_start
            signal2_end = signal1_end
            signal1 = predicted_data[signal1_start:signal1_end]
            signal2 = self.actual_data[signal2_start:signal2_end]
        elif pattern > 0:
            # Compare against specific insider trading pattern
            signal1_start = (window - 1) * self.window_size + day - 1
            signal1_end = signal1_start + int(self.pattern_desc_arr[pattern - 1])
            if len(predicted_data) < signal1_end:
                signal1_end = len(predicted_data)
            signal1 = predicted_data[signal1_start:signal1_end]
            pattern_length = int(self.pattern_desc_arr[pattern - 1])
            signal2 = self.pattern_mat[pattern - 1, :pattern_length]

        if len(signal1) > 0 and len(signal2) > 0:
            signal1 = np.array(signal1, dtype=np.float64).flatten()
            signal2 = np.array(signal2, dtype=np.float64).flatten()

            signal1_normalized = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-8)
            signal2_normalized = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-8)
            

            # Calculate DTW distance using Euclidean distance metric
            # DTW is non-negative, with 0 = identical sequences
            distance, path = fastdtw(signal1_normalized, signal2_normalized, dist=lambda x, y: abs(x - y))

            # Store raw DTW distance (not normalized as in paper)
            dtw_distance = distance

            # Check if this matches the threshold for anomaly detection
            if dtw_distance <= self.dtw_threshold and pattern > 0:
                actual_day_num = self.calculate_actual_day(window, day)
                
                self.result.append({
                    'method': method,
                    'method_name': method_name,
                    'window': window,
                    'pattern': pattern,
                    'day_in_window': day,
                    'actual_day': actual_day_num,
                    'dtw_distance': dtw_distance
                })

                # Visualization matching paper's figures
                self.plot_dtw_alignment(signal1_normalized, signal2_normalized, path, method_name, 
                                       window, pattern, day, dtw_distance)
            
            return dtw_distance
        
        return float('inf')
    def plot_dtw_alignment(self, signal1, signal2, path, method_name, 
                          window, pattern, day, dtw_distance):
        
        plt.figure(figsize=(12, 6))
        x1 = np.arange(len(signal1))
        x2 = np.arange(len(signal2))
        
        plt.plot(x1, signal1, 'b-', label='Predicted Signal', linewidth=2, alpha=0.7)
        plt.plot(x2, signal2, 'r-', label='Insider Trading Pattern', linewidth=2, alpha=0.7)
        
        # Draw alignment lines (sample every few points to avoid clutter)
        step = max(1, len(path) // 20)
        for i, j in path[::step]:
            plt.plot([i, j], [signal1[i], signal2[j]], 'k-', alpha=0.2, linewidth=0.5)
        
        plt.xlabel('Time (Days)', fontsize=12)
        plt.ylabel('Stock Volume', fontsize=12)
        plt.title(f'DTW Alignment - {method_name}\n' + 
                 f'Window: {window}, Pattern: {pattern}, Day: {day} | ' +
                 f'DTW Distance: {dtw_distance:.3f}',
                 fontsize=13, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        os.makedirs('output/figures', exist_ok=True)
        filename = f'output/figures/dtw_m{method_name}_w{window}_p{pattern}_d{day}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    def find_minimum_dtw_per_method(self):
        """
        Find the minimum DTW distance for each forecasting method,
        similar to the paper's reporting of minimum DTW scores.
        """
        if not self.result:
            print("No anomalies detected.")
            return
        
        df_results = pd.DataFrame(self.result)
        
        print("\n" + "="*70)
        print("MINIMUM DTW DISTANCES PER FORECASTING METHOD")
        print("="*70)
        
        for method in [1, 2, 3]:
            method_results = df_results[df_results['method'] == method]
            
            if len(method_results) > 0:
                min_idx = method_results['dtw_distance'].idxmin()
                min_result = method_results.loc[min_idx]
                
                print(f"\n{min_result['method_name']}:")
                print(f"  Minimum DTW Distance: {min_result['dtw_distance']:.3f}")
                print(f"  Window: {min_result['window']}, Pattern: {min_result['pattern']}, " +
                      f"Day in Window: {min_result['day_in_window']}")
                print(f"  Actual Day in Time Series: {min_result['actual_day']}")
        
        print("\n" + "="*70)
    def generate_hit_count_report(self):
        if not self.result:
            return
        
        df_results = pd.DataFrame(self.result)
        
        print("\n" + "="*70)
        print("HIT COUNTS PER PATTERN (DTW Distance < {})".format(self.dtw_threshold))
        print("="*70)
        
        # Count hits per pattern across all methods
        pattern_hits = df_results.groupby('pattern').agg({
            'method': 'count',
            'dtw_distance': 'mean'
        }).rename(columns={'method': 'total_hits', 'dtw_distance': 'avg_dtw'})
        
        # Breakdown by method - ensure all 3 methods are represented
        method_breakdown = df_results.groupby(['pattern', 'method']).size().unstack(fill_value=0)
        
        # Add missing method columns if they don't exist
        for method_num, method_name in [(1, 'Window Based'), (2, 'Day Ahead'), (3, 'Entire History')]:
            if method_num not in method_breakdown.columns:
                method_breakdown[method_num] = 0
        
        # Rename columns and ensure correct order
        method_breakdown = method_breakdown[[1, 2, 3]]
        method_breakdown.columns = ['Window Based', 'Day Ahead', 'Entire History']
        
        # Combine into report
        report = pd.concat([pattern_hits, method_breakdown], axis=1)
        report = report.sort_values('total_hits', ascending=False)
        
        print(report.to_string())
        
        # Save to CSV
        report.to_csv('output/pattern_hit_counts.csv')
        print("\nReport saved to 'output/pattern_hit_counts.csv'")

    def run_analysis(self):
        """Main analysis function"""

        # Load data
        self.load_data()
        
        if self.actual_data is None:
            print("Cannot proceed without data. Please check file paths.")
            return
        
        # Calculate methods matrix
        methods = [
            [len(self.predicted_data_window_based), 
             math.ceil(len(self.predicted_data_window_based) / self.window_size)],
            [len(self.predicted_data_day_based), 
             math.ceil(len(self.predicted_data_day_based) / self.window_size)],
            [len(self.predicted_data_historical_based), 
             math.ceil(len(self.predicted_data_historical_based) / self.window_size)]
        ]
        
        print("="*70)
        print("ANOMALOUSDTW INSIDER TRADING DETECTION")
        print("="*70)
        print(f"DTW Distance Threshold: {self.dtw_threshold}")
        print(f"Window Size: {self.window_size} days")
        print("\nMethods configuration:")
        for i, method_info in enumerate(methods, 1):
            print(f"Method {i}: Length={method_info[0]}, Windows={method_info[1]}")
        
        # Main analysis loops
        total_comparisons = 0
        for method in range(1, 4):  # methods 1, 2, 3
            print(f"\nProcessing method {method}...")
            num_window = methods[method - 1][1]
            
            for window in range(1, num_window + 1):
                for pattern in range(1, self.num_pattern + 1):
                    self.overlap = 10
                    if window == num_window:
                        self.overlap = 0
                    
                    # Calculate range for sliding window
                    pattern_size = int(self.pattern_desc_arr[pattern - 1])
                    max_day = self.window_size - (pattern_size - self.overlap)
                    
                    for day in range(1, max_day + 1):
                        self.DTW(method, window, pattern, day)
                        total_comparisons += 1
        
        print(f"\nTotal comparisons performed: {total_comparisons}")
        print(f"Anomalies detected (DTW ≤ {self.dtw_threshold}): {len(self.result)}")
        
        # Save detailed results
        os.makedirs('output', exist_ok=True)
        if self.result:
            df_results = pd.DataFrame(self.result)
            df_results.to_csv('output/dtw_anomalies_detailed.csv', index=False)
            print(f"\nDetailed results saved to 'output/dtw_anomalies_detailed.csv'")
            
            # Generate reports matching the paper
            self.find_minimum_dtw_per_method()
            self.generate_hit_count_report()
        else:
            print("\n" + "="*70)
            print("NO ANOMALIES DETECTED")
            print(f"No trading patterns matched insider trading patterns with DTW ≤ {self.dtw_threshold}")
            print("This suggests normal trading behavior")
            print("="*70)

def main():
    """Main function to run the analysis"""
    print("Time Series Similarity Analysis")
    print("=" * 50)
    
    analyzer = TimeSeriesSimilarity()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()