import pandas as pd
import numpy as np
import sys
import tqdm

def print_intervals(version=2):
    if version == 2:
        print('=== MetroPT 2 ===')
        final_metro = pd.read_csv('MetroPT2.csv')
        test_date = np.datetime64("2022-06-01T00:00:00.000000000")
    else:
        print('=== MetroPT 3 ===')
        final_metro = pd.read_csv('MetroPT3.csv')
        test_date = np.datetime64("2020-04-01T00:00:00.000000000")

    final_metro["timestamp"] = pd.to_datetime(final_metro["timestamp"])
    final_metro = final_metro.sort_values("timestamp")
    final_metro.reset_index(drop=True, inplace=True)

    test_data = final_metro[final_metro['timestamp'] >= test_date]
    test_data = test_data[['timestamp', 'LPS']]
    print(test_data['LPS'].mean())

    intervals = []
    min_interval_size = 60

    signal = np.where(test_data['LPS'] != 0)[0]
    t_start = signal[0]
    t_stop = signal[0]
    for i, idx in enumerate(signal[1:]):
        if idx-t_stop == 1:
            t_stop = idx
        else:
            if t_stop-t_start >= min_interval_size:
                intervals.append(pd.Interval(test_data.iloc[t_start]['timestamp'], test_data.iloc[t_stop]['timestamp'], closed='both'))
            t_start = idx
            t_stop = idx

    for i in intervals:
        print(i)

if __name__ == '__main__':
    print_intervals(version=2)
    print_intervals(version=3)