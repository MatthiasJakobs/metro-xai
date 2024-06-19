import numpy as np
import pickle as pkl
from operator import itemgetter
from itertools import groupby
import pandas as pd
from scipy.stats import zscore
from os.path import exists
from generate_chunks import load_data
from train_models import ModelTrainer


def extreme_anomaly(dist):
    q25, q75 = np.quantile(dist, [0.25, 0.75])
    return q75 + 3*(q75-q25)

def simple_lowpass_filter(arr, alpha):
    y = arr[0]
    filtered_arr = [y]
    for elem in arr[1:]:
        y = y + alpha * (elem - y)
        filtered_arr.append(y)
    return np.array(filtered_arr)


def detect_failures(anom_indices):
    failure_list = []
    failure = set()
    for i in range(len(anom_indices) - 1):
        if anom_indices[i] == 1 and anom_indices[i + 1] == 1:
            failure.add(i)
            failure.add(i + 1)
        elif len(failure) > 0:
            failure_list.append(failure)
            failure = set()

    if len(failure) > 0:
        failure_list.append(failure)

    return failure_list


def failure_list_to_interval(cycle_dates, failures):
    failure_intervals = []
    for failure in failures:
        failure = sorted(failure)
        from_timestamp = pd.Timestamp(cycle_dates[failure[0]][0])
        to_timestamp = pd.Timestamp(cycle_dates[failure[-1]][1])
        failure_intervals.append(pd.Interval(from_timestamp, to_timestamp, closed='both'))
    return failure_intervals


def collate_intervals(interval_list):
    diff_consecutive_intervals = [(interval_list[i+1].left - interval_list[i].right).days for i in range(len(interval_list)-1)]
    lt_1day = np.where(np.array(diff_consecutive_intervals) <= 1)[0]
    collated_intervals = []
    for k, g in groupby(enumerate(lt_1day), lambda ix: ix[0]-ix[1]):
        collated = list(map(itemgetter(1), g))
        collated_intervals.append(pd.Interval(interval_list[collated[0]].left, interval_list[collated[-1]+1].right, closed="both"))

    collated_intervals.extend([interval_list[i] for i in range(len(interval_list)) if i not in lt_1day and i-1 not in lt_1day])
    return sorted(collated_intervals)


def print_failures(cycle_dates, output):
    failures = detect_failures(output)
    failure_intervals = failure_list_to_interval(cycle_dates, failures)
    collated_intervals = collate_intervals(failure_intervals)
    for interval in collated_intervals:
        print(interval)


##### Results from the main paper #####

def generate_intervals(granularity, start_timestamp, end_timestamp):
    current_timestamp = start_timestamp
    interval_length = pd.offsets.DateOffset(**granularity)
    interval_list = []
    while current_timestamp < end_timestamp:
        interval_list.append(pd.Interval(current_timestamp, current_timestamp + interval_length, closed="left"))
        current_timestamp = current_timestamp + interval_length
    return interval_list


def main():
    print('Load data')

    train_chunks, training_chunk_dates, test_chunks, test_chunk_dates = load_data()

    alpha = 0.05

    print('Load model')
    model = ModelTrainer('configs/AE_MAD.json').fit()
    train_errors = model.calc_loss(train_chunks, train_chunks, average=False).mean(axis=(1,2))
    test_errors = model.calc_loss(test_chunks, test_chunks, average=False).mean(axis=(1,2))

    anom = extreme_anomaly(train_errors)
    binary_output = (test_errors > anom).astype(np.int8)
    print(np.mean(binary_output))

    output = simple_lowpass_filter(binary_output,alpha)
    failures = (output >= 0.5).astype(np.int8)
    print(np.mean(output))

    # TODO: Double check dates. Maybe we need to choose end of each segment as correct timestamp
    print_failures(test_chunk_dates, failures)

if __name__ == '__main__':
    main()