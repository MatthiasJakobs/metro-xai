import numpy as np
from operator import itemgetter
from itertools import groupby
import pandas as pd

def extreme_anomaly(dist):
    q05, q10, q25, q75, q90, q95, q97, q98, q99 = np.quantile(dist, [0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.97, 0.98, 0.99])
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
        # TODO: I think taking the right hand side of the first warning is correct, but needs to double check
        from_timestamp = pd.Timestamp(cycle_dates[failure[0]][1])
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
    return collated_intervals