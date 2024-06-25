import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby
import pandas as pd
from os.path import exists
from generate_chunks import load_data
from train_models import ModelTrainer


def extreme_anomaly(dist):
    q25, q75, q90, q95, q99 = np.quantile(dist, [0.25, 0.75, 0.9, 0.95, 0.99])
    #return q99
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


def main():
    train_chunks, training_chunk_dates, test_chunks, test_chunk_dates = load_data()

    alpha = 0.05
    threshold = 0.5
    print('Alpha:', alpha)

    plt.figure(figsize=(12, 6))

    # Plot failure one - air pressure
    plt.axvspan(xmin=np.datetime64('2022-06-04T10:19:24.300000000'), xmax=np.datetime64('2022-06-04T14:22:39.188000000'), color='gray', alpha=0.5)
    plt.axvline(x=np.datetime64('2022-06-04T11:26:01.422000000'), color='black', linestyle='--')
    # Plot failure two - oil leak
    plt.axvspan(xmin=np.datetime64('2022-07-11T10:10:18.948000000'), xmax=np.datetime64('2022-07-14T10:22:08.046000000'), color='gray', alpha=0.5)
    plt.axvline(x=np.datetime64('2022-07-13T19:43:52.593000000'), color='black', linestyle='--')

    models = ['LSTM_AE', 'TCN_AE', 'WAE_NOGAN']
    for model_name in models:
        print(model_name)
        model = ModelTrainer(f'configs/{model_name}.json').fit()
        val_size = int(0.3 * len(train_chunks))
        train_errors = model.calc_loss(train_chunks[-val_size:], train_chunks[-val_size:], average=False).mean(axis=(1,2))
        test_errors = model.calc_loss(test_chunks, test_chunks, average=False).mean(axis=(1,2))

        anom = extreme_anomaly(train_errors)
        binary_output = (test_errors > anom).astype(np.int8)

        output = simple_lowpass_filter(binary_output,alpha)
        failures = (output >= threshold).astype(np.int8)
        plt.plot(test_chunk_dates[:, 1], output, label=model_name)

        print_failures(test_chunk_dates, failures)
        print('---')

    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('test.png')

    # Save another grafic, this time zoomed in
    plt.xlim(np.datetime64('2022-06-04T06:00:00.000000000'), np.datetime64('2022-06-04T23:59:00.000000000'))
    plt.savefig('test2.png')

    plt.xlim(np.datetime64('2022-07-10T22:00:00.000000000'), np.datetime64('2022-07-15T12:00:00.000000000'))
    plt.savefig('test3.png')

    plt.xlim(np.datetime64('2022-06-19T00:00:00.000000000'), np.datetime64('2022-06-21T00:00:00.000000000'))
    plt.savefig('test4.png')


if __name__ == '__main__':
    main()