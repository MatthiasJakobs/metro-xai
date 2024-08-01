import numpy as np
import pickle
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby
import pandas as pd
from generate_chunks import load_data
from train_models import ModelTrainer
from os import makedirs
from extract_rules import construct_features

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

def failure_pt3():
    makedirs('plots', exist_ok=True)
    train_chunks, training_chunk_dates, test_chunks, test_chunk_dates = load_data(version=3)

    alpha = 0.05
    threshold = 0.5
    print('Alpha:', alpha)

    plt.figure(figsize=(12, 6))

    failures = [
        (np.datetime64('2020-04-18T00:00:00.000000000'), np.datetime64('2020-04-18T23:59:00.000000000')),
        (np.datetime64('2020-05-29T23:30:00.000000000'), np.datetime64('2020-05-30T06:00:00.000000000')),
        (np.datetime64('2020-06-05T10:00:00.000000000'), np.datetime64('2020-06-07T14:30:00.000000000')),
        (np.datetime64('2020-07-15T14:30:00.000000000'), np.datetime64('2020-07-15T19:00:00.000000000')),
    ]

    for failure in failures:
        plt.axvspan(xmin=failure[0], xmax=failure[1], color='gray', alpha=0.5)

    # Load unnormalized data
    with open('data/pt3_test_chunks_unnormalized.pkl', 'rb') as f:
        test_chunks_unnormalized = pickle.load(f)

    oil_temperature = test_chunks_unnormalized[..., 5].max(axis=-1)
    motor_current = test_chunks_unnormalized[..., 6].min(axis=-1)
    binary_output = (motor_current <= 0.014).astype(np.int8)
    binary_output += np.logical_and(motor_current > 0.014, oil_temperature > 73.487).astype(np.int8)
    output = simple_lowpass_filter(binary_output, alpha)
    #output = binary_output
    failures = (output >= threshold).astype(np.int8)
    print_failures(test_chunk_dates, failures)
    plt.plot(test_chunk_dates[:, 1], output, label='manual_tree')
    print(' ')
    # plt.plot(test_chunk_dates[:, 1], oil_temperature, label='oil_temperature')
    # plt.plot(test_chunk_dates[:, 1], motor_current, label='motor_current')

    X = construct_features(test_chunks_unnormalized, axis=1).swapaxes(1, 2)
    n_batches = X.shape[0]
    X = X.reshape(n_batches, -1)
    with open('models/pt2_dt_transfer_large.pkl', 'rb') as f:
        tree = pickle.load(f)
    binary_output = tree.predict(X)
    output = simple_lowpass_filter(binary_output, alpha)
    print_failures(test_chunk_dates, failures)
    plt.plot(test_chunk_dates[:, 1], output, label='large_tree')

    plt.legend()
    plt.tight_layout()
    plt.show()
    
    exit()

    # Plot failure one - air pressure
    plt.axvspan(xmin=np.datetime64('2022-06-04T10:19:24.300000000'), xmax=np.datetime64('2022-06-04T14:22:39.188000000'), color='gray', alpha=0.5)
    plt.axvline(x=np.datetime64('2022-06-04T11:26:01.422000000'), color='black', linestyle='--')
    # Plot failure two - oil leak
    plt.axvspan(xmin=np.datetime64('2022-07-11T10:10:18.948000000'), xmax=np.datetime64('2022-07-14T10:22:08.046000000'), color='gray', alpha=0.5)
    plt.axvline(x=np.datetime64('2022-07-13T19:43:52.593000000'), color='black', linestyle='--')

    models = ['TCN_AE', 'WAE_NOGAN']
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
    plt.savefig('plots/test.png')

    # Save another grafic, this time zoomed in
    plt.xlim(np.datetime64('2022-06-04T06:00:00.000000000'), np.datetime64('2022-06-04T23:59:00.000000000'))
    plt.savefig('plots/test2.png')

    plt.xlim(np.datetime64('2022-07-10T22:00:00.000000000'), np.datetime64('2022-07-15T12:00:00.000000000'))
    plt.savefig('plots/test3.png')

    plt.xlim(np.datetime64('2022-06-19T00:00:00.000000000'), np.datetime64('2022-06-21T00:00:00.000000000'))
    plt.savefig('plots/test4.png')


def set_style(width_pt, height_fraction=1):
    # Width of figure (in pts)
    fig_width_pt = width_pt 
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * height_fraction

    # Fonts
    tex_fonts = {
        # Use LaTeX to write all text
        #"text.usetex": True,
        #"font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6
    }

    plt.rcParams.update(tex_fonts)

    return (fig_width_in, fig_height_in)

def failure_pt2():
    makedirs('plots', exist_ok=True)
    train_chunks, training_chunk_dates, test_chunks, test_chunk_dates = load_data(version=2)

    alpha = 0.05
    threshold = 0.5
    print('Alpha:', alpha)

    plt.figure(figsize=set_style(468.0, height_fraction=0.5))

    # Plot failure one - air pressure
    plt.axvspan(xmin=np.datetime64('2022-06-04T10:19:24.300000000'), xmax=np.datetime64('2022-06-04T14:22:39.188000000'), color='gray', alpha=0.5)
    plt.axvline(x=np.datetime64('2022-06-04T11:26:01.422000000'), color='black', linestyle='--')
    # Plot failure two - oil leak
    plt.axvspan(xmin=np.datetime64('2022-07-11T10:10:18.948000000'), xmax=np.datetime64('2022-07-14T10:22:08.046000000'), color='gray', alpha=0.5)
    plt.axvline(x=np.datetime64('2022-07-13T19:43:52.593000000'), color='black', linestyle='--')

    #models = ['TCN_AE', 'WAE_NOGAN']
    models = ['WAE_NOGAN']
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

        if model_name == 'WAE_NOGAN':
            model_name = ''
            np.save('data/pt2_waenogan_output.npy', output.squeeze())

        plt.plot(test_chunk_dates[:, 1], output, label=model_name)

        print_failures(test_chunk_dates, failures)
        print('---')

    plt.ylabel(r'$p(failure)$')
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=r'$\tau_{fail}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/test.png')
    plt.savefig('plots/pfailure.pdf', transparent=True)

    # Save another grafic, this time zoomed in
    plt.xlim(np.datetime64('2022-06-04T06:00:00.000000000'), np.datetime64('2022-06-04T23:59:00.000000000'))
    plt.savefig('plots/test2.png')

    plt.xlim(np.datetime64('2022-07-10T22:00:00.000000000'), np.datetime64('2022-07-15T12:00:00.000000000'))
    plt.savefig('plots/test3.png')

    plt.xlim(np.datetime64('2022-06-19T00:00:00.000000000'), np.datetime64('2022-06-21T00:00:00.000000000'))
    plt.savefig('plots/test4.png')

    # Construct ground truth
    binary_output = (test_errors > anom).astype(np.int8)
    all_alerts = binary_output.squeeze()
    correct_alerts = all_alerts.copy()
    # Remove false positives
    correct_alerts[4000:9000] = 0
    # Save for further processing
    np.save('data/pt2_correct_alerts.npy', correct_alerts)


if __name__ == '__main__':
    failure_pt2()
    #failure_pt3()