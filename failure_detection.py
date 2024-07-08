import numpy as np
import pickle
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby
import pandas as pd
from os.path import exists
from generate_chunks import load_data
from train_models import ModelTrainer

def construct_features(X):
    # Assume: X.shape = (batch_size, L)
    avg = np.mean(X, axis=1, keepdims=True)
    var = np.var(X, axis=1, keepdims=True)
    mx = np.max(X, axis=1, keepdims=True)
    mn = np.min(X, axis=1, keepdims=True)
    #kurt = kurtosis(X, axis=1, keepdims=True)
    return np.concatenate([avg, var, mx, mn], axis=1)

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
    return collated_intervals


def main():
    train_chunks, training_chunk_dates, test_chunks, test_chunk_dates = load_data(version=2)

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

    with open('data/pt2_train_chunks_unnormalized.pkl', 'rb') as f:
        train_chunks_unnormalized = pickle.load(f)

    # Load unnormalized data
    with open('data/pt2_test_chunks_unnormalized.pkl', 'rb') as f:
        test_chunks_unnormalized = pickle.load(f)

    #models = ['WAE_NOGAN']
    models = []
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

    # Try a naive baseline for comparison
    # Only use Flowmeter variable. Whenever its non-zero, report the squared error
    print('NaiveFlowmeter')
    test_errors = np.mean((test_chunks[..., 6])**2, axis=-1)
    anom = 0
    binary_output = (test_errors > anom).astype(np.int8)

    output = simple_lowpass_filter(binary_output, alpha)
    failures = (output >= threshold).astype(np.int8)
    print_failures(test_chunk_dates, failures)
    plt.plot(test_chunk_dates[:, 1], output, label='NaiveFlowmeter')

    print('Manual Tree')

    binary_output = (test_chunks_unnormalized[..., 6].max(axis=1) > 9.566).astype(np.int8)

    output = simple_lowpass_filter(binary_output, alpha)
    failures = (output >= threshold).astype(np.int8)
    print_failures(test_chunk_dates, failures)
    plt.plot(test_chunk_dates[:, 1], output, label='Tree')

    # ----- Just use the rule, without smoothing or anything else
    print('Treshold')
    failures = (test_chunks_unnormalized[..., 6].max(axis=1) > 9.566).astype(np.int8)
    print_failures(test_chunk_dates, failures)
    plt.plot(test_chunk_dates[:, 1], failures, label='Threshold')

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

    # Construct ground truth
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    binary_output = (test_errors > anom).astype(np.int8)
    all_alerts = binary_output.squeeze()

    correct_alerts = all_alerts.copy()
    correct_alerts[4000:5000] = 0

    tree = DecisionTreeClassifier(max_depth=2, random_state=192781)
    X = np.concatenate([
        construct_features(test_chunks_unnormalized[..., 6])
    ], axis=1)
    feature_labels = ['flow_unnorm_mean', 'flow_unnorm_var', 'flow_unnorm_max', 'flow_unnorm_min']
    tree.fit(X, correct_alerts)
    print('train score (correct alerts)', tree.score(X, correct_alerts))
    fig, axs = plt.subplots(1,1)
    plot_tree(tree, ax=axs, feature_names=feature_labels, class_names=['no failure', 'failure'])
    fig.tight_layout()
    fig.savefig('tree.png')

    # Make sure this works on the real training part
    X = np.concatenate([
        construct_features(train_chunks_unnormalized[..., 6])
    ], axis=1)
    y = np.zeros((len(train_chunks)))
    print('score on real train data', tree.score(X, y))

    # Alternative based on other features
    print('=== alternative, no flow ===')
    channel_names = [
        'TP2',
        'TP3',
        'H1',
        'DV_pressure',
        'Reservoirs',
        'Oil_temperature',
        'Flowmeter',
        'Motor_current',
        'COMP'
    ]
    X = np.concatenate([
        construct_features(test_chunks_unnormalized[..., 0]),
        construct_features(test_chunks_unnormalized[..., 1]),
        construct_features(test_chunks_unnormalized[..., 2]),
        construct_features(test_chunks_unnormalized[..., 3]),
        construct_features(test_chunks_unnormalized[..., 4]),
        construct_features(test_chunks_unnormalized[..., 5]),
        construct_features(test_chunks_unnormalized[..., 7]),
        construct_features(test_chunks_unnormalized[..., 8]),
    ], axis=1)
    feature_names = sum([[cname+'_avg', cname+'_var', cname+'_max', cname+'_min'] for idx, cname in enumerate(channel_names) if idx not in [6]], [])
    print(feature_names)
    tree = DecisionTreeClassifier(max_depth=2, random_state=192781)
    tree.fit(X, correct_alerts)
    print('train score (correct alerts)', tree.score(X, correct_alerts))
    fig, axs = plt.subplots(1,1)
    plot_tree(tree, ax=axs, feature_names=feature_names, class_names=['no failure', 'failure'])
    fig.tight_layout()
    fig.savefig('alternative_rule.png')

    # Make sure this works on the real training part
    X = np.concatenate([
        construct_features(train_chunks_unnormalized[..., 0]),
        construct_features(train_chunks_unnormalized[..., 1]),
        construct_features(train_chunks_unnormalized[..., 2]),
        construct_features(train_chunks_unnormalized[..., 3]),
        construct_features(train_chunks_unnormalized[..., 4]),
        construct_features(train_chunks_unnormalized[..., 5]),
        construct_features(train_chunks_unnormalized[..., 7]),
        construct_features(train_chunks_unnormalized[..., 8]),
    ], axis=1)
    y = np.zeros((len(train_chunks)))
    print('score on real train data', tree.score(X, y))
    print('On all data:')
    all_chunks = np.concatenate([test_chunks_unnormalized, train_chunks_unnormalized], axis=0)
    X = np.concatenate([
        construct_features(all_chunks[..., 0]),
        construct_features(all_chunks[..., 1]),
        construct_features(all_chunks[..., 2]),
        construct_features(all_chunks[..., 3]),
        construct_features(all_chunks[..., 4]),
        construct_features(all_chunks[..., 5]),
        construct_features(all_chunks[..., 7]),
        construct_features(all_chunks[..., 8]),
    ], axis=1)
    y = np.concatenate([correct_alerts, np.zeros((len(train_chunks)))])
    tree = DecisionTreeClassifier(max_depth=3, random_state=192781)
    tree.fit(X, y)
    print(tree.score(X, y))
    fig, axs = plt.subplots(1,1)
    plot_tree(tree, ax=axs, feature_names=feature_names, class_names=['no failure', 'failure'])
    fig.tight_layout()
    fig.savefig('alternative_rule.png')


if __name__ == '__main__':
    main()