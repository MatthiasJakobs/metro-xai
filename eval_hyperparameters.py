import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import exists
from itertools import product

from generate_chunks import load_data
from train_models import ModelTrainer
from failure_detection import print_failures, simple_lowpass_filter

def prepare_data(prefix):
    model_name = f'{prefix.upper()}_TCN'
    model = ModelTrainer(f'configs/{model_name}.json').fit()
    train_chunks, training_chunk_dates, test_chunks, test_chunk_dates = load_data(version=model.version, scaler=model.scaler)

    val_size = int(0.3 * len(train_chunks))
    train_errors = model.calc_loss(train_chunks[-val_size:], train_chunks[-val_size:], average=False).mean(axis=(1,2))
    test_errors_raw = model.calc_loss(test_chunks, test_chunks, average=False)
    test_errors = test_errors_raw.mean(axis=(1,2))

    return train_errors, test_errors, test_chunk_dates

def interval_f1(actual_failures, predicted_failures, eps=1e-12):

    def has_overlap(s1, e1, s2, e2):
        return min(e1, e2) > max(s1, s2)

    # Count TP/FP on predicted side
    tp_pred = 0
    fp_pred = 0
    for p_s, p_e in predicted_failures:
        if p_e <= p_s:
            continue
        overlapped = False
        for a_s, a_e in actual_failures:
            if a_e <= a_s:
                continue
            if has_overlap(p_s, p_e, a_s, a_e):
                overlapped = True
                break
        if overlapped:
            tp_pred += 1
        else:
            fp_pred += 1

    # Count how many actual intervals are hit by any prediction (for recall)
    hit_actual = 0
    fn_actual = 0
    for a_s, a_e in actual_failures:
        if a_e <= a_s:
            continue
        overlapped = False
        for p_s, p_e in predicted_failures:
            if p_e <= p_s:
                continue
            if has_overlap(a_s, a_e, p_s, p_e):
                overlapped = True
                break
        if overlapped:
            hit_actual += 1
        else:
            fn_actual += 1

    precision = tp_pred / (tp_pred + fp_pred + eps)
    recall = hit_actual / (hit_actual + fn_actual + eps)

    f1 = (2.0 * precision * recall) / (precision + recall + eps)

    return f1, precision, recall, tp_pred, fp_pred, hit_actual, fn_actual

def interval_iou(actual_failures, predicted_failures):

    def total_intersection(a, b):
        inter = np.timedelta64(0, 'ns')
        for a_start, a_end in a:
            for b_start, b_end in b:
                start = max(a_start, b_start)
                end = min(a_end, b_end)
                if end > start:
                    inter += (end - start)
        return inter

    def total_duration(intervals):
        dur = np.timedelta64(0, 'ns')
        for start, end in intervals:
            if end > start:
                dur += (end - start)
        return dur

    intersection = total_intersection(actual_failures, predicted_failures)
    union = (
        total_duration(actual_failures)
        + total_duration(predicted_failures)
        - intersection
    )

    if union == np.timedelta64(0, 'ns'):
        return 0.0

    return intersection / union

def main(version=2):
    if version == 1:
        prefix = 'pt1'
        actual_failures = [
            (np.datetime64('2022-02-28T21:53:00'), np.datetime64('2022-03-01T02:00:00')),
            (np.datetime64('2022-03-23T14:54:00'), np.datetime64('2022-03-23T15:24:00')),
            (np.datetime64('2022-05-30T12:00:00'), np.datetime64('2022-06-02T06:18:00')),
        ]
    elif version == 2:
        prefix = 'pt2'
        actual_failures = [
            (np.datetime64('2022-06-04T10:19:00'), np.datetime64('2022-06-04T14:22:00')),
            (np.datetime64('2022-07-11T10:10:00'), np.datetime64('2022-07-14T10:22:00')),
        ]
    else:
        raise Exception('Unknown version numner', version)

    if not exists(f'data/{prefix}_train_errors.npy') or not exists(f'data/{prefix}_test_errors.npy') or not exists(f'data/{prefix}_test_chunk_dates.npy'):
        print('Prepare data')
        train_errors, test_errors, test_chunk_dates = prepare_data(prefix=prefix)
        np.save(f'data/{prefix}_train_errors.npy', train_errors)
        np.save(f'data/{prefix}_test_errors.npy', test_errors)
        np.save(f'data/{prefix}_test_chunk_dates.npy', test_chunk_dates)
    else:
        train_errors = np.load(f'data/{prefix}_train_errors.npy')
        test_errors = np.load(f'data/{prefix}_test_errors.npy')
        test_chunk_dates = np.load(f'data/{prefix}_test_chunk_dates.npy')

    #best_config = {'anom_factor': 2, 'lp': 0.1, 'q': 0.99}
    alphas = [0.05, 0.1, 0.15]
    betas = [1, 2, 3]

    print(f'=== Hyperparameter analysis for {prefix.upper()} ===')
    
    hyperparameters = list(product(alphas, betas))
    results = {'alpha':[], 'beta': [], 'Precision': [], 'Recall': [], 'F1': [], 'IoU': []}
    for (alpha, beta) in hyperparameters:
        anom = np.quantile(train_errors, [0.99])*beta
        binary_output = (test_errors > anom).astype(np.int8)
        output = simple_lowpass_filter(binary_output, alpha)

        detected_failures = []
        current_failure_start = None
        for t in range(1, len(output)):
            failure_start = output[t] > 0.5 and output[t-1] <= 0.5
            failure_stop = output[t] <= 0.5 and output[t-1] > 0.5
            if failure_start:
                current_failure_start = test_chunk_dates[t][0]
            if failure_stop:
                detected_failures.append([current_failure_start, test_chunk_dates[t][0]])
                
        # Compute scores
        iou = interval_iou(actual_failures, detected_failures)
        f1, precision, recall = interval_f1(actual_failures, detected_failures)[:3]
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1'].append(f1)
        results['IoU'].append(iou)
        results['alpha'].append(alpha)
        results['beta'].append(beta)
        #print(f'alpha={alpha} beta={beta} | f1={f1} iou={iou}')

    results = pd.DataFrame(results)
    results['IoU'] = results['IoU'].round(3)
    results['Recall'] = results['Recall'].round(3)
    results['Precision'] = results['Precision'].round(3)
    results['F1'] = results['F1'].round(3)
    print(results)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run hyperparameter evaluation')
    parser.add_argument('--version', '-v', type=int, choices=[1,2], default=2, help='PT version (1 or 2)')
    args = parser.parse_args()
    main(version=args.version)
