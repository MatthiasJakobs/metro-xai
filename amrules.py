import tqdm
import numpy as np
import collections
import pickle
import pandas as pd

from generate_chunks import load_data
from train_models import ModelTrainer
from failure_detection import simple_lowpass_filter

from river import base
from river import compose
from river import stream
from river import rules
from river import tree
from river import drift
from river import imblearn


RuleStat = collections.namedtuple("RuleStat", ["n_rules", "n_drifts", "n_literals"])

# Inspect rules
def save_rules(regressor, model_name, dataset_name):
    amrules = extract_regressor(regressor)

    with open(f"../global_explanations/{model_name}_{dataset_name}.txt", "w") as f:
        for i in range(len(amrules) - 1):
            f.write(f"Rule {i}\n")
            repr_orig = repr(amrules[i])
            if amrules.pred_type == "mean":
                repr_final = (
                    f"\t{repr_orig} → {amrules[i].pred_model.mean.get():.4f}\n\n"
                )
                f.write(repr_final)
            else:
                f.write(f"\t{repr_orig}\n\n")


def get_model_template(model_name, **params):
    amrules = rules.AMRules(
        splitter=tree.splitter.TEBSTSplitter(1),
        pred_type="mean",
        ordered_rule_set=False,
        n_min=100,
        delta=0.05,
        min_samples_split=15,
        **params,
    )

    if model_name == "AMRules":
        return amrules
    elif model_name == "ChebyUS":
        return imblearn.ChebyshevUnderSampler(regressor=amrules, sp=0.15, seed=42)
    elif model_name == "ChebyOS":
        return imblearn.ChebyshevOverSampler(regressor=amrules)


def extract_regressor(model):
    if isinstance(model, rules.AMRules):
        regressor = model
    else:
        regressor = model.regressor

    return regressor


def run_amrules(dataset_name, ignored, params, error_threshold, model_name):
    def converter(v):
        if len(v) > 0:
            return float(v)

    df = pd.read_csv(f"../datasets/{dataset_name}.csv")
    feature_names = list(df)
    metadata = df.loc[:, ignored]
    del df

    converters = {name: converter for name in feature_names if name not in ignored}

    reg = get_model_template(model_name, **params)

    # Testing
    dataset = stream.iter_csv(
        f"../datasets/{dataset_name}.csv",
        converters=converters,
        drop=ignored,
        drop_nones=True,
        target=None,
    )
   
    ys = {}
    local_explanations = []
    resource_usage = ["instances,time_tr,time_ts,total_time,memory\n"]
    total_trt = 0
    total_tst = 0

    rules_log = []
    rule_stats = collections.deque(maxlen=2)

    for i, (x, _) in enumerate(dataset):
        y = x.pop("re", None)
        if y >= error_threshold:
            covered = []
            for rule in extract_regressor(reg)._rules.values():
                 covered.append(rule.covers(x))

            if any(covered):
                local_explanations.append(f"Sample {i + 1}\ty={y}\n")
                local_explanations.append(f"Timestamp: {metadata.iloc[i, :]}\n")
                local_explanations.append("\nAMRules prediction:\n")
                local_explanations.append(
                    "_____________________________________________________\n"
                )
                local_explanations.append(extract_regressor(reg).debug_one(x))
                local_explanations.append("\n\n\n")
                local_explanations.append(
                    f"AMRules anomaly score: {extract_regressor(reg).anomaly_score(x)}\n"
                )
                local_explanations.append(
                    "_____________________________________________________\n"
                )
                local_explanations.append("\n\n")


            reg.learn_one(x, y)

        if (i + 1) % 1000 == 0:
            metric_vals = [str(total_trt), str(total_tst), str(total_trt + total_tst)]
            metric_vals.append(reg._memory_usage)  # noqa
            metric_vals = ",".join(metric_vals)
            resource_usage.append(str(i + 1) + "," + metric_vals + "\n")

    with open(
            f"../local_explanations/output_anomalies_{model_name}_{dataset_name}.txt", "w"
    ) as f_anom:
        local_explanations = "".join(local_explanations)
        f_anom.write(local_explanations)

    with open(f"../resource_usage/{model_name}_{dataset_name}.csv", "w") as err_out:
        resource_usage = "".join(resource_usage)
        err_out.write(resource_usage)

    with open(f"../models/{model_name}_{dataset_name}.bin", "wb") as f_model:
        pickle.dump(reg, f_model, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"../rules_log/{model_name}_{dataset_name}.txt", "w") as f_changes:
        changelog = "".join(rules_log)
        f_changes.write(changelog)

    return reg


class NoDrift(base.DriftDetector):
    def update(self, x):
        pass

# def process_starter(dataset_name, ignored, error_threshold, model_name):
#     reg = run_amrules(
#         dataset_name=dataset_name,
#         ignored=ignored,
#         #params={"drift_detector": NoDrift()},
#         params={"drift_detector":drift.binary.EDDM()},
#         error_threshold=error_threshold,
#         model_name=model_name,
#     )

import re
from typing import Dict, List

_rule_block = re.compile(
    r'(?ms)^Rule\s+\d+:\s*(.+?)(?=(?:^Rule\s+\d+:)|^[^\S\r\n]*Prediction\s*\(mean\):|^Final prediction:|\Z)'
)

def parse_rules_block(block: str) -> List[str]:
    rules = []
    for m in _rule_block.finditer(block):
        text = m.group(1)
        # Normalize whitespace to a single line
        text = ' '.join(text.split())
        rules.append(text)
    return rules

def parse_rules(d: Dict[int, str]) -> Dict[int, List[str]]:
    return {k: parse_rules_block(v) for k, v in d.items()}

def extract_rules(features, dates, errors, lp_output, feature_names, model='AMRules', threshold=0.5):
    # Create AMRules model
    amrules = get_model_template(model)

    # Repeat errors and lp output to get second-level granularity
    errors = np.repeat(errors, 1800)
    lp_output = np.repeat(lp_output, 1800)

    # Convert features into dict
    n_features = features.shape[-1]
    features = features.reshape(-1, n_features)
    
    # Iterate / learn over failures
    local_explanations = {}
    failure_bounds = []

    #for i in tqdm.trange(1620000):
    for i in tqdm.trange(len(lp_output), disable=True):
        out = lp_output[i]
        x = {f'{feature_names[k]}': float(v) for k, v in enumerate(features[i]) }
        y = errors[i]

        failure_start = lp_output[i] > threshold and lp_output[i-1] <= threshold
        failure_stop =  lp_output[i] <= threshold and lp_output[i-1] > threshold

        if failure_start:
            fail = [i]
        if failure_stop:
            fail.append(i)
            failure_bounds.append(fail)
            fail = []
        if out >= threshold:
            covered = []
            for rule in extract_regressor(amrules)._rules.values():
                 covered.append(rule.covers(x))

            if any(covered):
                local_explanations[i] = extract_regressor(amrules).debug_one(x)

            amrules.learn_one(x, y)

    describe_failures(failure_bounds, local_explanations)

def describe_failures(failure_bounds, rule_dict):
    for (fail_start, fail_stop) in failure_bounds:

        r = {k: v for k, v in rule_dict.items() if k >= fail_start and k <= fail_stop }
        r = parse_rules(r)
        r = {k: ' AND '.join(v) for k, v in r.items()}
        if len(r) == 0:
            print(f'No rule fitting failure {(fail_start, fail_stop)}')
            continue

        # Find unique rules
        unique_rules = np.unique(list(r.values()))
        print(f'For failure {(fail_start, fail_stop)} there are {len(unique_rules)} unique rules. Reporting the highest-support rule:')
        support = [len([v for v in r.values() if v == _r ]) for _r in unique_rules]

        top_3_indices = np.argsort(support)[-3:]
        failure_length = (fail_stop - fail_start)
        for idx in top_3_indices[::-1]:
            supp = support[idx] 
            supp_percent = support[idx] / failure_length
            error = (failure_length-supp)
            error_percent = error / failure_length
            print(f'{unique_rules[idx]} -> Support: {supp} ({supp_percent}) | error: {error} ({error_percent})')

        print(' ')


def main(version=2, method='AMRules'):

    path = f'configs/PT{version}_TCN.json'
    model = ModelTrainer(path).fit()

    print('Load data')
    train_chunks, training_chunk_dates, test_chunks, test_chunk_dates = load_data(version=model.version, scaler=model.scaler)
    with open(f'data/pt{version}_test_chunks_unnormalized.pkl', 'rb') as f:
        test_chunks_unnormalized = pickle.load(f)

    threshold = 0.5
    if version == 2:
        alpha = 0.15
        anom_factor = 3
        q = 0.99
    else:
        # TODO
        alpha = 0.1
        anom_factor = 2
        q = 0.99

    feature_names = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Flowmeter', 'Motor_current', 'COMP']

    print('Calculate model outputs')
    val_size = int(0.3 * len(train_chunks))
    train_errors = model.calc_loss(train_chunks[-val_size:], train_chunks[-val_size:], average=False).mean(axis=(1,2))
    test_errors = model.calc_loss(test_chunks, test_chunks, average=False).mean(axis=(1,2))

    anom = np.quantile(train_errors, q=q) * anom_factor
    binary_output = (test_errors > anom).astype(np.int8)

    output = simple_lowpass_filter(binary_output,alpha)

    extract_rules(test_chunks_unnormalized, test_chunk_dates, test_errors, output, feature_names, threshold=threshold, model=method)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run AMRules extraction')
    parser.add_argument('--version', '-v', type=int, choices=[1,2], default=2, help='PT version (1 or 2)')
    parser.add_argument('--method', '-m', type=str, default='AMRules', help='Method name (e.g. AMRules, ChebyUS, ChebyOS)')
    args = parser.parse_args()
    main(version=args.version, method=args.method)
