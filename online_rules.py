import tqdm
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from generate_chunks import load_data
from train_models import ModelTrainer
from failure_detection import simple_lowpass_filter
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from plotting import default_plot

def construct_features(X, axis=-1):
    # Assume: X.shape = (batch_size, L, n_channels)
    avg = np.mean(X, axis=axis, keepdims=True)
    var = np.var(X, axis=axis, keepdims=True)
    mx = np.max(X, axis=axis, keepdims=True)
    mn = np.min(X, axis=axis, keepdims=True)
    return np.concatenate([avg, var, mx, mn], axis=axis)

def compare_trees(tree1, tree2):
    if tree1.get_depth() != tree2.get_depth():
        return False
    if len(tree1.tree_.feature.shape) != len(tree2.tree_.feature.shape):
        return False
    if (tree1.tree_.feature != tree2.tree_.feature).any():
        return False
    if (tree1.tree_.threshold != tree2.tree_.threshold).any():
        return False

    return True

def find_unique_trees(X, y, n=50, random_state=None):
    min_depth = 100
    trees = []
    for _ in range(n):
        tree = DecisionTreeClassifier(random_state=random_state)
        tree.fit(X, y)
        acc = tree.score(X, y)
        if acc == 1 and tree.get_depth() <= min_depth:
            if not any([compare_trees(tree, other_tree) for other_tree in trees]) or len(trees) == 0:
                min_depth = tree.get_depth()
                trees.append(tree)
    return trees

class OnlineRL:

    def __init__(self, warning_thresh=0.01, failure_thresh=0.5, random_state=182616, save_prefix='flowmeter'):
        self.warning_thresh = warning_thresh
        self.failure_thresh = failure_thresh
        self.rng = np.random.RandomState(random_state)
        self.save_prefix = save_prefix

    def run(self, output, X, history, feature_names):
        self.log = []

        # Buffer stores the datapoints during warning
        buffer = []
        trees = []

        global_buffer = []

        for t in range(len(X)): 
            is_warning = output[t] > self.warning_thresh
            is_failure = output[t] > self.failure_thresh
            is_increasing = (output[t] >= output[t-1])

            state = {'t': t, 'output': output[t], 'is_warning': is_warning, 'is_failure': is_failure, 'failure_increase': False}

            if is_warning and is_increasing:
                buffer.append(np.expand_dims(X[t], 0))
            else:
                # If has been part of an failure, add to global buffer
                if is_failure and len(buffer) > 0:
                    global_buffer.append(np.concatenate(buffer))
                # Reset buffer and add to history
                buffer = []
                history.append(np.expand_dims(X[t], 0))

            if is_failure:
                if is_increasing:
                    reported = False
                    X_good = np.concatenate(history)
                    y_good = np.zeros((X_good.shape[0]))
                    X_anom = np.concatenate(buffer)
                    y_anom = np.ones((X_anom.shape[0]))
                    _x, _y = np.concatenate([X_good, X_anom]), np.concatenate([y_good, y_anom])
                    _x = _x.reshape(_x.shape[0], -1)
                    state['failure_increase'] = True

                    # See if any rule still applies
                    new_trees = []
                    for tree in trees:
                        applies = tree.score(_x, _y) == 1
                        if applies:
                            new_trees.append(tree)
                    
                    
                    trees = new_trees
                    if len(trees) == 0:
                        trees = find_unique_trees(_x, _y, random_state=self.rng)

                    state['n_trees'] = len(trees)
                    state['rules'] = [export_text(tree, feature_names=feature_names) for tree in trees]

                    # if t == 991:
                    #     with open('d1_tree.pkl', 'wb') as f:
                    #         pickle.dump(trees[0], f)
                    # if t == 11126:
                    #     with open('d2_tree.pkl', 'wb') as f:
                    #         pickle.dump(trees[0], f)

                else:
                    if not reported:
                        print(f'Decreasing at t={t}, these are the rules:')
                        for tidx, tree in enumerate(trees):
                            print(export_text(tree, feature_names=feature_names))
                            fig, ax = plt.subplots(1,1)
                            plot_tree(tree, ax=ax, feature_names=feature_names, class_names=['no failure', 'failure'])
                            fig.tight_layout()
                            fig.savefig(f'plots/{self.save_prefix}_rules_t={t}_{tidx}.png')
                        reported = True
                    buffer = []
                    trees = []
                    state['n_trees'] = 0
                    state['rules'] = []

            self.log.append(state)

        # Fit global rules

        X_good = np.concatenate(history)
        y_good = np.zeros((X_good.shape[0]))
        X_anom = np.concatenate(global_buffer)
        y_anom = np.ones((X_anom.shape[0]))
        _x, _y = np.concatenate([X_good, X_anom]), np.concatenate([y_good, y_anom])
        _x = _x.reshape(_x.shape[0], -1)

        trees = find_unique_trees(_x, _y, random_state=self.rng)
        print('--- Global rule(s) found: ---')
        for idx, tree in enumerate(trees):
            print(export_text(tree, feature_names=feature_names))
            fig, ax = default_plot(subplots=(1,1), height_fraction=1.5)
            plot_tree(tree, ax=ax, feature_names=feature_names, class_names=['no failure', 'failure'], precision=2)
            fig.tight_layout()
            fig.savefig(f'plots/{self.save_prefix}_globalrules_{idx}.pdf', transparent=True)
            with open(f'models/{self.save_prefix}_tree_{idx}.pickle', 'wb') as f:
                pickle.dump(tree, f)

        self.log = pd.DataFrame(self.log)


def main():
    model = ModelTrainer(f'configs/TCN.json').fit()

    print('Load data')
    train_chunks, training_chunk_dates, test_chunks, test_chunk_dates = load_data(version=model.version, scaler=model.scaler)
    with open('data/pt2_train_chunks_unnormalized.pkl', 'rb') as f:
        train_chunks_unnormalized = pickle.load(f)
    with open('data/pt2_test_chunks_unnormalized.pkl', 'rb') as f:
        test_chunks_unnormalized = pickle.load(f)
    print('done')

    channel_names = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Flowmeter', 'Motor_current', 'COMP']

    train_chunks_features = construct_features(train_chunks_unnormalized, axis=1).swapaxes(1,2)
    test_chunks_features = construct_features(test_chunks_unnormalized, axis=1).swapaxes(1,2)

    transformed_feature_names = [[fname+'_mean', fname+'_var', fname+'_max', fname+'_min'] for fname in channel_names]
    transformed_feature_names = sum(transformed_feature_names, [])

    # train_chunks_features = construct_features(train_chunks_unnormalized, axis=1).swapaxes(1,2)
    # train_chunks_features = train_chunks_features[..., [2, 3]]
    # test_chunks_features = construct_features(test_chunks_unnormalized, axis=1).swapaxes(1,2)
    # test_chunks_features = test_chunks_features[..., [2, 3]]

    # transformed_feature_names = [[fname+'_max', fname+'_min'] for fname in channel_names]
    # transformed_feature_names = sum(transformed_feature_names, [])

    alpha = 0.15
    threshold = 0.5

    print('Calculate model outputs')
    val_size = int(0.3 * len(train_chunks))
    train_errors = model.calc_loss(train_chunks[-val_size:], train_chunks[-val_size:], average=False).mean(axis=(1,2))
    test_errors = model.calc_loss(test_chunks, test_chunks, average=False).mean(axis=(1,2))

    #anom = extreme_anomaly(train_errors)
    anom = np.quantile(train_errors, q=0.99) * 3
    binary_output = (test_errors > anom).astype(np.int8)

    output = simple_lowpass_filter(binary_output,alpha)
    failures = (output >= threshold).astype(np.int8)
    print('done')

    # History stores the "good" examples assumed to be non-anomalous
    history = [train_chunks_features]

    print(' ')
    print('Start OnlineRL with all features')
    orl = OnlineRL(save_prefix='flowmeter')
    orl.run(output, test_chunks_features, history, transformed_feature_names)
    print('done')
    orl.log.to_csv('flowmeter_failures.csv')

    print(' ')
    print('Start OnlineRL without Flowmeter')

    ## Restrict to not use Flowmeter
    feature_indices = np.array([0, 1, 2, 3, 4, 5, 7, 8])

    # History stores the "good" examples assumed to be non-anomalous
    history = [train_chunks_features[:, feature_indices]]
    transformed_feature_names = [tfn for tfn in transformed_feature_names if 'Flowmeter' not in tfn]

    orl = OnlineRL(save_prefix='noflowmeter')
    orl.run(output, test_chunks_features[:, feature_indices], history, transformed_feature_names)
    print('done')
    orl.log.to_csv('no_flowmeter_failures.csv')

if __name__ == '__main__':
    main()