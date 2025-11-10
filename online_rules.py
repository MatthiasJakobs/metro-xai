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

MIN_SUPP = 0.999

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

def compute_size(tree):
    tree = tree.tree_
    leaf_count = 0
    
    for node in range(tree.node_count):
        if tree.children_left[node] == tree.children_right[node]:
            leaf_count += 1
    
    return leaf_count

def find_unique_trees(X, y, n=50, random_state=None):
    min_depth = 100
    trees = []
    for _ in range(n):
        tree = DecisionTreeClassifier(random_state=random_state)
        tree.fit(X, y)
        acc = tree.score(X, y)
        if acc >= MIN_SUPP and tree.get_depth() <= min_depth:
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

            failure_start = output[t] > self.failure_thresh and output[t-1] <= self.failure_thresh
            failure_stop = output[t] <= self.failure_thresh and output[t-1] > self.failure_thresh

            state = {'t': t, 'output': output[t], 'is_warning': is_warning, 'is_failure': is_failure, 'failure_increase': False}

            if failure_start:
                print(f'Start to observe failure at t={t}')

            if is_warning:
                buffer.append(np.expand_dims(X[t], 0))
            else:
                # If has been part of an failure, add to global buffer
                if len(buffer) > 0:
                    global_buffer.append(np.concatenate(buffer))
                # Reset buffer and add to history
                buffer = []
                history.append(np.expand_dims(X[t], 0))

            if is_failure:
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
                    applies = tree.score(_x, _y) >= MIN_SUPP
                    if applies:
                        new_trees.append(tree)
                
                trees = new_trees
                if len(trees) == 0:
                    trees = find_unique_trees(_x, _y, random_state=self.rng)

            if failure_stop:
                 print(f'Stop to observe failure at t={t}, these are the rules:')
                 sizes = [compute_size(tree) for tree in trees]
                 print(sizes)
                 for tidx, tree in enumerate(trees[:3]):
                     print(export_text(tree, feature_names=feature_names))
                     fig, ax = plt.subplots(1,1)
                     plot_tree(tree, ax=ax, feature_names=feature_names, class_names=['no failure', 'failure'])
                     fig.tight_layout()
                     fig.savefig(f'plots/{self.save_prefix}_rules_t={t}_{tidx}.png')
                 buffer = []
                 trees = []
                 state['n_trees'] = 0
                 state['rules'] = []

            self.log.append(state)

        # Fit global rules

        # X_good = np.concatenate(history)
        # y_good = np.zeros((X_good.shape[0]))
        # X_anom = np.concatenate(global_buffer)
        # y_anom = np.ones((X_anom.shape[0]))
        # _x, _y = np.concatenate([X_good, X_anom]), np.concatenate([y_good, y_anom])
        # _x = _x.reshape(_x.shape[0], -1)

        # trees = find_unique_trees(_x, _y, random_state=self.rng)
        # print('--- Global rule(s) found: ---')
        # for idx, tree in enumerate(trees):
        #     print(export_text(tree, feature_names=feature_names))
            # fig, ax = default_plot(subplots=(1,1), height_fraction=1.5)
            # plot_tree(tree, ax=ax, feature_names=feature_names, class_names=['no failure', 'failure'], precision=2)
            # fig.tight_layout()
            # fig.savefig(f'plots/{self.save_prefix}_globalrules_{idx}.pdf', transparent=True)
            # with open(f'models/{self.save_prefix}_tree_{idx}.pickle', 'wb') as f:
            #     pickle.dump(tree, f)

        self.log = pd.DataFrame(self.log)

def run_pt2_new():
    model = ModelTrainer(f'configs/PT2_TCN.json').fit()

    print('Load data (MetroPT2)')
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

    alpha = 0.15

    print('Calculate model outputs')
    val_size = int(0.3 * len(train_chunks))
    train_errors = model.calc_loss(train_chunks[-val_size:], train_chunks[-val_size:], average=False).mean(axis=(1,2))
    test_errors = model.calc_loss(test_chunks, test_chunks, average=False).mean(axis=(1,2))

    anom = np.quantile(train_errors, q=0.99) * 3
    binary_output = (test_errors > anom).astype(np.int8)

    output = simple_lowpass_filter(binary_output,alpha)

    # History stores the "good" examples assumed to be non-anomalous
    history = [train_chunks_features]

    print(' ')
    print('Start OnlineRL with all features')
    orl = OnlineRL(save_prefix='flowmeter')
    orl.run(output, test_chunks_features, history, transformed_feature_names)
    print('done')

    print(' ')
    print('Start OnlineRL without Flowmeter')

    ## Restrict to not use Flowmeter
    feature_indices = np.array([0, 1, 2, 3, 4, 5, 7, 8])

    # History stores the "good" examples assumed to be non-anomalous
    history = [train_chunks_features[:, feature_indices]]
    transformed_feature_names = [tfn for tfn in transformed_feature_names if 'Flowmeter' not in tfn]

    orl = OnlineRL(save_prefix='noflowmeter')
    orl.run(output, test_chunks_features[:, feature_indices], history, transformed_feature_names)
    print(' ')

def run_pt1_new():
    model = ModelTrainer(f'configs/PT1_TCN.json').fit()

    print('Load data (MetroPT1)')
    train_chunks, training_chunk_dates, test_chunks, test_chunk_dates = load_data(version=model.version, scaler=model.scaler)
    with open('data/pt1_train_chunks_unnormalized.pkl', 'rb') as f:
        train_chunks_unnormalized = pickle.load(f)
    with open('data/pt1_test_chunks_unnormalized.pkl', 'rb') as f:
        test_chunks_unnormalized = pickle.load(f)

    channel_names = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Flowmeter', 'Motor_current', 'COMP']

    train_chunks_features = construct_features(train_chunks_unnormalized, axis=1).swapaxes(1,2)
    test_chunks_features = construct_features(test_chunks_unnormalized, axis=1).swapaxes(1,2)

    transformed_feature_names = [[fname+'_mean', fname+'_var', fname+'_max', fname+'_min'] for fname in channel_names]
    transformed_feature_names = sum(transformed_feature_names, [])

    alpha = 0.1

    val_size = int(0.3 * len(train_chunks))
    train_errors = model.calc_loss(train_chunks[-val_size:], train_chunks[-val_size:], average=False).mean(axis=(1,2))
    test_errors = model.calc_loss(test_chunks, test_chunks, average=False).mean(axis=(1,2))

    anom = np.quantile(train_errors, q=0.99) * 2
    binary_output = (test_errors > anom).astype(np.int8)

    output = simple_lowpass_filter(binary_output,alpha)

    # History stores the "good" examples assumed to be non-anomalous
    history = [train_chunks_features]

    orl = OnlineRL(save_prefix='pt1')
    orl.run(output, test_chunks_features, history, transformed_feature_names)
    print(' ')


def main():
    run_pt1_new()
    run_pt2_new()

if __name__ == '__main__':
    main()
