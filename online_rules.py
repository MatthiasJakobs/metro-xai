import tqdm
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from generate_chunks import load_data
from train_models import ModelTrainer
from failure_detection import simple_lowpass_filter
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import confusion_matrix
from plotting import default_plot
from os.path import exists

MIN_SUPP = 0.999

def construct_features(X, axis=-1):
    # Assume: X.shape = (n_batches, L, n_channels)
    # Do this thing batched to save memory
    batch_size = 1000
    to_return = []
    n_batches = X.shape[0]
    for _ in range(0, n_batches, batch_size):
        avg = np.mean(X[:batch_size], axis=axis, keepdims=True)
        var = np.var(X[:batch_size], axis=axis, keepdims=True)
        mx = np.max(X[:batch_size], axis=axis, keepdims=True)
        mn = np.min(X[:batch_size], axis=axis, keepdims=True)
        to_return.append(np.concatenate([avg, var, mx, mn], axis=axis))
        X = X[batch_size:]

    return np.concatenate(to_return, axis=0)

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

def describe_tree(tree, x, y, feature_names):
    size = compute_size(tree)
    n_wrong, error_rate, support = compute_error_rate(tree, x, y)
    textual_representation = export_text(tree, feature_names=feature_names)
    print('---'*5)
    print('Wrong predictions', n_wrong)
    print('Error rate', error_rate)
    print('Support', support)
    print('Number of leafs', size)
    print(textual_representation)
    print('---'*5)


def compute_size(tree):
    tree = tree.tree_
    leaf_count = 0
    
    for node in range(tree.node_count):
        if tree.children_left[node] == tree.children_right[node]:
            leaf_count += 1
    
    return leaf_count

def compute_error_rate(tree, x, y):
    preds = tree.predict(x)
    tn, fp, fn, tp = confusion_matrix(y, preds, normalize=None).ravel().tolist()
    wrong_predictions = fp + fn
    return wrong_predictions, (fp+fn) / len(y), (tn+tp) / len(y)

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

class BaseHistoryManager:

    def __init__(self, random_state=None):
        self.rng = np.random.RandomState(random_state)

    def add_initial_history(self, initial_history):
        self.history = initial_history
        self.reservoir_length = len(initial_history)
        print('Reservoir size:', self.reservoir_length)

    def add_item(self, t, item):
        if self.should_add(t):
            self.update_history(item)

    def update_history(self, item):
        raise NotImplementedError()

    def should_add(self, t):
        raise NotImplementedError()

    def get_history(self):
        return self.history

# This class just adds every item to the history, as an upper bound on expected performance
class InfiniteHistoryManager(BaseHistoryManager):

    def update_history(self, item):
        self.history.append(item)

    def should_add(self, t):
        return True

class UniformHistoryManager(BaseHistoryManager):
    
    def should_add(self, t):
        # Notation from original paper
        K = self.reservoir_length
        n = t

        # Roll the dice
        p_K = K / (n+K+1)

        return self.rng.rand() >= p_K

    def update_history(self, item):
        # Draw random indice and replace with item
        index = self.rng.randint(self.reservoir_length)
        self.history[index] = item

class ExponentialHistoryManager(UniformHistoryManager):

    def should_add(self, t):
        # Notation from original paper
        K = self.reservoir_length
        beta = 1.1

        # Roll the dice
        p_K = K * (1-np.exp(-1 / (beta * K)))

        return self.rng.rand() >= p_K

class OnlineRL:

    def __init__(self, warning_thresh=0.01, failure_thresh=0.5, random_state=182616, save_prefix='flowmeter'):
        self.warning_thresh = warning_thresh
        self.failure_thresh = failure_thresh
        self.rng = np.random.RandomState(random_state)
        self.save_prefix = save_prefix
        #self.history_manager = UniformHistoryManager(random_state) 
        #self.history_manager = InfiniteHistoryManager(random_state)
        self.history_manager = ExponentialHistoryManager(random_state)

    def run(self, output, X, history, feature_names):
        self.history_manager.add_initial_history(history)

        # Buffer stores the datapoints during warning
        buffer = []
        trees = []

        global_buffer = []

        for t in range(len(X)): 
            is_warning = output[t] > self.warning_thresh
            is_failure = output[t] > self.failure_thresh

            failure_start = output[t] > self.failure_thresh and output[t-1] <= self.failure_thresh
            failure_stop = output[t] <= self.failure_thresh and output[t-1] > self.failure_thresh

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
                #history.append(np.expand_dims(X[t], 0))
                self.history_manager.add_item(t, np.expand_dims(X[t], 0))

            if is_failure:
                X_good = np.concatenate(self.history_manager.get_history())
                #X_good = np.concatenate(history)
                y_good = np.zeros((X_good.shape[0]))
                X_anom = np.concatenate(buffer)
                y_anom = np.ones((X_anom.shape[0]))
                _x, _y = np.concatenate([X_good, X_anom]), np.concatenate([y_good, y_anom])
                _x = _x.reshape(_x.shape[0], -1)

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
                 for tree in trees:
                    describe_tree(tree, _x, _y, feature_names)
                 buffer = []
                 trees = []

def run_pt2_new():
    model = ModelTrainer(f'configs/PT2_TCN.json').fit()

    print('Load data (MetroPT2)')

    # Start with unnormalized chunks first to save memory
    with open('data/pt2_train_chunks_unnormalized.pkl', 'rb') as f:
        train_chunks_unnormalized = pickle.load(f)
    train_chunks_features = construct_features(train_chunks_unnormalized, axis=1).swapaxes(1,2)
    del train_chunks_unnormalized

    with open('data/pt2_test_chunks_unnormalized.pkl', 'rb') as f:
        test_chunks_unnormalized = pickle.load(f)
    test_chunks_features = construct_features(test_chunks_unnormalized, axis=1).swapaxes(1,2)
    del test_chunks_unnormalized

    if not exists('data/pt2_train_errors.npy') or not exists('data/pt2_test_errors.npy'):
        print('Calculate model outputs')
        train_chunks, _, test_chunks, _ = load_data(version=model.version, scaler=model.scaler)
        val_size = int(0.3 * len(train_chunks))
        train_errors = model.calc_loss(train_chunks[-val_size:], train_chunks[-val_size:], average=False).mean(axis=(1,2))
        test_errors = model.calc_loss(test_chunks, test_chunks, average=False).mean(axis=(1,2))
        np.save('data/pt2_train_errors.npy', train_errors)
        np.save('data/pt2_test_errors.npy', test_errors)
    else:
        train_errors = np.load('data/pt2_train_errors.npy')
        test_errors = np.load('data/pt2_test_errors.npy')

    print('done loading data')

    alpha = 0.15
    anom = np.quantile(train_errors, q=0.99) * 3
    binary_output = (test_errors > anom).astype(np.int8)
    output = simple_lowpass_filter(binary_output,alpha)

    channel_names = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Flowmeter', 'Motor_current', 'COMP']

    transformed_feature_names = [[fname+'_mean', fname+'_var', fname+'_max', fname+'_min'] for fname in channel_names]
    transformed_feature_names = sum(transformed_feature_names, [])

    # History stores the "good" examples assumed to be non-anomalous
    history = list(np.expand_dims(train_chunks_features, 1))

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
    history = list(np.expand_dims(train_chunks_features[:, feature_indices], 1))
    transformed_feature_names = [tfn for tfn in transformed_feature_names if 'Flowmeter' not in tfn]

    orl = OnlineRL(save_prefix='noflowmeter')
    orl.run(output, test_chunks_features[:, feature_indices], history, transformed_feature_names)
    print(' ')

def run_pt1_new():
    model = ModelTrainer(f'configs/PT1_TCN.json').fit()

    print('Load data (MetroPT1)')
    with open('data/pt1_train_chunks_unnormalized.pkl', 'rb') as f:
        train_chunks_unnormalized = pickle.load(f)
    train_chunks_features = construct_features(train_chunks_unnormalized, axis=1).swapaxes(1,2)
    del train_chunks_unnormalized

    with open('data/pt1_test_chunks_unnormalized.pkl', 'rb') as f:
        test_chunks_unnormalized = pickle.load(f)
    test_chunks_features = construct_features(test_chunks_unnormalized, axis=1).swapaxes(1,2)
    del test_chunks_unnormalized

    if not exists('data/pt1_train_errors.npy') or not exists('data/pt1_test_errors.npy'):
        train_chunks, _, test_chunks, _ = load_data(version=model.version, scaler=model.scaler)
        val_size = int(0.3 * len(train_chunks))
        train_errors = model.calc_loss(train_chunks[-val_size:], train_chunks[-val_size:], average=False).mean(axis=(1,2))
        test_errors = model.calc_loss(test_chunks, test_chunks, average=False).mean(axis=(1,2))
        np.save('data/pt1_train_errors.npy', train_errors)
        np.save('data/pt1_test_errors.npy', test_errors)
    else:
        train_errors = np.load('data/pt1_train_errors.npy')
        test_errors = np.load('data/pt1_test_errors.npy')

    channel_names = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Flowmeter', 'Motor_current', 'COMP']

    transformed_feature_names = [[fname+'_mean', fname+'_var', fname+'_max', fname+'_min'] for fname in channel_names]
    transformed_feature_names = sum(transformed_feature_names, [])

    alpha = 0.1
    anom = np.quantile(train_errors, q=0.99) * 2
    binary_output = (test_errors > anom).astype(np.int8)
    output = simple_lowpass_filter(binary_output,alpha)

    # History stores the "good" examples assumed to be non-anomalous
    history = list(np.expand_dims(train_chunks_features, 1))

    orl = OnlineRL(save_prefix='pt1')
    orl.run(output, test_chunks_features, history, transformed_feature_names)
    print(' ')


def main(version=2):
    if version == 1:
        run_pt1_new()
    if version == 2:
        run_pt2_new()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run AMRules extraction')
    parser.add_argument('--version', '-v', type=int, choices=[1,2], default=2, help='PT version (1 or 2)')
    args = parser.parse_args()
    main(version=args.version)
