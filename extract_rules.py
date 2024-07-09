import pickle
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

def construct_features(X, axis=-1):
    # Assume: X.shape = (batch_size, L, n_channels)
    avg = np.mean(X, axis=axis, keepdims=True)
    var = np.var(X, axis=axis, keepdims=True)
    mx = np.max(X, axis=axis, keepdims=True)
    mn = np.min(X, axis=axis, keepdims=True)
    return np.concatenate([avg, var, mx, mn], axis=axis)

def fit_rules():
    with open('data/pt2_test_chunks_unnormalized.pkl', 'rb') as f:
        test_chunks_unnormalized = pickle.load(f)
    with open('data/pt2_train_chunks_unnormalized.pkl', 'rb') as f:
        train_chunks_unnormalized = pickle.load(f)
    
    print('=== Find rules using all available channels ===')

    X = construct_features(test_chunks_unnormalized, axis=1).swapaxes(1, 2)
    n_batches = X.shape[0]
    X = X.reshape(n_batches, -1)
    correct_alerts = np.load('data/pt2_correct_alerts.npy')
    y = correct_alerts.copy()

    channel_names = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Flowmeter', 'Motor_current', 'COMP']
    transformed_feature_names = [[fname+'_mean', fname+'_var', fname+'_max', fname+'_min'] for fname in channel_names]
    transformed_feature_names = sum(transformed_feature_names, [])

    # Transform data
    tree = DecisionTreeClassifier(max_depth=1, random_state=192781)
    tree.fit(X, y)
    print('train score (correct alerts)', tree.score(X, y))
    fig, axs = plt.subplots(1,1)
    plot_tree(tree, ax=axs, feature_names=transformed_feature_names, class_names=['no failure', 'failure'])
    fig.tight_layout()
    fig.savefig('plots/tree.png')

    # Make sure this works on the real training part
    X = construct_features(train_chunks_unnormalized, axis=1).swapaxes(1, 2)
    n_batches = X.shape[0]
    X = X.reshape(n_batches, -1)
    y = np.zeros((n_batches))
    print('score on first two months', tree.score(X, y))

    # Alternative based on other features
    print('=== Alternative, do not use Flowmeter to save costs ===')

    all_chunks = np.concatenate([test_chunks_unnormalized, train_chunks_unnormalized], axis=0)
    X = construct_features(all_chunks, axis=1).swapaxes(1, 2)
    X = X[:, [0, 1, 2, 3, 4, 5, 7, 8]]
    n_batches = X.shape[0]
    X = X.reshape(n_batches, -1)
    y = np.concatenate([correct_alerts, np.zeros((len(train_chunks_unnormalized)))])

    transformed_feature_names = [[fname+'_mean', fname+'_var', fname+'_max', fname+'_min'] for fname in channel_names if fname != 'Flowmeter']
    transformed_feature_names = sum(transformed_feature_names, [])

    tree = DecisionTreeClassifier(max_depth=2, random_state=192781)
    tree.fit(X, y)
    print('max depth 2', tree.score(X, y))
    fig, axs = plt.subplots(1,1)
    plot_tree(tree, ax=axs, feature_names=transformed_feature_names, class_names=['no failure', 'failure'])
    fig.tight_layout()
    fig.savefig('plots/alternative_rule_simple.png')

    tree = DecisionTreeClassifier(max_depth=3, random_state=192781)
    tree.fit(X, y)
    print('max depth 3', tree.score(X, y))
    fig, axs = plt.subplots(1,1)
    plot_tree(tree, ax=axs, feature_names=transformed_feature_names, class_names=['no failure', 'failure'])
    fig.tight_layout()
    fig.savefig('plots/alternative_rule.png')

    print('=== Check transferability to MetroPT3 ===')
    with open('data/pt3_train_chunks_unnormalized.pkl', 'rb') as f:
        train_chunks_unnormalized = pickle.load(f)
    
    # First, reconstruct PT2 features without Flowmeter and COMP
    X = construct_features(all_chunks, axis=1).swapaxes(1, 2)
    X = X[:, [0, 1, 2, 3, 4, 5, 7]]
    n_batches = X.shape[0]
    X = X.reshape(n_batches, -1)

    # Generate PT3 features
    _X = construct_features(train_chunks_unnormalized, axis=1).swapaxes(1, 2)
    n_batches = _X.shape[0]
    _X = _X.reshape(n_batches, -1)
    _y = np.zeros((len(_X)))

    X = np.concatenate([X, _X], axis=0)
    y = np.concatenate([y, _y], axis=0)

    transformed_feature_names = [[fname+'_mean', fname+'_var', fname+'_max', fname+'_min'] for fname in channel_names if fname != 'Flowmeter' and fname != 'COMP']
    transformed_feature_names = sum(transformed_feature_names, [])

    tree = DecisionTreeClassifier(max_depth=4, random_state=192781)
    tree.fit(X, y)
    print('max depth 4', tree.score(X, y))
    fig, axs = plt.subplots(1,1)
    plot_tree(tree, ax=axs, feature_names=transformed_feature_names, class_names=['no failure', 'failure'])
    fig.tight_layout()
    fig.savefig('plots/transfer_large.png')

    # TODO: Restrict to two features, since tree came out different. But maybe using mean of oil_temp is better than max?
    X = X[..., [22, 27]]
    transformed_feature_names = ['Oil_temperature_max', 'Motor_current_min']

    tree = DecisionTreeClassifier(max_depth=2, random_state=192781)
    tree.fit(X, y)
    print('max depth 2', tree.score(X, y))
    fig, axs = plt.subplots(1,1)

    plot_tree(tree, ax=axs, feature_names=transformed_feature_names, class_names=['no failure', 'failure'])
    fig.tight_layout()
    fig.savefig('plots/transfer_rule.png')


def main():
    fit_rules()

if __name__ == '__main__':
    main()