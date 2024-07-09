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

def pt2_fit_rules():
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

def main():
    pt2_fit_rules()
    exit()
    # Load data
    with open('data/pt3_train_chunks_unnormalized.pkl', 'rb') as f:
        train_chunks_unnormalized = pickle.load(f)

    with open('data/pt3_test_chunks_unnormalized.pkl', 'rb') as f:
        test_chunks_unnormalized = pickle.load(f)

    max_temperature = train_chunks_unnormalized[..., 5].max(axis=-1)
    min_current = train_chunks_unnormalized[...,  6].min(axis=-1)
    train_labels = np.zeros((len(max_temperature)))
    
    current_thresh = 0.014
    temperature_thresh = 73.487
    failures = (min_current <= current_thresh).astype(np.int8)
    failures += np.logical_and(min_current > current_thresh, max_temperature > temperature_thresh).astype(np.int8)

    # Load old data
    old_X = np.load('data/pt2_X_tree.npy')
    old_y = np.load('data/pt2_y_tree.npy')
    old_temperature = old_X[..., 22]
    old_currents = old_X[..., 27]

    tn, fp, fn, tp = confusion_matrix(train_labels, failures).ravel()
    best_f1 = f1_score(train_labels, failures)
    print('Evaluated on pt3 train data')
    print(f'tn={tn}, fp={fp}, fn={fn}, tp={tp}')
    print('f1', best_f1)
    print('Starting sampling of parameters')
    rng = np.random.RandomState(123817)
    min_fp = fp

    X_temp = np.concatenate([old_temperature, max_temperature])
    X_current = np.concatenate([old_currents, min_current])
    y = np.concatenate([old_y, train_labels])

    # Train a tree, for checking
    tree = DecisionTreeClassifier(max_depth=2, random_state=12345)
    tree_train_data = np.concatenate([X_temp[:, None], X_current[:, None]], axis=-1)
    tree.fit(tree_train_data, y)
    print(tree.score(tree_train_data, y))
    fig, axs = plt.subplots(1,1)
    plot_tree(tree, ax=axs, feature_names=['Oil_temperature_max', 'Motor_current_min'], class_names=['no failure', 'failure'])
    fig.tight_layout()
    fig.savefig('total_tree.png')
    exit()

    n_samples = 500
    cts = rng.normal(loc=current_thresh, scale=0.01, size=n_samples)
    tts = rng.normal(loc=temperature_thresh, scale=5, size=n_samples)
    for i in tqdm.trange(n_samples):
        ct = current_thresh
        tt = tts[i]
        failures = (X_current <= ct).astype(np.int8)
        failures += np.logical_and(X_current > ct, X_temp > tt).astype(np.int8)

        f1 = f1_score(y, failures)
        if f1 > best_f1:
            best_f1 = f1
            best_ct = ct
            best_tt = tt

        #tn, fp, fn, tp = confusion_matrix(y, failures).ravel()
        #print(f'tn={tn}, fp={fp}, fn={fn}, tp={tp}')
        # if fp < min_fp:
        #     min_fp = fp
        #     best_ct = ct
        #     best_tt = tt
    print(best_ct, best_tt, best_f1)


if __name__ == '__main__':
    main()