import numpy as np
from model_utilities import *
from train_CLL import train_algorithm

def generate_synthetic_data():
    """ Generate synthetic data """

    np.random.seed(900)
    n  = 20000
    d  = 200
    m = 10
    Ys = 2 * np.random.randint(2, size=(n,)) - 1

    feature_accs = 0.2 * np.random.random((d,)) + 0.5
    train_data = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            if np.random.random() > feature_accs[j]:
                train_data[i,j] = 1 if Ys[i] == 1 else 0
            else:
                train_data[i,j] = 0 if Ys[i] == 1 else 1


    # Initialize the weak signals
    ws_accs = 0.1 * np.random.random((m,)) + 0.6
    ws_coverage = 0.3
    Ws = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if np.random.random() < ws_coverage:
                Ws[i,j] = Ys[i] if np.random.random() < ws_accs[j] else -Ys[i]

    # Convert weak_signals to correct format
    weak_signals = Ws.copy()
    weak_signals[Ws==0] = -1
    weak_signals[Ws==-1] = 0

    n,m = weak_signals.shape
    weak_signals = np.expand_dims(weak_signals.T, axis=-1)

    # Convert Y and weak_signals to correct format
    train_labels = 0.5 * (Ys + 1)

    indexes = np.arange(n)
    np.random.seed(2000)
    test_indexes = np.random.choice(n, int(n * 0.2), replace=False)
    weak_signals = np.delete(weak_signals, test_indexes, axis=1)

    test_labels = train_labels[test_indexes]
    test_data = train_data[test_indexes]
    train_indexes = np.delete(indexes, test_indexes)
    train_labels = train_labels[train_indexes]
    train_data = train_data[train_indexes]

    data = {}
    data['train'] = train_data, train_labels
    data['test'] = test_data, test_labels
    data['weak_signals'] = weak_signals

    return data


def run_experiment(dataset, true_bound=False):
    """ Run CLL experiments """

    batch_size = 32
    train_data, train_labels = dataset['train']
    test_data, test_labels = dataset['test']
    weak_signals = dataset['weak_signals']
    m, n, k = weak_signals.shape

    weak_errors = np.ones((m, k)) * 0.01

    if true_bound:
        weak_errors = get_error_bounds(train_labels, weak_signals)
        weak_errors = np.asarray(weak_errors)

    # Set up the constraints
    constraints = set_up_constraint(weak_signals, weak_errors)
    constraints['weak_signals'] = weak_signals
    mv_labels = majority_vote_signal(weak_signals)

    y = train_algorithm(constraints)
    accuracy = accuracy_score(train_labels, y)
    model = mlp_model(train_data.shape[1], k)
    model.fit(train_data, y, batch_size=batch_size, epochs=20, verbose=1)
    test_predictions = model.predict(test_data)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    print("CLL Label accuracy is: ", accuracy)
    print("CLL Test accuracy is: \n", test_accuracy)
    print("Majority vote accuracy is: ", accuracy_score(train_labels, mv_labels))


run_experiment(generate_synthetic_data())
# run_experiment(read_text_data('../datasets/sst-2/'))
# run_experiment(read_text_data('../datasets/imdb/'))
# run_experiment(read_text_data('../datasets/yelp/'))
