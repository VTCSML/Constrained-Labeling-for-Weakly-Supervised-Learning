import numpy as np
import codecs
import json
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense


def calculate_bounds(true_labels, predicted_labels, mask=None):
    """ Calculate error rate on data points the weak signals label """

    if len(true_labels.shape) == 1:
        predicted_labels = predicted_labels.ravel()
    assert predicted_labels.shape == true_labels.shape

    if mask is None:
        mask = np.ones(predicted_labels.shape)
    if len(true_labels.shape) == 1:
        mask = mask.ravel()

    error_rate = true_labels*(1-predicted_labels) + \
        predicted_labels*(1-true_labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        error_rate = np.sum(error_rate*mask, axis=0) / np.sum(mask, axis=0)
        error_rate = np.nan_to_num(error_rate)

    # check results are scalars
    if np.isscalar(error_rate):
        error_rate = np.asarray([error_rate])
    return error_rate


def get_error_bounds(true_labels, weak_signals):
    """ Get error bounds of the weaks signals
        returns a list of size num_weak x num_classes
    """
    error_bounds = []
    mask = weak_signals >= 0

    for i, weak_probs in enumerate(weak_signals):
        active_mask = mask[i]
        error_rate = calculate_bounds(true_labels, weak_probs, active_mask)
        error_bounds.append(error_rate)
    return error_bounds


def build_constraints(a_matrix, bounds):
    """ params:
        a_matrix left hand matrix of the inequality size: num_weak x num_data x num_class type: ndarray
        bounds right hand vectors of the inequality size: num_weak x num_data type: ndarray
        return:
        dictionary containing constraint vectors
    """

    m, n, k = a_matrix.shape
    assert (m, k) == bounds.shape, \
        "The constraint matrix shapes don't match"

    constraints = dict()
    constraints['A'] = a_matrix
    constraints['b'] = bounds
    return constraints


def set_up_constraint(weak_signals, error_bounds):
    """ Set up error constraints for A and b matrices """

    constraint_set = dict()
    m, n, k = weak_signals.shape
    precision_amatrix = np.zeros((m, n, k))
    error_amatrix = np.zeros((m, n, k))
    constants = []

    for i, weak_signal in enumerate(weak_signals):
        active_signal = weak_signal >= 0
        precision_amatrix[i] = -1 * weak_signal * active_signal / \
            (np.sum(active_signal*weak_signal, axis=0) + 1e-8)
        error_amatrix[i] = (1 - 2 * weak_signal) * active_signal

        # error denom to check abstain signals
        error_denom = np.sum(active_signal, axis=0)
        error_amatrix[i] /= error_denom

        # constants for error constraints
        constant = (weak_signal*active_signal) / error_denom
        constants.append(constant)

    # set up error upper bounds constraints
    constants = np.sum(constants, axis=1)
    assert len(constants.shape) == len(error_bounds.shape)
    bounds = error_bounds - constants
    error_set = build_constraints(error_amatrix, bounds)
    constraint_set['error'] = error_set

    return constraint_set


def accuracy_score(y_true, y_pred):
    """ Calculate accuracy of the model """

    try:
        n, k = y_true.shape
        if k > 1:
            assert y_true.shape == y_pred.shape
            return np.mean(np.equal(np.argmax(y_true, axis=-1),
                                    np.argmax(y_pred, axis=-1)))
    except:
        if len(y_true.shape) == 1:
            y_pred = np.round(y_pred.ravel())

    assert y_true.shape == y_pred.shape
    return np.mean(np.equal(y_true, np.round(y_pred)))


def prepare_mmce(weak_signals, labels):
    """ Convert weak_signals to format for mmce """

    crowd_labels = np.zeros(weak_signals.shape)
    true_labels = labels.copy()
    try:
        n, k = true_labels.shape
    except:
        k = 1
    crowd_labels[weak_signals == 1] = 2
    crowd_labels[weak_signals == 0] = 1
    if k > 1:
        true_labels = np.argmax(true_labels, axis=1)
    true_labels += 1

    if len(crowd_labels.shape) > 2:
        assert crowd_labels.any() != 0
        m, n, k = crowd_labels.shape
        if k > 1:
            for i in range(k):
                crowd_labels[:, :, i] = i+1
            crowd_labels[weak_signals == -1] = 0
        crowd_labels = crowd_labels.transpose((1, 0, 2))
        crowd_labels = crowd_labels.reshape(n, m*k)
    return crowd_labels.astype(int), true_labels.ravel().astype(int)


def read_text_data(datapath):
    """ Read text datasets """

    train_data = np.load(datapath + 'data_features.npy', allow_pickle=True)[()]
    weak_signals = np.load(datapath + 'weak_signals.npy', allow_pickle=True)[()]
    train_labels = np.load(datapath + 'data_labels.npy', allow_pickle=True)[()]
    test_data = np.load(datapath +'test_features.npy', allow_pickle=True)[()]
    test_labels = np.load(datapath + 'test_labels.npy', allow_pickle=True)[()]

    if len(weak_signals.shape) == 2:
        weak_signals = np.expand_dims(weak_signals.T, axis=-1)

    data = {}
    data['train'] = train_data, train_labels
    data['test'] = test_data, test_labels
    data['weak_signals'] = weak_signals
    return data


def majority_vote_signal(weak_signals):
    """ Calculate majority vote labels for the weak_signals"""

    baseline_weak_labels = np.rint(weak_signals)
    mv_weak_labels = np.ones(baseline_weak_labels.shape)
    mv_weak_labels[baseline_weak_labels == -1] = 0
    mv_weak_labels[baseline_weak_labels == 0] = -1
    mv_weak_labels = np.sign(np.sum(mv_weak_labels, axis=0))
    break_ties = np.random.randint(2, size=int(np.sum(mv_weak_labels == 0)))
    mv_weak_labels[mv_weak_labels == 0] = break_ties
    mv_weak_labels[mv_weak_labels == -1] = 0
    return mv_weak_labels


def mlp_model(dimension, output):
    """ Simple MLP model"""

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(dimension,)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adagrad', metrics=['accuracy'])

    return model
