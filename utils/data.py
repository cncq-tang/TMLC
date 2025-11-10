import numpy as np
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as scsp


def read_mymat(path, name, sp, sparse=False):
    """
    :param path: dataset path
    :param name: dataset name
    :param sp: sp-['X', 'Y']
    :param sparse:
    :return: X, Y, n_sample, n_view, n_class, dims
    """
    mat_file = path + name
    f = scio.loadmat(mat_file)

    if 'X' in sp:
        Xa = f['X']
        Xa = Xa.reshape(Xa.shape[1], )
        X = []
        if sparse:
            for x in Xa:
                X.append(scsp.csc_matrix(x).toarray().astype(np.float32))
        else:
            for x in Xa:
                X.append(x.astype(np.float32))
    else:
        X = None

    if 'Y' in sp:
        if name == 'HandWritten.mat':
            Y = (f['Y']).astype(np.int64)
        else:
            Y = (f['Y'] - 1).astype(np.int64)
        Y = np.squeeze(Y)
    else:
        Y = None

    n_sample = X[0].shape[0]
    n_view = len(X)
    n_class = len(np.unique(Y))
    dims = []
    for view in range(n_view):
        dims.append([X[view].shape[1]])
    return X, Y, n_sample, n_view, n_class, dims


def build_normal_dataset(Y, ratio, seed=999):
    np.random.seed(seed=seed)
    Y_idx = np.array([idx for idx in range(len(Y))])
    num_normal_train = np.int_(np.ceil(len(Y_idx) * ratio))
    train_idx_idx = np.random.choice(len(Y_idx), num_normal_train, replace=False)
    train_idx = Y_idx[train_idx_idx]
    test_idx = np.array(list(set(Y_idx.tolist()) - set(train_idx.tolist())))
    partition = {'train': train_idx, 'test': test_idx}
    return partition


def build_imbalanced_dataset(Y, ratio, num_classes, seed=999):
    np.random.seed(seed=seed)
    Y_idx = np.array([idx for idx in range(len(Y))])
    num_normal_train = np.int_(np.ceil(len(Y_idx) * ratio))
    train_idx = np.random.choice(len(Y_idx), num_normal_train, replace=False)
    test_idx = np.array(list(set(Y_idx.tolist()) - set(train_idx.tolist())))

    class_counts = []
    train_labels = Y[train_idx]
    for k in range(num_classes):
        class_idx = train_idx[train_labels == k]
        class_counts.append(class_idx)

    N_0 = len(class_counts[0])
    decay_rate = 0.30
    samples_nums = [max(1, int(N_0 * np.exp(-decay_rate * k))) for k in range(num_classes)]

    imbalanced_train_idx = []
    for k in range(num_classes):
        if len(class_counts[k]) > samples_nums[k]:
            k_idx = np.random.choice(class_counts[k], size=samples_nums[k], replace=False)
            imbalanced_train_idx.extend(k_idx)
        else:
            imbalanced_train_idx.extend(class_counts[k])
    imbalanced_train_idx = np.array(imbalanced_train_idx, dtype=np.int64)
    partition = {'train': imbalanced_train_idx, 'test': test_idx}
    imbalanced_idx_label = Y[imbalanced_train_idx]
    imbalanced_idx_counts = np.bincount(imbalanced_idx_label)
    return partition


def normalize(x, min=0):
    if min == 0:
        scaler = MinMaxScaler((0, 1))
    else:
        scaler = MinMaxScaler((-1, 1))

    norm_x = scaler.fit_transform(x)
    return norm_x


def postprocessing(X_test, Y_test, num_classes, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False,
                   ratio_conflict=0.5):
    test_samples = X_test[0].shape[0]
    index = np.arange(test_samples)
    num_views = len(X_test)
    if addNoise:
        selects = np.random.choice(index, size=int(ratio_noise * len(index)), replace=False)
        for i in selects:
            views = np.random.choice(np.array(num_views), size=np.random.randint(num_views), replace=False)
            for v in views:
                X_test[v][i] = np.random.normal(X_test[v][i], sigma)
    if addConflict:
        records = dict()
        for c in range(num_classes):
            i = np.where(Y_test == c)[0][0]
            temp = dict()
            for v in range(num_views):
                temp[v] = X_test[v][i]
            records[c] = temp
        selects = np.random.choice(index, size=int(ratio_conflict * len(index)), replace=False)
        for i in selects:
            v = np.random.randint(num_views)
            X_test[v][i] = records[(Y_test[i] + 1) % num_classes][v]
