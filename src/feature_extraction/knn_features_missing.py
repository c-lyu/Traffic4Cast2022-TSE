import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from src.utils.load import cfg

PROCESSED = cfg["PROCESSED"]


def create_features(y_train, nbrs, k, p, obj="test"):
    y_mean = []
    y_std = []
    y_25 = []
    y_50 = []
    y_75 = []
    y_min = []
    y_max = []
    for i in range(len(nbrs)):
        if k == "all":
            y0 = y_train
        else:
            y0 = y_train[nbrs[i]]
        y_mean.append(np.mean(y0, axis=0))
        y_std.append(np.std(y0, axis=0))
        y_25.append(np.quantile(y0, 0.25, axis=0))
        y_50.append(np.quantile(y0, 0.5, axis=0))
        y_75.append(np.quantile(y0, 0.75, axis=0))
        y_min.append(np.min(y0, axis=0))
        y_max.append(np.max(y0, axis=0))
    y_mean = np.concatenate(np.stack(y_mean))
    y_std = np.concatenate(np.stack(y_std))
    y_25 = np.concatenate(np.stack(y_25))
    y_50 = np.concatenate(np.stack(y_50))
    y_75 = np.concatenate(np.stack(y_75))
    y_min = np.concatenate(np.stack(y_min))
    y_max = np.concatenate(np.stack(y_max))
    x_knn = np.array([y_mean, y_std, y_25, y_50, y_75, y_min, y_max]).T

    np.savez_compressed(
        PROCESSED / f"london/knn_eng_{obj}_p{p}{k}_missing.npz",
        x_knn,
    )


# load data
x_error, x_correct = pd.read_pickle(PROCESSED / "london/error_index.pckl")
x_test = np.load(PROCESSED / "london/X_test.npz")["arr_0"][:100]
x_test = np.reshape(x_test, (x_test.shape[0], 4 * x_test.shape[1]))
x_test_error0 = x_test[x_error]
mask = ~np.isnan(x_test_error0[0])

x_test_list = []
for i in range(len(x_test)):
    x_test_list.append(x_test[i][mask])
x_test = np.stack(x_test_list)

x_support = np.load(PROCESSED / "london/X_support_missing.npz")["arr_0"]
x_train = np.load(PROCESSED / "london/X_train_missing.npz")["arr_0"]

y_support = np.load(PROCESSED / "london/y_support_eta_missing.npz")["arr_0"]
y_support = np.reshape(y_support, (len(x_support), -1))

x_support = np.reshape(x_support, (x_support.shape[0], 4 * x_support.shape[1]))
x_support = np.nan_to_num(x_support, nan=0)
x_train = np.reshape(x_train, (x_train.shape[0], 4 * x_train.shape[1]))
x_train = np.nan_to_num(x_train, nan=0)

x_support_list = []
for i in range(len(x_support)):
    x_support_list.append(x_support[i][mask])
x_support = np.stack(x_support_list)

x_train_list = []
for i in range(len(x_train)):
    x_train_list.append(x_train[i][mask])
x_train = np.stack(x_train_list)

x_support = np.nan_to_num(x_support, nan=0)
x_train = np.nan_to_num(x_train, nan=0)
x_test = np.nan_to_num(x_test, nan=0)

knn = NearestNeighbors(p=1, n_jobs=-1)
knn.fit(x_support)  # to avoid info leakage in the validation set
# calculate the distance between observations
ks = [2, 5, 10, 30, 50, 100]
print("- generate knn y_eta features")
for k in tqdm(ks):
    nbrs = knn.kneighbors(x_test, n_neighbors=k, return_distance=False)
    create_features(y_support, nbrs, k, 1, "test")

    nbrs = knn.kneighbors(x_train, n_neighbors=k, return_distance=False)
    create_features(y_support, nbrs, k, 1, "train")

print("\n- generate allnn y_eta features")
y_mean = np.mean(y_support, axis=0)
y_std = np.std(y_support, axis=0)
y_25 = np.quantile(y_support, 0.25, axis=0)
y_50 = np.quantile(y_support, 0.5, axis=0)
y_75 = np.quantile(y_support, 0.75, axis=0)
y_min = np.min(y_support, axis=0)
y_max = np.max(y_support, axis=0)

y_allnn = np.stack((y_mean, y_std, y_25, y_50, y_75, y_min, y_max), axis=1)
np.savez_compressed(PROCESSED / "london/knn_eng_allnn_missing.npz", y_allnn)
