import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from src.utils.load import load_basedir


def get_class_count_boost(row0):
    # count
    count_1 = (row0 == 1).astype(int).sum(axis=1)
    count_3 = (row0 == 3).astype(int).sum(axis=1)
    count_5 = (row0 == 5).astype(int).sum(axis=1)
    count_nan = (np.isnan(row0)).astype(int).sum(axis=1)
    vc = np.stack([count_1, count_3, count_5, count_nan]).T / row0.shape[1]
    return vc


def create_features(sg_speed_support, nbrs, k, obj="test"):
    y_mean = []
    y_std = []
    for i in range(len(nbrs)):
        y0 = sg_speed_support[nbrs[i]]
        y_mean.append(np.nanmean(y0, axis=0))
        y_std.append(np.nanstd(y0, axis=0))
    y_mean = np.concatenate(np.stack(y_mean))
    y_std = np.concatenate(np.stack(y_std))
    x_knn = np.concatenate((y_mean, y_std), axis=1)

    np.savez_compressed(
        PROCESSED + "/" + city + "/knn_eng_" + obj + "_p1" + str(k) + "_missing.npz",
        x_knn,
    )


BASEDIR = load_basedir("data")
SPEED_FOLDER = "data/speed_classes"
CACHEDIR = Path("cache")
PROCESSED = "processed"

# load data
city = "london"

print("===Preparing speed features for the samples with high missing rate for london")
# need to run speed_features_fully.py first
sg_speed = np.load(PROCESSED + "/" + city + "/x_sg.npz")["arr_0"]

nan_all = np.load(PROCESSED + "/" + city + "/nan_percent_all.npy")
sg_speed_support = sg_speed[nan_all < 80]
sg_speed_train = sg_speed[nan_all > 80]

x_error, x_correct = pd.read_pickle(PROCESSED + "/" + city + "/error_index.pckl")
x_test = np.load(PROCESSED + "/" + city + "/X_test.npz")["arr_0"][:100]
x_test = np.reshape(x_test, (x_test.shape[0], 4 * x_test.shape[1]))
x_test_error0 = x_test[x_error]
mask = ~np.isnan(x_test_error0[0])

x_test_list = []
for i in range(len(x_test)):
    x_test_list.append(x_test[i][mask])
x_test = np.stack(x_test_list)

x_support = np.load(PROCESSED + "/" + city + "/X_support_missing.npz")["arr_0"]
x_train = np.load(PROCESSED + "/" + city + "/X_train_missing.npz")["arr_0"]

y_support = np.load(PROCESSED + "/" + city + "/y_support_eta_missing.npz")["arr_0"]
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
knn.fit(x_support)

ks = [1, 2, 5, 10, 50]
print("\n- generate knn speed features")
for k in tqdm(ks):
    nbrs = knn.kneighbors(x_test, n_neighbors=k, return_distance=False)
    create_features(sg_speed_support, nbrs, k, "test_speed")

    nbrs = knn.kneighbors(x_train, n_neighbors=k, return_distance=False)
    create_features(sg_speed_support, nbrs, k, "train_speed")
