import numpy as np
import pandas as pd
import pickle
import random
from pathlib import Path

from src.utils.load import load_basedir
from src.feature_extraction.read_knn_features import (
    read_knn_features,
    read_speed_features,
)
from src.feature_extraction.feature_combination import feature_combine

BASEDIR = load_basedir("data")
CACHEDIR = Path("cache")
PROCESSED = "processed"
SAVE_PROCESSED = "processed/test_outputs_eta"

city = "london"
num_t_per_day = 64
num_val = 0

p_knn = "p1"
p_speed = "p1"

x_static = pd.read_parquet(PROCESSED + "/" + city + "/x_static_eta.parquet")
x_static["sg_id"] = x_static.index
x_static = x_static.to_numpy()
x_allnn = np.load(PROCESSED + "/" + city + "/knn_eng_allnn_missing.npz")["arr_0"]
x_static = np.concatenate([x_static, x_allnn], axis=1)  # add all nn as a feature

y = np.load(PROCESSED + "/" + city + "/y_train_eta_missing.npz")["arr_0"]

x_nodes = np.load(PROCESSED + "/" + city + "/x_nodes_eta_missing.npz")["arr_0"]
x_static_train = np.repeat(x_static[None, :], len(y), axis=0)
# knn features
ks = [2, 5, 10, 30, 50, 100]
x_knn = read_knn_features(PROCESSED, city, ks, "train", y.shape, p_knn, missing=True)
# supersegment speed statistics
ks = [1, 2, 5, 10, 50]
x_sg_speed = read_speed_features(
    PROCESSED, city, ks, "train", y.shape, p_speed, missing=True
)

x_all = np.concatenate((x_knn, x_static_train, x_nodes, x_sg_speed), axis=2)
del x_nodes, x_static_train, x_knn, x_sg_speed

random.seed(11)
idx_train = random.sample(range(len(x_all)), len(x_all) - num_val)
x_train = x_all[idx_train]
y_train = y[idx_train]

# x_val = x_all[[i for i in range(len(x_all)) if i not in idx_train]]
# y_val = y[[i for i in range(len(x_all)) if i not in idx_train]]

del x_all

x_train = np.concatenate(x_train)
y_train = np.concatenate(y_train)

# x_val = np.concatenate(x_val)
# y_val = np.concatenate(y_val)

x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)
# x_val = pd.DataFrame(x_val)
# y_val = pd.DataFrame(y_val)

y_cols = []
ks = [2, 5, 10, 30, 50, 100]
for k in ks:
    for f in ["mean", "std", "25", "50", "75", "min", "max"]:
        y_cols.append(f + "_" + str(k))

speed_cols = []
ks = [1, 2, 5, 10, 50]
for k in ks:
    for metric in [
        "median_speed",
        "ffs",
        "vol_class_1",
        "vol_class_3",
        "vol_class_5",
        "vol_class_nan",
    ]:
        for f in ["mean", "std"]:
            speed_cols.append(metric + "_" + f + "_" + str(k))

static_cols = [
    "n_nodes",
    "length",
    "n_oneway",
    "sl_mean",
    "sl_std",
    "sl_25",
    "sl_50",
    "sl_75",
    "sl_min",
    "sl_max",
    "haversine",
    "shortest_tt",
    "mean_all",
    "std_all",
    "25_all",
    "50_all",
    "75_all",
    "min_all",
    "max_all",
    "thre_tt_0",
    "thre_tt_1",
    "thre_tt_2",
    "sg_id",
]

dynamic_cols = []
for i in [1, 2, 3, 4]:
    for f in ["node_vol_sum", "node_vol_mean", "node_vol_std", "node_valid_num"]:
        dynamic_cols.append(f + "_" + str(i))

col_name = y_cols + static_cols + dynamic_cols + speed_cols

# x_train.columns,x_val.columns = col_name,col_name
x_train.columns = col_name

x_nodes_test = np.load(PROCESSED + "/" + city + "/x_nodes_test_eta.npz")["arr_0"]
x_static_test = np.repeat(x_static[None, :], 100, axis=0)

ks = [2, 5, 10, 30, 50, 100]
x_knn_test = read_knn_features(
    PROCESSED, city, ks, "test", y.shape, p_knn, missing=True
)
# supersegment speed statistics
ks = [1, 2, 5, 10, 50]
x_sg_speed = read_speed_features(
    PROCESSED, city, ks, "test", y.shape, p_speed, missing=True
)

x_test = np.concatenate((x_knn_test, x_static_test, x_nodes_test, x_sg_speed), axis=2)
del x_nodes_test, x_static_test, x_knn_test, x_static, x_sg_speed

x_error, x_correct = pd.read_pickle(PROCESSED + "/london/error_index.pckl")
x_test = x_test[x_error]

x_test = np.concatenate(x_test)
# x_test = np.nan_to_num(x_test, nan=0)
x_test = pd.DataFrame(x_test)
x_test.columns = col_name

f = open(PROCESSED + "/london/missing_df_" + p_knn + ".pckl", "wb")
# pickle.dump([x_train, y_train, x_val, y_val, x_test, weights], f)
pickle.dump([x_train, y_train, x_test], f)
f.close()
