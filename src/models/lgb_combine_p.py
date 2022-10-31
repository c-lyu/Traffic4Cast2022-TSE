import gc
import random
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.feature_extraction.read_knn_features import (
    read_knn_features,
    read_speed_features,
)
from src.feature_extraction.feature_combination import feature_combine
from src.utils.load import cfg

PROCESSED = cfg["PROCESSED"]
SAVE_PROCESSED = PROCESSED / "test_outputs_eta"

cities = ["london"]
num_t_per_day = 64
num_val = 500

loss_train = []
loss_val = []
for city in cities:
    x_static = pd.read_parquet(PROCESSED / city / "x_static_eta.parquet")
    x_static["sg_id"] = x_static.index
    x_static = x_static.to_numpy()
    x_allnn = np.load(PROCESSED / city / "knn_eng_allnn.npz")["arr_0"]
    x_static = np.concatenate([x_static, x_allnn], axis=1)  # add all nn as a feature

    y = np.load(PROCESSED / city / "y_eta.npz")["arr_0"]
    y = np.reshape(y, (-1, len(x_static)))
    y_hist_mean = np.mean(y, axis=0)
    # y = y - y_hist_mean

    x_nodes = np.load(PROCESSED / city / "x_nodes_eta.npz")["arr_0"]
    # knn features
    ks = [5, 10, 30, 50, 100]
    x_knn1 = read_knn_features(PROCESSED, city, ks, "train", y.shape, p="p1")
    # p2 knn features
    x_knn2 = read_knn_features(PROCESSED, city, ks, "train", y.shape, p="p2")

    x_static_train = np.repeat(x_static[None, :], len(y), axis=0)
    # supersegment speed statistics
    ks = [5, 10, 50]
    x_sg_speed = read_speed_features(PROCESSED, city, ks, "train", y.shape, p="p1")

    x_all = np.concatenate(
        (x_knn1, x_knn2, x_static_train, x_nodes, x_sg_speed), axis=2
    )
    del x_nodes, x_static_train, x_knn1, x_knn2, x_sg_speed, x_allnn
    gc.collect()

    if city == "london":
        num_nan_raw = np.load(PROCESSED / city / "nan_percent_all.npy")
        x_all = x_all[num_nan_raw < 80]
        y = y[num_nan_raw < 80]

    random.seed(11)
    idx_train = random.sample(range(len(x_all)), len(x_all) - num_val)
    x_train = x_all[idx_train]
    y_train = y[idx_train]

    x_val = x_all[[i for i in range(len(x_all)) if i not in idx_train]]
    y_val = y[[i for i in range(len(x_all)) if i not in idx_train]]

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_val = np.concatenate(x_val)
    y_val = np.concatenate(y_val)

    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    x_val = pd.DataFrame(x_val)
    y_val = pd.DataFrame(y_val)

    del x_all
    gc.collect()

    y_cols1 = []
    for k in [5, 10, 30, 50, 100]:
        for f in ["mean", "std", "25", "50", "75", "min", "max"]:
            y_cols1.append(f + "_" + str(k))

    y_cols2 = []
    for k in [5, 10, 30, 50, 100]:
        for f in ["p2_mean", "p2_std", "p2_25", "p2_50", "p2_75", "p2_min", "p2_max"]:
            y_cols2.append(f + "_" + str(k))

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

    speed_cols = []
    for k in [5, 10, 50]:
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

    col_name = y_cols1 + y_cols2 + static_cols + dynamic_cols + speed_cols

    x_train.columns, x_val.columns = col_name, col_name

    x_static_test = np.repeat(x_static[None, :], 100, axis=0)
    x_nodes_test = np.load(PROCESSED / city / "x_nodes_test_eta.npz")["arr_0"]

    ks = [5, 10, 30, 50, 100]
    x_knn_test1 = read_knn_features(
        PROCESSED, city, ks, "test", y.shape, p="p1"
    )
    x_knn_test2 = read_knn_features(
        PROCESSED, city, ks, "test", y.shape, p="p2"
    )

    # supersegment speed statistics
    ks = [5, 10, 50]
    x_sg_speed = read_speed_features(PROCESSED, city, ks, "test", y.shape, p="p1")

    x_test = np.concatenate(
        (x_knn_test1, x_knn_test2, x_static_test, x_nodes_test, x_sg_speed), axis=2
    )
    del x_nodes_test, x_static_test, x_knn_test1, x_knn_test2, x_static, x_sg_speed, y
    gc.collect()

    x_test = np.concatenate(x_test)
    x_test = pd.DataFrame(x_test)
    x_test.columns = col_name
    # x_test = np.nan_to_num(x_test, nan=0)

    # new features
    x_train = feature_combine(x_train)
    x_val = feature_combine(x_val)
    x_test = feature_combine(x_test)

    x_train = x_train.astype("float32")
    x_val = x_val.astype("float32")
    x_test = x_test.astype("float32")
    y_train = y_train.astype("float32")
    y_val = y_val.astype("float32")
    lgb_model = lgb.LGBMRegressor(
        num_leaves=64,  # QL: 42 -> 64
        objective="l1",  # QL: mse -> l1
        first_metric_only=True,
        n_estimators=100,
        learning_rate=0.1,  # QL: 0.1->0.01
        max_depth=7,
        colsample_bytree=0.7,
        seed=42,
        n_jobs=-1,
    )

    lgb_model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_metric=["mae", "rmse"],
        eval_names=["train", "valid"],
        categorical_feature=["sg_id"],
        callbacks=[
            lgb.early_stopping(20, first_metric_only=True),
            lgb.log_evaluation(10),
        ],
    )

    loss_train.append(lgb_model.evals_result_["train"]["l1"][-1])
    loss_val.append(lgb_model.evals_result_["valid"]["l1"][-1])
    joblib.dump(lgb_model, PROCESSED / "checkpoints" / f"lgb_1+2_model_{city}.pkl")

    # make prediction
    y_hat = lgb_model.predict(x_test)
    y_hat = y_hat.reshape((100, -1))

    if city == "london":
        y_missing = np.load(PROCESSED / "london_missing_y_eta_hat.npz")["arr_0"]
        x_error, x_correct = pd.read_pickle(PROCESSED / city / "error_index.pckl")
        for i in range(6):
            y_hat[x_error[i]] = y_missing[i]

    np.savez_compressed(SAVE_PROCESSED / city / "y_eta_hat.npz", y_hat)

print("Train Loss: ", loss_train)
print("Mean of train loss: %.2f", sum(loss_train) / 3)
print("Valid Loss: ", loss_val)
print("Mean of valid loss: %.2f", sum(loss_val) / 3)
