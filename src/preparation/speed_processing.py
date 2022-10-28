import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

from t4c22.t4c22_config import load_road_graph
from t4c22.dataloading.t4c22_dataset import T4c22Dataset
from src.utils.load import none_filter
from src.utils.load import cfg

BASEDIR = cfg["BASEDIR"]
CACHEDIR = cfg["CACHEDIR"]
PROCESSED = cfg["PROCESSED"]
SPEED_FOLDER = BASEDIR / "speed_classes"


# load data
cities = ["london", "melbourne", "madrid"]

num_t_per_day = 64  # 6:00-22:00 --> see prepare_train_test_arrays.py
start_t = 24

for city in cities:
    dataset = T4c22Dataset(
        root=BASEDIR,
        city=city,
        split="train",
        day_t_filter=none_filter,  # <--
        cachedir=CACHEDIR / "processed",
    )

    df_edges, df_nodes, _ = load_road_graph(BASEDIR, city)
    df_edges.set_index(["u", "v"], inplace=True)

    # have to run it batch-wise, otherwise crash
    days = [i[0] for i in dataset.day_t]
    sc_files = list(OrderedDict.fromkeys(days))

    for k in tqdm(range(0, len(sc_files), 10)):
        x_speed = []
        k_end = k + 10 if k + 10 < len(sc_files) else len(sc_files)
        for i in range(k, k_end):
            df_sc = pd.read_parquet(
                SPEED_FOLDER + "/" + city + "/speed_classes_" + sc_files[i] + ".parquet"
            )
            df_sc.set_index(["u", "v"], inplace=True)
            for j in range(start_t, start_t + num_t_per_day):
                x0 = df_sc.loc[
                    df_sc["t"] == j,
                    ["volume_class", "median_speed_kph", "free_flow_kph"],
                ]
                x0 = pd.concat([x0, df_edges["speed_kph"]], axis=1)
                x0 = x0.reindex(df_edges.index)
                x_speed.append(np.array(x0.iloc[:, :3]))
        x_speed = np.stack(x_speed)
        # col_mean = np.nanmean(x_speed, axis=1)
        # inds = np.where(np.isnan(x_speed))
        # x_speed[inds] = col_mean[inds[0],inds[2]]
        np.save(PROCESSED + "/" + city + "/" + "x_speed_" + str(k), x_speed)

    x_speed = []
    for k in tqdm(range(0, len(sc_files), 10)):
        x0 = np.load(PROCESSED + "/" + city + "/" + "x_speed_" + str(k) + ".npy")
        x_speed.append(x0)
        os.remove(PROCESSED + "/" + city + "/" + "x_speed_" + str(k) + ".npy")
    x_speed = np.concatenate(x_speed)
    np.savez_compressed(PROCESSED + "/" + city + "/" + "x_speed", x_speed)
