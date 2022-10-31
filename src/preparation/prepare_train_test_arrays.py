import torch
import numpy as np
from numpy import savez_compressed
from tqdm import tqdm

from t4c22.dataloading.t4c22_dataset import T4c22Dataset
from t4c22.t4c22_config import load_road_graph
from src.dataloading.dataset import T4c22TestDataset
from src.utils.load import get_edge_node_idx
from src.utils.load import none_filter
from src.utils.metrics import *
from src.utils.load import cfg

BASEDIR = cfg["BASEDIR"]
CACHEDIR = cfg["CACHEDIR"]
PROCESSED = cfg["PROCESSED"]
SPEED_FOLDER = BASEDIR / "speed_classes"


cities = ["london", "melbourne", "madrid"]


def get_data_daytime(city):
    dataset = T4c22Dataset(
        root=BASEDIR,
        city=city,
        split="train",
        day_t_filter=none_filter,
        competition="eta",
        cachedir=CACHEDIR / "processed",
    )
    x_imputed = np.load(CACHEDIR / "impute" / f"x_imputed_{city}_gp.npz")["x"]
    num_valid_edges = x_imputed.shape[0]  # <--
    x_imputed = x_imputed.reshape(
        num_valid_edges, -1, 95
    )  # <-- num_valid_edges * day * t

    sc_files = sorted((SPEED_FOLDER / city).glob("speed_classes_*.parquet"))

    X = []
    X_raw = []
    Y = []
    for i, ((x, y), (day, t)) in tqdm(
        enumerate(zip(dataset, dataset.day_t)), total=len(dataset)
    ):
        if not 24 <= t < 88:  # <--
            continue  # <--

        i_day = [k for k, sc_file in enumerate(sc_files) if day in str(sc_file)][0]
        X.append(x_imputed[:, i_day, t - 4 : t])
        X_raw.append(x)
        Y.append(y)

    X = np.stack(X, axis=0)
    X_raw = np.stack(X_raw, axis=0)
    Y = torch.cat(Y, dim=0).numpy()
    return X, X_raw, Y


for city in cities:

    df_edges, df_nodes, _ = load_road_graph(BASEDIR, city)
    edge_u_idx, edge_v_idx = get_edge_node_idx(BASEDIR, city)

    valid_node_indices = df_nodes[df_nodes.counter_info.ne("")].index.to_list()
    nodes_valid = df_nodes.counter_info.ne("")

    X_support, X_raw_support, y_support = get_data_daytime(city)  # <--
    y_support = np.nan_to_num(y_support, nan=-1)
    X_raw_support = np.asarray(
        [X_raw_support[i][valid_node_indices] for i in range(len(X_raw_support))]
    )

    savez_compressed(PROCESSED / city / "X.npz", X_support)
    savez_compressed(PROCESSED / city / "X_raw.npz", X_raw_support)
    savez_compressed(PROCESSED / city / "y_eta.npz", y_support)


for city in cities:

    df_edges, df_nodes, _ = load_road_graph(BASEDIR, city)
    edge_u_idx, edge_v_idx = get_edge_node_idx(BASEDIR, city)

    valid_node_indices = df_nodes[df_nodes.counter_info.ne("")].index.to_list()
    nodes_valid = df_nodes.counter_info.ne("")

    test_dataset = T4c22TestDataset(
        root=BASEDIR, city=city, cachedir=CACHEDIR / "processed"
    )

    X_test = np.asarray([x[nodes_valid].numpy() for x, _ in tqdm(test_dataset)])

    savez_compressed(PROCESSED / city / "X_test.npz", X_test)
