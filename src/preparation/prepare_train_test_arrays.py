import os
os.chdir("/home/superadmin/Desktop/Qinglong_Lu/Traffic4Cast2022-TSE")
from src.utils.miscs import config_sys_path
config_sys_path(".")

import numpy as np

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch
from t4c22.dataloading.t4c22_dataset import T4c22Dataset
from src.dataloading.dataset import T4c22TestDataset
from t4c22.t4c22_config import load_road_graph

from src.utils.load import load_basedir, get_edge_node_idx
from src.utils.load import none_filter   # <--
from src.utils.metrics import *

from numpy import savez_compressed


# load BASEDIR from file, change to your data root
BASEDIR = load_basedir("data")
CACHEDIR = Path("cache")
PROCESSED = "processed"
SPEED_FOLDER = Path('data/speed_classes')

cities = ["london", "melbourne", "madrid"]

def get_data_daytime(city):
    dataset = T4c22Dataset(root=BASEDIR, city=city, split="train",
                            day_t_filter=none_filter,
                            competition='eta',
                            cachedir=CACHEDIR / "processed")
    x_imputed = np.load(CACHEDIR / 'impute' / f'x_imputed_{city}_gp.npz')['x']
    num_valid_edges = x_imputed.shape[0]  # <--
    x_imputed = x_imputed.reshape(num_valid_edges, -1, 95)  # <-- num_valid_edges * day * t
    
    sc_files = sorted((SPEED_FOLDER / city).glob("speed_classes_*.parquet"))
    
    X = []
    X_raw = []
    Y = []
    for i, ((x, y), (day, t)) in tqdm(enumerate(zip(dataset, dataset.day_t)), total=len(dataset)):
        if not 24 <= t < 88:   # <--
            continue   # <--
        
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
    X_raw_support = np.asarray([X_raw_support[i][valid_node_indices] for i in range(len(X_raw_support))])

    savez_compressed(PROCESSED+'/'+city+"/"+'X', X_support)
    savez_compressed(PROCESSED+'/'+city+"/"+'X_raw', X_raw_support)
    savez_compressed(PROCESSED+'/'+city+"/"+'y_eta', y_support)
    

for city in cities:

    df_edges, df_nodes, _ = load_road_graph(BASEDIR, city)
    edge_u_idx, edge_v_idx = get_edge_node_idx(BASEDIR, city)

    valid_node_indices = df_nodes[df_nodes.counter_info.ne("")].index.to_list()
    nodes_valid = df_nodes.counter_info.ne("")
    
    test_dataset = T4c22TestDataset(root=BASEDIR, city=city, cachedir=CACHEDIR / "processed")

    X_test = np.asarray([x[nodes_valid].numpy() for x, _ in tqdm(test_dataset)])

    
    savez_compressed(PROCESSED+'/'+city+"/"+'X_test', X_test)
