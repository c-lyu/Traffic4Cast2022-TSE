import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import argparse

from t4c22.t4c22_config import load_road_graph
from src.utils.load import cfg

BASEDIR = cfg["BASEDIR"]
CACHEDIR = cfg["CACHEDIR"]
PROCESSED = cfg["PROCESSED"]

def edge_stat(row, df_edges, speed_limit_field):
    edge_list = [e["edge"] for e in row["edges"]]
    row_edge = df_edges[(df_edges["edge"].isin(edge_list))]
    sl_mean = np.mean(row_edge[speed_limit_field])
    sl_std = np.std(row_edge[speed_limit_field])
    sl_25 = np.quantile(row_edge[speed_limit_field], 0.25)
    sl_50 = np.quantile(row_edge[speed_limit_field], 0.5)
    sl_75 = np.quantile(row_edge[speed_limit_field], 0.75)
    sl_min = np.min(row_edge[speed_limit_field])
    sl_max = np.max(row_edge[speed_limit_field])

    return [sl_mean, sl_std, sl_25, sl_50, sl_75, sl_min, sl_max]


def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def cal_dist(row, df_nodes):
    coor1 = df_nodes.loc[
        df_nodes["node_id"] == int(row["identifier"].split(",")[0]), ["x", "y"]
    ]
    lon1 = coor1["x"].values[0]
    lat1 = coor1["y"].values[0]

    coor2 = df_nodes.loc[
        df_nodes["node_id"] == int(row["identifier"].split(",")[1]), ["x", "y"]
    ]
    lon2 = coor2["x"].values[0]
    lat2 = coor2["y"].values[0]
    return 1000 * haversine(lon1, lat1, lon2, lat2)


def get_network_static(args):
    print("===Processing %s data..." % args.city)
    speed_limit_field = "speed_kph"
    if args.city == "madrid":
        speed_limit_field = "parsed_maxspeed"

    print("Loading data...")
    df_edges, df_nodes, rgss_df = load_road_graph(
        BASEDIR, args.city, skip_supersegments=False
    )
    df_static = pd.DataFrame()
    print("Summarizing supersegments...")
    edge_maxspeeds_kph = {}
    edge_lengths_m = {}
    for u, v, sl, lm in zip(
        df_edges["u"],
        df_edges["v"],
        df_edges[speed_limit_field],
        df_edges["length_meters"],
    ):
        edge_maxspeeds_kph[(u, v)] = sl
        edge_lengths_m[(u, v)] = lm

    supersegments = []
    for identifier, nodes in zip(rgss_df["identifier"], rgss_df["nodes"]):
        edges = []

        for n1, n2 in zip(nodes[:-1], nodes[1:]):
            e = (n1, n2)
            edges.append(
                {
                    "edge": e,
                    "max_speed": edge_maxspeeds_kph[e],
                    "length": edge_lengths_m[e],
                }
            )
        supersegments.append({"identifier": identifier, "edges": edges})
    print("Create static features...")
    df_static["n_nodes"] = rgss_df.apply(lambda x: len(x["nodes"]), axis=1)

    rgss_df = pd.concat([rgss_df, pd.DataFrame(supersegments)["edges"]], axis=1)
    df_static["length"] = rgss_df.apply(
        lambda x: sum([e["length"] for e in x["edges"]]), axis=1
    )

    df_edges["edge"] = df_edges[["u", "v"]].apply(tuple, axis=1)
    df_static["n_oneway"] = rgss_df.apply(
        lambda x: df_edges[
            (df_edges["edge"].isin([e["edge"] for e in x["edges"]]))
            & (df_edges["oneway"])
        ].shape[0],
        axis=1,
    )

    cols = ["sl_%s" % x for x in ["mean", "std", "25", "50", "75", "min", "max"]]
    df_static[cols] = rgss_df.apply(
        edge_stat, args=(df_edges, speed_limit_field), axis=1, result_type="expand"
    )

    # calculate harversine
    df_static["haversine"] = rgss_df.apply(cal_dist, df_nodes=df_nodes, axis=1)
    # shortest travel time
    df_static["shortest_tt"] = rgss_df.apply(
        lambda x: sum([e["length"] / e["max_speed"] for e in x["edges"]]), axis=1
    )
    # static travel time statistic
    y = np.load(PROCESSED / args.city / "y_eta.npz")["arr_0"]
    y = np.reshape(y, (-1, len(df_static)))
    count_0 = (y <= 1800).astype(int).sum(axis=0)
    count_1 = ((1800 < y) & (y <= 2400)).astype(int).sum(axis=0)
    count_2 = (2400 < y).astype(int).sum(axis=0)
    thre = np.stack([count_0, count_1, count_2]).T / y.shape[1]
    thre = pd.DataFrame(thre)
    thre.columns = ["18", "18-24", "24"]
    df_static = pd.concat([df_static, thre], axis=1)
    # save static features
    print("Saving static features...")

    df_static.to_parquet(PROCESSED / args.city / "x_static_eta.parquet")

    print("Compute the allnn statistics...")
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    y_25 = np.quantile(y, 0.25, axis=0)
    y_50 = np.quantile(y, 0.5, axis=0)
    y_75 = np.quantile(y, 0.75, axis=0)
    y_min = np.min(y, axis=0)
    y_max = np.max(y, axis=0)

    y_allnn = np.stack((y_mean, y_std, y_25, y_50, y_75, y_min, y_max), axis=1)
    np.savez_compressed(PROCESSED / args.city / "knn_eng_allnn.npz", y_allnn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="london")

    args = parser.parse_args()

    get_network_static(args)
