import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from t4c22.t4c22_config import load_road_graph
from src.utils.load import cfg

BASEDIR = cfg["BASEDIR"]
PROCESSED = cfg["PROCESSED"]
SPEED_FOLDER = BASEDIR / "speed_classes"


def get_class_count_boost(row0):
    # count
    count_1 = (row0 == 1).astype(int).sum(axis=1)
    count_3 = (row0 == 3).astype(int).sum(axis=1)
    count_5 = (row0 == 5).astype(int).sum(axis=1)
    count_nan = (np.isnan(row0)).astype(int).sum(axis=1)
    vc = np.stack([count_1, count_3, count_5, count_nan]).T / row0.shape[1]
    return vc


def create_features(sg_speed_support, nbrs, k):
    y_mean = []
    y_std = []
    for i in range(len(nbrs)):
        y0 = sg_speed_support[nbrs[i]]
        y_mean.append(np.nanmean(y0, axis=0))
        y_std.append(np.nanstd(y0, axis=0))
    y_mean = np.concatenate(np.stack(y_mean))
    y_std = np.concatenate(np.stack(y_std))
    x_knn = np.concatenate((y_mean, y_std), axis=1)
    return x_knn


def cal_speed_features(args, ks):
    num_t_per_day = 64
    print("===Preparing speed features for %s..." % args.city)
    print("- consider k in ", ks)
    if args.create_x_sg:
        print("- summarize supersegment speed information")
        df_edges, df_nodes, rgss_df = load_road_graph(
            BASEDIR, args.city, skip_supersegments=False
        )

        supersegments = []
        for identifier, nodes in zip(rgss_df["identifier"], rgss_df["nodes"]):
            edges = []

            for n1, n2 in zip(nodes[:-1], nodes[1:]):
                e = (n1, n2)
                edges.append(e)
            supersegments.append({"identifier": identifier, "edges": edges})

        rgss_df = pd.concat([rgss_df, pd.DataFrame(supersegments)["edges"]], axis=1)
        df_edges["edge"] = df_edges[["u", "v"]].apply(tuple, axis=1)

        x_speed = np.load(PROCESSED / args.city / "x_speed.npz")["arr_0"]
        sg_speed = []
        vcs = []
        for i, sg in tqdm(rgss_df.iterrows(), total=len(rgss_df)):
            idx = df_edges[df_edges["edge"].isin(sg["edges"])].index
            rows = x_speed[:, idx, :]
            sg_speed.append(np.nanmean(rows[:, :, 1:], axis=1))
            vc = get_class_count_boost(rows[:, :, 0])
            vcs.append(vc)
        sg_speed = np.transpose(np.stack(sg_speed), (1, 0, 2))
        vcs = np.transpose(np.stack(vcs), (1, 0, 2))
        sg_speed = np.concatenate((sg_speed, vcs), axis=2)

        np.savez_compressed(PROCESSED / args.city / "x_sg.npz", sg_speed)
    else:
        sg_speed = np.load(PROCESSED / args.city / "x_sg.npz")["arr_0"]

    x_impute = np.load(PROCESSED / args.city / "X.npz")["arr_0"]
    y_eta = np.load(PROCESSED / args.city / "y_eta.npz")["arr_0"]
    y_eta = np.reshape(y_eta, (len(x_impute), -1))

    x_test = np.load(PROCESSED / args.city / "X_test.npz")["arr_0"][:100]
    x_test = np.reshape(x_test, (x_test.shape[0], 4 * x_test.shape[1]))

    num_days = int(len(x_impute) / num_t_per_day)
    k_collect = []
    print("\n- generate knn speed features for trainning data")
    for d in tqdm(range(num_days)):
        x_support = np.concatenate(
            (x_impute[: d * num_t_per_day], x_impute[(d + 1) * num_t_per_day :]), axis=0
        )
        x_train = x_impute[d * num_t_per_day : (d + 1) * num_t_per_day]

        x_support = np.reshape(x_support, (x_support.shape[0], 4 * x_support.shape[1]))
        x_support = np.nan_to_num(x_support, nan=0)
        x_train = np.reshape(x_train, (x_train.shape[0], 4 * x_train.shape[1]))

        x_support = np.nan_to_num(x_support, nan=0)
        x_train = np.nan_to_num(x_train, nan=0)
        x_test = np.nan_to_num(x_test, nan=0)

        sg_speed_support = np.concatenate(
            (sg_speed[: d * num_t_per_day], sg_speed[(d + 1) * num_t_per_day :]), axis=0
        )

        knn = NearestNeighbors(p=int(args.knn_p), n_jobs=-1)
        knn.fit(x_support)  # to avoid info leakage in the validation set
        # calculate the distance between observations
        k_day = []
        for k in ks:
            nbrs = knn.kneighbors(x_train, n_neighbors=k, return_distance=False)
            x_tmp = create_features(sg_speed_support, nbrs, k)
            k_day.append(x_tmp)

        k_collect.append(k_day)
    k_collect = np.stack(k_collect)
    x_support = np.reshape(x_impute, (x_impute.shape[0], 4 * x_impute.shape[1]))
    x_support = np.nan_to_num(x_support, nan=0)
    x_test = np.nan_to_num(x_test, nan=0)
    knn = NearestNeighbors(p=int(args.knn_p), n_jobs=-1)
    knn.fit(x_support)
    print("- generate knn speed features for test data and saving data...")
    for i, k in enumerate(ks):
        x_tmp = k_collect[:, i, :, :]
        x_knn = np.concatenate(x_tmp)
        np.savez_compressed(
            f"{PROCESSED}/{args.city}/knn_eng_train_speed_p{args.knn_p}{k}.npz", x_knn
        )

        nbrs = knn.kneighbors(x_test, n_neighbors=k, return_distance=False)
        x_tmp = create_features(sg_speed, nbrs, k)
        np.savez_compressed(
            f"{PROCESSED}/{args.city}/knn_eng_test_speed_p{args.knn_p}{k}.npz", x_tmp
        )


if __name__ == "__main__":
    ks = [5, 10, 50]
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="london")
    parser.add_argument("--create_x_sg", action="store_true")
    parser.add_argument("--knn_p", type=str, default="1")

    args = parser.parse_args()

    cal_speed_features(args, ks)
