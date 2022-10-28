import argparse
import numpy as np

from t4c22.t4c22_config import load_road_graph
from src.utils.load import cfg

BASEDIR = cfg["BASEDIR"]
CACHEDIR = cfg["CACHEDIR"]
PROCESSED = cfg["PROCESSED"]
road_graph_folder = BASEDIR / "road_graphs"


def loop_statistics(args):
    print(
        "===Preparing %s... calculate_missing=%s, construct_test=%s"
        % (args.city, args.calculate_missing, args.construct_test)
    )
    x_impute = np.load(PROCESSED + "/" + args.city + "/X.npz")["arr_0"]
    if args.calculate_missing & (args.city == "london"):
        num_nan_raw = np.load(PROCESSED + "/london/" + "nan_percent_all.npy")
        x_impute = x_impute[num_nan_raw > 80]

    x_test = np.load(PROCESSED + "/" + args.city + "/X_test.npz")["arr_0"][:100]

    df_edges, df_nodes, rgss_df = load_road_graph(
        BASEDIR, args.city, skip_supersegments=False
    )

    nodes_valid = df_nodes.counter_info.ne("")
    df_nodes = df_nodes[nodes_valid].reset_index(drop=True)

    # for trainning data
    df_sum = []
    df_mean = []
    df_std = []
    num_valid = []
    for i, row in rgss_df.iterrows():
        row_index = df_nodes[df_nodes["node_id"].isin(row["nodes"])].index
        arr_row = x_impute[:, row_index, :]
        df_sum.append(np.nansum(arr_row, axis=1))
        df_mean.append(np.nanmean(arr_row, axis=1))
        df_std.append(np.nanstd(arr_row, axis=1))
        num_valid.append(np.count_nonzero(~np.isnan(arr_row), axis=1))

    df_sum = np.transpose(np.stack(df_sum), (1, 0, 2))
    df_mean = np.transpose(np.stack(df_mean), (1, 0, 2))
    df_std = np.transpose(np.stack(df_std), (1, 0, 2))
    num_valid = np.transpose(np.stack(num_valid), (1, 0, 2))

    x_nodes = np.concatenate((df_sum, df_mean, df_std, num_valid), axis=2)
    if args.calculate_missing:
        np.savez_compressed(
            PROCESSED + "/" + args.city + "/x_nodes_eta_missing.npz", x_nodes
        )
    else:
        np.savez_compressed(PROCESSED + "/" + args.city + "/x_nodes_eta.npz", x_nodes)

    # for test data
    if args.construct_test:
        df_sum = []
        df_mean = []
        df_std = []
        num_valid = []
        for i, row in rgss_df.iterrows():
            row_index = df_nodes[df_nodes["node_id"].isin(row["nodes"])].index
            arr_row = x_test[:, row_index, :]
            df_sum.append(np.nansum(arr_row, axis=1))
            df_mean.append(np.nanmean(arr_row, axis=1))
            df_std.append(np.nanstd(arr_row, axis=1))
            num_valid.append(np.count_nonzero(~np.isnan(arr_row), axis=1))

        df_sum = np.transpose(np.stack(df_sum), (1, 0, 2))
        df_mean = np.transpose(np.stack(df_mean), (1, 0, 2))
        df_std = np.transpose(np.stack(df_std), (1, 0, 2))
        num_valid = np.transpose(np.stack(num_valid), (1, 0, 2))

        x_nodes = np.concatenate((df_sum, df_mean, df_std, num_valid), axis=2)
        np.savez_compressed(
            PROCESSED + "/" + args.city + "/x_nodes_test_eta.npz", x_nodes
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="london")
    parser.add_argument("--calculate_missing", action="store_true")
    parser.add_argument("--construct_test", action="store_true")

    args = parser.parse_args()

    loop_statistics(args)
