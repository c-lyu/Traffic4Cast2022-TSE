def feature_combine(x_df):
    """Make combinatorial features from the original features.

    Parameters
    ----------
    x_df : pandas.DataFrame
        The original features.
    """
    x_df["node_valid_num_1/n_nodes"] = x_df["node_valid_num_1"] / x_df["n_nodes"]
    x_df["node_valid_num_sum"] = (
        x_df["node_valid_num_1"]
        + x_df["node_valid_num_2"]
        + x_df["node_valid_num_3"]
        + x_df["node_valid_num_4"]
    )
    x_df["node_valid_num_sum/n_nodes"] = x_df["node_valid_num_sum"] / x_df["n_nodes"]
    
    x_df["mean_all+3*std_all"] = x_df["mean_all"] + 3 * x_df["std_all"]
    x_df["mean_all-3*std_all"] = x_df["mean_all"] - 3 * x_df["std_all"]
    x_df["mean_all+1*std_all"] = x_df["mean_all"] + 1 * x_df["std_all"]
    x_df["mean_all-1*std_all"] = x_df["mean_all"] - 1 * x_df["std_all"]
    x_df["mean_all+2*std_all"] = x_df["mean_all"] + 2 * x_df["std_all"]
    x_df["mean_all-2*std_all"] = x_df["mean_all"] - 2 * x_df["std_all"]
    
    x_df["mean_50+3*std_50"] = x_df["mean_50"] + 3 * x_df["std_50"]
    x_df["mean_50-3*std_50"] = x_df["mean_50"] - 3 * x_df["std_50"]
    x_df["mean_50+1*std_50"] = x_df["mean_50"] + 1 * x_df["std_50"]
    x_df["mean_50-1*std_50"] = x_df["mean_50"] - 1 * x_df["std_50"]
    x_df["mean_50+2*std_50"] = x_df["mean_50"] + 2 * x_df["std_50"]
    x_df["mean_50-2*std_50"] = x_df["mean_50"] - 2 * x_df["std_50"]
    x_df["mean_30+3*std_30"] = x_df["mean_30"] + 3 * x_df["std_30"]
    x_df["mean_30-3*std_30"] = x_df["mean_30"] - 3 * x_df["std_30"]
    x_df["mean_30+1*std_30"] = x_df["mean_30"] + 1 * x_df["std_30"]
    x_df["mean_30-1*std_30"] = x_df["mean_30"] - 1 * x_df["std_30"]
    x_df["mean_30+2*std_30"] = x_df["mean_30"] + 2 * x_df["std_30"]
    x_df["mean_30-2*std_30"] = x_df["mean_30"] - 2 * x_df["std_30"]
    
    x_df["50_30/75_all"] = x_df["50_30"] / x_df["75_all"]
    x_df["50_30/50_all"] = x_df["50_30"] / x_df["50_all"]
    x_df["50_50/50_all"] = x_df["50_50"] / x_df["50_all"]

    x_df["mean_all/shortest_tt"] = x_df["mean_all"] / x_df["shortest_tt"]
    x_df["mean_30/shortest_tt"] = x_df["mean_30"] / x_df["shortest_tt"]
    x_df["mean_50/shortest_tt"] = x_df["mean_50"] / x_df["shortest_tt"]
    x_df["mean_all-shortest_tt"] = x_df["mean_all"] - x_df["shortest_tt"]
    x_df["mean_30-shortest_tt"] = x_df["mean_30"] - x_df["shortest_tt"]
    x_df["mean_50-shortest_tt"] = x_df["mean_50"] - x_df["shortest_tt"]
    x_df["min_all-shortest_tt"] = x_df["min_all"] - x_df["shortest_tt"]
    x_df["min_30-shortest_tt"] = x_df["min_30"] - x_df["shortest_tt"]
    x_df["min_50-shortest_tt"] = x_df["min_50"] - x_df["shortest_tt"]

    x_df["mean_30/mean_all"] = x_df["mean_30"] / x_df["mean_all"]
    x_df["mean_50/mean_all"] = x_df["mean_50"] / x_df["mean_all"]
    x_df["mean_30/mean_50"] = x_df["mean_30"] / x_df["mean_50"]
    x_df["mean_100/mean_all"] = x_df["mean_100"] / x_df["mean_all"]
    x_df["mean_30-mean_all"] = x_df["mean_30"] - x_df["mean_all"]
    x_df["mean_50-mean_all"] = x_df["mean_50"] - x_df["mean_all"]
    x_df["mean_100-mean_all"] = x_df["mean_100"] - x_df["mean_all"]
    x_df["mean_30-mean_50"] = x_df["mean_30"] - x_df["mean_50"]

    x_df["min_all-max_all"] = x_df["min_all"] - x_df["max_all"]
    x_df["min_all-mean_all"] = x_df["min_all"] - x_df["mean_all"]
    x_df["max_all-mean_all"] = x_df["max_all"] - x_df["mean_all"]
    x_df["min_all/max_all"] = x_df["min_all"] / x_df["max_all"]
    x_df["min_all/mean_all"] = x_df["min_all"] / x_df["mean_all"]
    x_df["max_all/mean_all"] = x_df["max_all"] / x_df["mean_all"]

    x_df["haversine/length"] = x_df["haversine"] / x_df["length"]

    x_df["min_30/min_all"] = x_df["min_30"] / x_df["min_all"]
    x_df["min_50/min_all"] = x_df["min_50"] / x_df["min_all"]
    x_df["max_30/max_all"] = x_df["max_30"] / x_df["max_all"]
    x_df["max_50/max_all"] = x_df["max_50"] / x_df["max_all"]

    return x_df
