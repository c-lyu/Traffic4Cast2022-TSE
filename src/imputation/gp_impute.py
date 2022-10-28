import argparse
import logging
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import pyarrow

from t4c22.dataloading.t4c22_dataset import T4c22Dataset
from t4c22.t4c22_config import load_road_graph
from src.imputation.gp.gp_imputer import GaussianProcessImputer
from src.utils.load import none_filter
from src.utils.metrics import *
from src.utils.logging import setup_logger
from src.utils.load import cfg

BASEDIR = cfg["BASEDIR"]
CACHEDIR = cfg["CACHEDIR"]
PROCESSED = cfg["PROCESSED"]

Path(CACHEDIR / "impute").mkdir(parents=True, exist_ok=True)


def impute_data(args, logger):
    city = args.city

    # load road graph
    _, df_nodes, _ = load_road_graph(BASEDIR, city)
    nodes_valid = df_nodes.counter_info.ne("")

    train_dataset = T4c22Dataset(
        root=BASEDIR,
        city=city,
        split="train",
        day_t_filter=none_filter,
        cachedir=CACHEDIR / "processed",
    )
    logger.info(f"train set size: {len(train_dataset)}")

    logger.info(f"All datasets loaded!")
    day_list = sorted(set([day for day, t in train_dataset.day_t]))
    logger.debug(f"train - {day_list}")

    logger.info("Start implementing data imputation...")
    x_raw_path = CACHEDIR / "impute" / f"x_raw_{city}.npz"
    impute_path = CACHEDIR / "impute" / f"x_imputed_{city}_gp.npz"

    # sum up and impute data
    logger.info("Start reading data...")
    n_day_15min = 96 - 1
    n_dayt = len(train_dataset)
    n_dayt_ext = n_dayt // 92 * n_day_15min
    n_valid_nodes = len(nodes_valid[nodes_valid])
    logger.info(f"{n_dayt} {n_dayt_ext} {n_valid_nodes}")

    if x_raw_path.exists():
        x_raw = np.load(x_raw_path)["x"]
    else:
        x_raw = np.zeros((n_valid_nodes, n_dayt_ext))
        print(f"xraw {x_raw.shape}")
        for (day, t), (x, _) in tqdm(
            zip(train_dataset.day_t, train_dataset), total=n_dayt
        ):
            day_idx = day_list.index(day)
            dayt_idx = day_idx * n_day_15min + t

            if t == 4:
                x_raw[:, dayt_idx - 4 : dayt_idx] = x[nodes_valid].numpy()
            else:
                x_raw[:, dayt_idx - 1] = x[nodes_valid].numpy()[:, -1]
        np.savez_compressed(x_raw_path, x=x_raw)

    logger.info("Start imputing data...")
    gp_imputer = GaussianProcessImputer(bin_size=10, batch_size=200, n_jobs=args.n_jobs)
    _, x_imputed = gp_imputer.impute(x_raw)

    np.savez_compressed(impute_path, x=x_imputed)
    logger.info(f"Imputation result saved to {impute_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default="london")
    parser.add_argument("--n_jobs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--exp_name", type=str, default="impute")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        "logger", log_dir / f"{args.exp_name}.log", level=logging.DEBUG
    )

    impute_data(args, logger)
