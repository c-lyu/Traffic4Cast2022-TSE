import os
import zipfile
import argparse
import shutil
from pathlib import Path

import pandas as pd
from numpy import load
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

from t4c22.t4c22_config import load_road_graph
from src.utils.load import cfg

BASEDIR = cfg["BASEDIR"]
CACHEDIR = cfg["CACHEDIR"]
PROCESSED = cfg["PROCESSED"]
SAVE_PROCESSED = PROCESSED / "test_output_eta"
SUBMISSION = PROCESSED / "submission_eta"


def write_df_to_parquet(df: pd.DataFrame, fn: Path, compression="snappy"):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, fn, compression=compression)

def make_submission_df_eta(city, base_dir, test_output_dir):
    _, _, df_supersegments = load_road_graph(base_dir, city, skip_supersegments=False)
    
    df_hat = []
    y_hat = load(f'{SAVE_PROCESSED}/{city}/y_eta_hat.npz')['arr_0']
    for i in tqdm(range(100)):
        y_hat0 = y_hat[i]
        submission = pd.DataFrame({
            "eta": y_hat0,
            "identifier": df_supersegments.identifier.values,
            })
        submission["test_idx"] = i
        df_hat.append(submission)
    df_hat = pd.concat(df_hat)
    
    return df_hat[['identifier', 'test_idx', 'eta']]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_name", type=str, default="submission")

    args = parser.parse_args()
    submission_name = args.submission_name
    
    cities  = ["london", "melbourne", "madrid"]
    for city in cities:
        df = make_submission_df_eta(city, BASEDIR, SAVE_PROCESSED)
        if not os.path.exists(SUBMISSION / submission_name / city / "labels"):
            os.makedirs(SUBMISSION / submission_name / city / "labels")
        write_df_to_parquet(df=df, fn=SUBMISSION / submission_name / city / "labels" / f"eta_labels_test.parquet")
    submission_zip = SUBMISSION / f"{submission_name}.zip"
    with zipfile.ZipFile(submission_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for city in cities:
            z.write(
                filename=SUBMISSION / submission_name / city / "labels" / f"eta_labels_test.parquet",
                arcname=os.path.join(city, "labels", f"eta_labels_test.parquet"),
            )
    print(submission_zip)
    shutil.rmtree(SUBMISSION / submission_name)

