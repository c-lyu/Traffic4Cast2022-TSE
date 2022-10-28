#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:22:13 2022

@author: superadmin
"""


import os
import sys
sys.path.insert(0, os.path.abspath("../../NeurIPS2022-traffic4cast"))  # noqa:E402
sys.path.insert(1, os.path.abspath(".."))  # noqa:E402
sys.path.insert(1, os.path.abspath("../.."))  # noqa:E402

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil

from t4c22.t4c22_config import load_road_graph
from src.utils.load import load_basedir
import pyarrow as pa
import pyarrow.parquet as pq
from numpy import load
import pandas
import os
import zipfile

BASEDIR = load_basedir("../../../traffic4cast/T4C_INPUTS_2022")
CACHEDIR = Path("../../cache")
PROCESSED = "../../processed"
SAVE_PROCESSED = "../../processed/test_outputs_eta"
SUBMISSION = load_basedir("../../processed/submissions_eta")

submission_name = sys.argv[1]

# df = pd.read_parquet('../../../traffic4cast/T4C_INPUTS_ETA_2022/road_graph/madrid/road_graph_supersegments.parquet')


def write_df_to_parquet(df: pandas.DataFrame, fn: Path, compression="snappy"):
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

