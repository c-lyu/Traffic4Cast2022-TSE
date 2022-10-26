# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 14:09:23 2022

@author: lasts
"""
import os
import sys
sys.path.insert(0, os.path.abspath("../../NeurIPS2022-traffic4cast"))  # noqa:E402
sys.path.insert(1, os.path.abspath(".."))  # noqa:E402
sys.path.insert(1, os.path.abspath("../.."))  # noqa:E402

import pandas as pd
import numpy as np
from math import ceil
from src.imputation.nmf.matrix_factorization import DataImputation

import pickle
from pathlib import Path
#%%
from t4c22.t4c22_config import load_road_graph
from src.utils.load import load_basedir
#%%
def train_imputation(x, batch_size=3000):
  # x = pd.DataFrame(train_dataset[0])
  # for day_t in range(1,n_dayts):
  #   x = pd.concat([x, pd.DataFrame(train_dataset[day_t]).iloc[:,-1]], axis=1)
  n_batch = ceil(x.shape[1] / batch_size)
  print('The dataset will be divided into %s batches'%n_batch)
  n_dims = 200
  n_iter = 10
  impute = DataImputation(n_iter, n_dims, 0.3, 0.3)
  
  batch_estimated = []
  batch_rmse = []
  batch_mape = []
  for bat in range(n_batch):
    print('Imputing batch %s'%bat)
    x_batch = x[:,bat*batch_size:(bat+1)*batch_size]
    x_batch = np.where(x_batch==0, np.nan, x_batch)
    x_batch = np.log10(x_batch)
    
    log_aps, rmse_train, mape_train, parameters, data_estimated_impute = impute.train(x_batch)
    
    x_batch = pd.DataFrame(10**x_batch)
    x_tmp = np.where(x_batch.isnull(), data_estimated_impute, x_batch)
    batch_estimated.append(pd.DataFrame(x_tmp))
    batch_rmse.append(rmse_train[-1])
    batch_mape.append(mape_train[-1])
  
  x_imputed = pd.concat(batch_estimated, axis=1).values
  return x, x_imputed, batch_rmse, batch_mape

if __name__ == '__main__':
  BASEDIR = load_basedir("../../../traffic4cast/T4C_INPUTS_2022")
  CACHEDIR = Path("../../cache")
  city = "london"
  df_edges, df_nodes, _ = load_road_graph(BASEDIR, city)
  X_support_sample, y_support_sample = pd.read_pickle('../../../traffic4cast/T4C_INPUTS_2022/data_sample.pckl')
  del y_support_sample
  n_dayts = X_support_sample.shape[0]
  x_raw, x_imputed, rmse_train, mape_train = train_imputation(X_support_sample, n_dayts, batch_size=1000)
  del X_support_sample
  f = open('../../cache/clustering/imputation_results.pckl','wb')
  pickle.dump([x_raw, x_imputed], f)
  f.close()