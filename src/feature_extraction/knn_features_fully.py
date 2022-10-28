import os
os.chdir("/home/superadmin/Desktop/Qinglong_Lu/Traffic4Cast2022-TSE")
from src.utils.miscs import config_sys_path
config_sys_path(".")

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import argparse
#%%
CACHEDIR = Path("cache")
PROCESSED = "processed"
#%%
def create_features(y_train, nbrs, k):
    y_mean = []
    y_std = []
    y_25 = []
    y_50 = []
    y_75 = []
    y_min = []
    y_max = []
    for i in range(len(nbrs)):
        if k == 'all':
            y0 = y_train
        else:
            y0 = y_train[nbrs[i]]
        y_mean.append(np.mean(y0, axis=0))
        y_std.append(np.std(y0, axis=0))
        y_25.append(np.quantile(y0, 0.25, axis=0))
        y_50.append(np.quantile(y0, 0.5, axis=0))
        y_75.append(np.quantile(y0, 0.75, axis=0))
        y_min.append(np.min(y0, axis=0))
        y_max.append(np.max(y0, axis=0))
    y_mean = np.concatenate(np.stack(y_mean))
    y_std = np.concatenate(np.stack(y_std))
    y_25 = np.concatenate(np.stack(y_25))
    y_50 = np.concatenate(np.stack(y_50))
    y_75 = np.concatenate(np.stack(y_75))
    y_min = np.concatenate(np.stack(y_min))
    y_max = np.concatenate(np.stack(y_max))
    x_knn = np.array([y_mean, y_std, y_25, y_50, y_75, y_min, y_max]).T
    
    return x_knn
#%%
def construct_knn_features(args, ks):
    print('===Preparing knn features for %s...' % args.city)
    print('- consider k in ', ks)
    num_t_per_day = 64
    x_impute = np.load(PROCESSED + '/'+args.city+'/X.npz')['arr_0']
    y = np.load(PROCESSED + '/'+args.city+'/y_eta.npz')['arr_0']
    y = np.reshape(y, (len(x_impute), -1))
    
    x_test = np.load(PROCESSED+'/'+args.city+'/X_test.npz')['arr_0'][:100]
    x_test = np.reshape(x_test, (x_test.shape[0], 4*x_test.shape[1]))
    
    num_days = int(len(x_impute)/num_t_per_day)
    num_sg = y.shape[1]
    k_collect = []
    print('- generate knn y_eta features for trainnning data')
    for d in tqdm(range(num_days)):
        x_support = np.concatenate((x_impute[:d*num_t_per_day],x_impute[(d+1)*num_t_per_day:]), axis=0)
        x_train = x_impute[d*num_t_per_day:(d+1)*num_t_per_day]
        
        y_support = np.concatenate((y[:d*num_t_per_day],y[(d+1)*num_t_per_day:]), axis=0)
        y_train = y[d*num_t_per_day:(d+1)*num_t_per_day]
    
        x_support = np.reshape(x_support, (x_support.shape[0], 4*x_support.shape[1]))
        x_support = np.nan_to_num(x_support, nan=0)
        x_train = np.reshape(x_train, (x_train.shape[0], 4*x_train.shape[1]))
    
        x_support = np.nan_to_num(x_support, nan=0)
        x_train = np.nan_to_num(x_train, nan=0)
    
    #%%
        knn = NearestNeighbors(p=int(args.knn_p), n_jobs=-1)
        knn.fit(x_support)
        k_day = []
        for k in ks:
            
            nbrs = knn.kneighbors(x_train, n_neighbors=k, return_distance=False)
            x_tmp = create_features(y_support, nbrs, k)
            k_day.append(x_tmp)            
        
        k_collect.append(k_day)
    k_collect = np.stack(k_collect)
    x_support = np.reshape(x_impute, (x_impute.shape[0], 4*x_impute.shape[1]))
    x_support = np.nan_to_num(x_support, nan=0)
    x_test = np.nan_to_num(x_test, nan=0)
    knn = NearestNeighbors(p=int(args.knn_p), n_jobs=-1)
    knn.fit(x_support)
    print('\n- generate knn y_eta features for test data and saving data...')
    for i, k in enumerate(ks):
        x_tmp = k_collect[:,i,:,:]
        x_knn = np.concatenate(x_tmp)
        np.savez_compressed(PROCESSED+'/'+args.city+'/knn_eng_train_p'+args.knn_p+str(k)+'.npz', x_knn)
        
        nbrs = knn.kneighbors(x_test, n_neighbors=k, return_distance=False)
        x_tmp = create_features(y, nbrs, k)
        np.savez_compressed(PROCESSED+'/'+args.city+'/knn_eng_test_p'+args.knn_p+str(k)+'.npz', x_tmp)
#%%
if __name__ == '__main__':
    ks = [5, 10, 30, 50, 100]
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='london')
    parser.add_argument('--knn_p', type=str, default='1')
    
    args = parser.parse_args()
    
    construct_knn_features(args, ks)


                

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
