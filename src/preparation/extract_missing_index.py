import os
os.chdir("/home/superadmin/Desktop/Qinglong_Lu/Traffic4Cast2022-TSE")
from src.utils.miscs import config_sys_path
config_sys_path(".")

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
#%%
CACHEDIR = Path("cache")
PROCESSED = "processed"
SAVE_PROCESSED = "processed/test_outputs_eta"
#%%
cities = ['london','melbourne','madrid']
num_t_per_day = 64
num_week = 3 # size of the trainning set
week_val = 11
#%%
# calculate the missing rate in X of every sample
num_nan_test = []
num_nan_train = []
for city in cities:
    x_test = np.load(PROCESSED+'/'+city+'/X_test.npz')['arr_0'][:100]
    x_raw = np.load(PROCESSED+'/'+city+'/X_raw.npz')['arr_0']
    num_nan_city_test = []
    num_nan_city_train = []
    for i in range(len(x_test)):
        nan_percent = 100*x_test[i][np.isnan(x_test[i])].shape[0]/(x_test[i].shape[0]*x_test[i].shape[1])
        num_nan_city_test.append(nan_percent)
    num_nan_test.append(num_nan_city_test)
    for i in range(len(x_raw)):
        nan_percent = 100*x_raw[i][np.isnan(x_raw[i])].shape[0]/(x_raw[i].shape[0]*x_raw[i].shape[1])
        num_nan_city_train.append(nan_percent)
    num_nan_train.append(num_nan_city_train)
#%%
num_nan = num_nan_test.copy()
df_num = pd.DataFrame(num_nan).T
df_num.columns = cities
# only london has 6 samples have 90% nan values
city = 'london'

x_error = df_num[df_num[city]>80].index.to_list()
x_correct = [i for i in range(100) if i not in x_error]

f = open(PROCESSED + '/'+city+'/' + 'error_index.pckl', 'wb')
pickle.dump([x_error, x_correct], f)
f.close()
#%%
# to ensure that using mean to represent the nan samples useful, all trainning samples should not be nan samples
x_raw = np.load(PROCESSED+'/'+city+"/"+'X_raw.npz')['arr_0']

num_nan_raw = []
for i in range(len(x_raw)):
    nan_percent = 100*x_raw[i][np.isnan(x_raw[i])].shape[0]/(x_raw[i].shape[0]*x_raw[i].shape[1])
    num_nan_raw.append(nan_percent)
    print('%d-th sample: %.2f values are nan.' % (i, nan_percent))
num_nan_raw = np.array(num_nan_raw)
np.save(PROCESSED + '/'+city+'/' + 'nan_percent_all', num_nan_raw)


    
    
    
    
