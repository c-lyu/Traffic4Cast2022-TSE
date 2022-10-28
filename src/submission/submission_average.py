#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 08:50:16 2022

@author: superadmin
"""

import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
PROCESSED = '../../processed'
SAVE_PROCESSED = '../../processed/test_outputs_eta'

cities = ['london','melbourne','madrid']
all_sub = []
subs = ['alpha_wusong', 'alpha_leiheng', 'beta_muhong', 
        'omega_luzhishen', 'alpha_gongsunsheng', 'omega_yangzhi', 
        'omega_daizong', 'omega_suochao', 'alpha_xuning', 'alpha_liutang']

sub_london = []
sub_melbourne = []
sub_madrid = []
for sub in subs:
    with zipfile.ZipFile(PROCESSED + '/submissions_eta/'+sub+'.zip') as myzip:
        sub_city = []
        for city in cities:
            df_tmp = myzip.open(city+'/labels/eta_labels_test.parquet')
            df = pd.read_parquet(df_tmp)
            sub_city.append(df['eta'])
        sub_london.append(sub_city[0])
        sub_melbourne.append(sub_city[1])
        sub_madrid.append(sub_city[2])
#%%
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(25,5))
for j in range(5):
    ax[j].scatter(sub_london[1], sub_london[j])
#%%
london5 = np.mean(sub_london[:2], axis=0).reshape(100,-1)
melbourne5 = np.mean(sub_melbourne[:2], axis=0).reshape(100,-1)
madrid5 = np.mean(sub_madrid[:2], axis=0).reshape(100,-1)
#%%
delta = 2*np.random.binomial(n=1, p=0.5, size=len(sub_london[0]))-1
# london5 = london5 + 0.02*gra*delta

yplus = 59.784754435221
yminus = 59.782447814941
gra = (yplus-yminus)/(2*delta)

london5[:,mask] = y_test_hat
#%%
london5 = london5.reshape(100,-1)
np.savez_compressed(SAVE_PROCESSED+'/london/y_eta_hat', london5)
np.savez_compressed(SAVE_PROCESSED+'/melbourne/y_eta_hat', melbourne5)
np.savez_compressed(SAVE_PROCESSED+'/madrid/y_eta_hat', madrid5)





