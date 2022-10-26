# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:10:31 2022

@author: lasts
"""

import pandas as pd
import numpy as np
#%%
class DataImputation(object):
    def __init__(self, n_epochs, n_dims, lambda_U, lambda_V):
        self.n_epochs = n_epochs
        self.n_dims = n_dims
        self.lambda_U = lambda_U
        self.lambda_V = lambda_V
        self.parameters = {}
    
    def initialize_parameters(self):
        np.random.seed(11)
        U = np.random.normal(0.0, 1.0/self.lambda_U, (self.n_dims, self.n_venues))
        V = np.random.normal(0.0, 1.0/self.lambda_V, (self.n_dims, self.n_slices))
        
        self.parameters['U'] = U
        self.parameters['V'] = V
        self.parameters['lambda_U'] = self.lambda_U
        self.parameters['lambda_V'] = self.lambda_V
    #%%
    def update_parameters(self):
        U = self.parameters['U']
        V = self.parameters['V']
        lambda_U = self.parameters['lambda_U']
        lambda_V = self.parameters['lambda_V']
        
        for i in range(self.n_venues):
            V_j = V[:, self.R[i, :] > 0]
            U[:, i] = np.dot(np.linalg.inv(np.dot(V_j, V_j.T) + lambda_U * np.identity(self.n_dims)), np.dot(self.R[i, self.R[i, :] > 0], V_j.T))
            
        for j in range(self.n_slices):
            U_i = U[:, self.R[:, j] > 0]
            V[:, j] = np.dot(np.linalg.inv(np.dot(U_i, U_i.T) + lambda_V * np.identity(self.n_dims)), np.dot(self.R[self.R[:, j] > 0, j], U_i.T))
            
        self.parameters['U'] = U
        self.parameters['V'] = V
    #%%
    def log_a_posteriori(self):
        lambda_U = self.parameters['lambda_U']
        lambda_V = self.parameters['lambda_V']
        U = self.parameters['U']
        V = self.parameters['V']
        
        UV = np.dot(U.T, V)
        R_UV = (self.R[self.R > 0] - UV[self.R > 0])
        
        return -0.5 * (np.sum(np.dot(R_UV, R_UV.T)) + lambda_U * np.sum(np.dot(U, U.T)) + lambda_V * np.sum(np.dot(V, V.T)))
    #%%
    def evaluate(self, data_true):
        U = self.parameters['U']
        V = self.parameters['V']
        
        value_index = (~np.isnan(data_true))&(data_true>0)
        num = data_true[~value_index].size
        
        data_estimated = U.T @ V
        
        
        rmse = np.sqrt(sum((data_true[value_index]-data_estimated[value_index])**2)/num)
        mape = sum(np.absolute((data_true[value_index]-data_estimated[value_index])/data_true[value_index]))/num
        return rmse, mape
    #%%
    def update_max_min_pops(self):
        U = self.parameters['U']
        V = self.parameters['V']

        R = U.T @ V
        min_pop = np.min(R)
        max_pop = np.max(R)

        self.parameters['min_pop'] = min_pop
        self.parameters['max_pop'] = max_pop
    #%%
    def restore_estimation(self):
        U = self.parameters['U']
        V = self.parameters['V']

        data_estimated = U.T @ V

        q95 = np.quantile(data_estimated, 0.95)
        # q10 = np.quantile(data_estimated, 0.1)

        # data_estimated = np.where(data_estimated<q10, 0, data_estimated)
        data_estimated = np.where(data_estimated>q95, 2, data_estimated)

        data_estimated = pd.DataFrame(10**data_estimated)
        return data_estimated
    #%%
    def train(self, df_train):
        self.R = df_train.copy()
        self.n_venues = self.R.shape[0]
        self.n_slices = self.R.shape[1]
        self.parameters = {}
        
        self.initialize_parameters()
        log_aps = []
        rmse_train = []
        mape_train = []

        self.update_max_min_pops()
        rmse, mape = self.evaluate(df_train)
        rmse_train.append(rmse)
        mape_train.append(mape)
        
        for k in range(self.n_epochs):
            self.update_parameters()
            log_ap = self.log_a_posteriori()
            log_aps.append(log_ap)

            if (k + 1) % 10 == 0:
                self.update_max_min_pops()
                rmse, mape = self.evaluate(df_train)
                rmse_train.append(rmse)
                mape_train.append(mape)
                print('Log p a-posteriori at iteration', k + 1, ':', log_ap)
                print('RMSE at iteration ', k+1, ': ', rmse)
                print('MAPE at iteration ', k+1, ': ', mape)

        self.update_max_min_pops()
        data_estimated = self.restore_estimation()

        return log_aps, rmse_train, mape_train, self.parameters, data_estimated
    #%%
    
        