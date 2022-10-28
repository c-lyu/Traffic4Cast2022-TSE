import pandas as pd
import numpy as np
#%%
def read_knn_features(path, city, ks, obj, shape, p, zone='', missing=False):
    knn_eng = []
    for k in ks:
        if missing:
            features = np.load(path+'/'+city+'/knn_eng_'+obj+'_'+p+str(k)+'_missing.npz')['arr_0']
        else:
            features = np.load(path+'/'+city+'/knn_eng_'+obj+'_'+p+str(k)+zone+'.npz')['arr_0']
        knn_eng.append(features)
    knn_features = np.concatenate(knn_eng, axis=1)
    if obj == 'train':
        x_knn = np.reshape(knn_features, (shape[0],shape[1],-1))
    else:
        x_knn = np.reshape(knn_features, (100,int(len(knn_features)/100),-1))

    return x_knn
   
def read_speed_features(path, city, ks, obj, shape, p, missing=False):
    knn_eng = []
    for k in ks:
        if missing:
            features = np.load(path+'/'+city+'/knn_eng_'+obj+'_speed_'+p+str(k)+'_missing.npz')['arr_0']   
        else:
            features = np.load(path+'/'+city+'/knn_eng_'+obj+'_speed_'+p+str(k)+'.npz')['arr_0']
        knn_eng.append(features)
    knn_features = np.concatenate(knn_eng, axis=1)
    if obj == 'train':
        x_sg_speed = np.reshape(knn_features, (shape[0],shape[1],-1))
    else:
        x_sg_speed = np.reshape(knn_features, (100,int(len(knn_features)/100),-1))
    return x_sg_speed
