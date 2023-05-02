#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 13:46:46 2023

@author: cmwever73

Your challenge is to identify the number of discrete clusters present in 
the data, and create a clustering model that separates the data into that 
number of clusters. You should also visualize the clusters to evaluate the 
level of separation achieved by your model.
"""

import numpy as np
import pandas as pd
import random 
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


import mod_one_sumup as mos

#read in data
clstr_df = pd.read_csv('data/clusters.csv')
clstr_data = [{clstr_df.columns[indx]: data 
           for indx, data in enumerate(clstr_df.loc[rw,:])} 
           for rw in range(len(clstr_df))]
#explore
print(clstr_data[random.randint(0,len(clstr_df))])

obs_data = np.array([[data[ftr] for ftr in data ]for data in clstr_data])

ftrs_scld = MinMaxScaler().fit_transform(obs_data)

pca = PCA(n_components=2).fit(ftrs_scld)
ftrs_2d = pca.transform(ftrs_scld)
print(ftrs_2d[:15])

# plot data
cmpnt_0 = [ftr[0] for ftr in ftrs_2d]
cmpnt_1 = [ftr[1] for ftr in ftrs_2d]
mos.scttr_plt(('Component 0', cmpnt_0), ('Component 1', cmpnt_1))


#try k-means clustering on dataset up to 11 clusters
kmns_clstrs = []
for i in range(1, 12):
    clrs = ['#1a1535', '#5f54a6', '#7d70cc', '#d1a3ff', '#f4adff', '#f973ff',\
            '#b7438b', '#f15252', '#e47676', '#ef6e19', '#ea9049', '#eaad49']
    clrs_mp = {i : clr for i, clr in enumerate(clrs)}
    mdl = KMeans(n_clusters=i, n_init=100, max_iter=1000)
    preds = mdl.fit_predict(obs_data)
    kmns_clstrs.append((mdl, preds, [clrs[i] for i in preds], mdl.inertia_))
    

#plot each kmeans model
wcss_vls = []
for kmns_clstr in kmns_clstrs:
    mdl, preds, clr_lst, wcss = kmns_clstr
    mos.scttr_plt(('Component 0', cmpnt_0), ('Component 1', cmpnt_1), clr_lst)
    wcss_vls.append(wcss)

mos.ln_plt(('# clusters', range(1,12)), ('wcss', wcss_vls))
    
    
#try agglomerative clustering
kmns_clstrs = []
for i in range(1, 12):
    clrs = ['#1a1535', '#5f54a6', '#7d70cc', '#d1a3ff', '#f4adff', '#f973ff',\
            '#b7438b', '#f15252', '#e47676', '#ef6e19', '#ea9049', '#eaad49']
    clrs_mp = {i : clr for i, clr in enumerate(clrs)}
    mdl = AgglomerativeClustering(n_clusters=i)
    preds = mdl.fit_predict(obs_data)
    kmns_clstrs.append((mdl, preds, [clrs[i] for i in preds]))
    

#plot each kmeans model
for kmns_clstr in kmns_clstrs:
    mdl, preds, clr_lst = kmns_clstr
    mos.scttr_plt(('Component 0', cmpnt_0), ('Component 1', cmpnt_1), clr_lst)


