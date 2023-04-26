#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:18:59 2023

@author: cmwever73

Wine Challenge:
In this challenge, you must train a classification model to analyze the c
hemical and visual features of wine samples and classify them based on their 
cultivar (grape variety).

Your challenge is to explore the data and train a classification model 
that achieves an overall Recall metric of over 0.95 (95%).

Once done, test model with these two data points:
    [13.72,1.43,2.5,16.7,108,3.4,3.67,0.19,2.04,6.8,0.89,2.87,1285]
    [12.37,0.94,1.36,10.6,88,1.98,0.57,0.28,0.42,1.95,1.05,1.82,520]



For myself:
    i. make vertical boxplot func to compare features by label
    ii. func to retrurn same output as sklearn report 
        (accuracy, preciison, f1score etc)
    iii. func to return ROC plot and AOC number
    iv. func to make confusion matrix
    v. sketch out what should be in a Class -- prob wants to amke ML class
"""
import pandas as pd
import random


import mod_one_sumup as mos

#read in data
wn_df = pd.read_csv('data/wine.csv')
wn_data = [{wn_df.columns[indx]: data 
           for indx, data in enumerate(wn_df.loc[rw,:])} 
           for rw in range(len(wn_df))]
#explore
print(wn_data[random.randint(0,len(wn_df))])

#label is WineVariety
lbl = 'WineVariety'
wn_var = ('Wine Variety', [data[lbl] for data in wn_data])
ftrs = list(wn_data[0].keys())
ftrs.remove(lbl)
#make box plots
to_plt_data = []
for ftr in ftrs:
    ftr_data = (ftr, [data[ftr] for data in wn_data])
    to_plt_data.append({'X': wn_var, 'Y': ftr_data, 'Title': lbl+' by '+ftr })
    # mos.vert_bxplt(wn_var, ftr_data)    
mos.vert_bxplt_many(to_plt_data)

#preprocess (get rid of nulls, scale, nu vs cat)

ptntl_nlls = []
for indx, rw in enumerate(wn_data):
    for ftr in rw:
        try:
            int(rw[ftr])
        except:
            ptntl_nlls.append({'index':indx, 'data': rw})
print(ptntl_nlls)
#returns empty list, so no void nulls

#prep data (split, scale)

#split data 70/30 for train and test

#test with variodu learning rates, and thesholds -- try OvO and OvR?
