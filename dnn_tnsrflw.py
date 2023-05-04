#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:33:15 2023

@author: cmwever73

Try Jpynb: Deep Neural Network (Tensorflow) from Azure ml-basics git repo

TensorFlow is a framework for creating machine learning models, including deep 
neural networks (DNNs). In this example, we'll use Tensorflow to create a 
simple neural network that classifies penguins into species based on the 
length and depth of their culmen (bill), their flipper length, and their body 
mass.

**Citation**: The penguins dataset used in the this exercise is a subset of 
data collected and made available by [Dr. Kristen Gorman]
(https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php) and the
[Palmer Station, Antarctica LTER](https://pal.lternet.edu/), a member of the 
[Long Term Ecological Research Network](https://lternet.edu/)

**Note**: In reality, you can solve the penguin classification problem easily 
using classical machine learning techniques without the need for a deep 
learning model; but it's a useful, easy to understand dataset with which to 
demonstrate the principles of neural networks in this notebook.

"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


#explore dataset -- running in azcert repo, so get data from ml-basics

#read in data
pngn_df = pd.read_csv('../ml-basics/data/penguins.csv')
pngn_data = [{pngn_df.columns[indx]: data 
           for indx, data in enumerate(pngn_df.loc[rw,:])} 
           for rw in range(len(pngn_df))]

# Deep Learning models work best when features are on similar scales
# In a real solution, we'd implement some custom normalization for each 
# feature, but to keep things simple we'll just rescale the FlipperLength and 
# BodyMass so they're on a similar scale to the bill measurements 

pngn_df['FlipperLength'] = pngn_df['FlipperLength']/10
pngn_df['BodyMass'] = pngn_df['BodyMass']/100
# The dataset is too small to be useful for deep learning
# So we'll oversample it to increase its size
for i in range(1,3):
    pngn_df = pngn_df.append(pngn_df)

pngn_data_tst = pngn_data
for i in range(1,3):
    pngn_data_tst.append([[data] for data in pngn_data])

# Display a random sample of 10 observations\r\n",
smpl = pngn_df.sample(10)

pngn_clsss = {0: 'Adelie', 1: 'Gentoo', 2: 'Chinstrap'}

#for my own sake, try minMax scale
lbl = 'Species'
obs_data = np.array([[data[ftr] for ftr in data if ftr != lbl]for data in pngn_data])

ftrs_scld = MinMaxScaler().fit_transform(obs_data)



