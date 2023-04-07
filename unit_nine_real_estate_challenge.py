#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:54:21 2023

@author: cmwever73

End of Module 1 - Real Estate Challenge. Will try to all this with
numpy/pandas for sake of leaning library and future exam

data:
transaction_date - the transaction date 
            (for example, 2013.250=2013 March, 2013.500=2013 June, etc.)
house_age - the house age (in years)
transit_distance - the distance to the nearest light rail station (in meters)
local_convenience_stores - the number of convenience stores within walking distance
latitude - the geographic coordinate, latitude
longitude - the geographic coordinate, longitude
price_per_unit house price of unit area (3.3 square meters)

Your challenge is to explore and prepare the data, identify predictive 
features that will help predict the price_per_unit label, and train a 
regression model that achieves the lowest Root Mean Square Error (RMSE) you 
can achieve (which must be less than 7) when evaluated against a 
test subset of data.


Once that is done, save your trained model, and then use it to predict 
the price-per-unit for the following real estate transactions:
    
2013.167	16.2	289.3248	5	24.98203	121.54348
2013.000	13.6	4082.015	0	24.94155	121.50381

"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, export_text

rlest_df = pd.read_csv('data/real_estate.csv')
print(rlest_df.head())

#what are the numeric fields?
print(rlest_df.columns)
print(rlest_df.head())

label = rlest_df['price_per_unit']

#plot label hist and boxplot
fg, ax = plt.subplots(2,1, sharex= True, figsize=(10,14))
ax[0].hist(label, bins=100)
ax[0].set_ylabel('Freq')
ax[0].axvline(label.mean(), color='#FDC298', linestyle=':', linewidth=2)
ax[0].axvline(label.median(), color='#8950DC', linestyle=':', linewidth=2)
ax[1].boxplot(label, vert=False)
ax[1].x_label('Price per Unit')


#hists for numeric fields
flds = list(rlest_df.columns)
for fld in flds:
    if fld not in ['price_per_unit', 'transaction_date']:
        fig = plt.figure(figsize=((10,14)))
        ax = fig.gca()
        ax.hist(rlest_df[fld], bins=100)
        ax.set_ylabel('Freq')
        ax.set_xlabel(fld)
        ax.axvline(rlest_df[fld].mean(), color='#FDC298', linestyle=':', linewidth=2)
        ax.axvline(rlest_df[fld].median(), color='#8950DC', linestyle=':', linewidth=2)
plt.show()



#check all transaction dates are in 2013
not_2013 = [date for date in list(rlest_df.transaction_date) if date > 2013.999 or date < 2013]
print(len(not_2013))
#there are some 2012 date points




#bar chart for cat field
counts = rlest_df['transaction_date'].value_counts().sort_index()
fig = plt.figure(figsize=((9,12)))
ax = fig.gca()
counts.plot.bar(ax=ax, color = '#F4D03F')
ax.set_title('Transaction Date Counts')
ax.set_xlabel('Transaction Date')
ax.set_ylabel('Counts')
plt.show()



