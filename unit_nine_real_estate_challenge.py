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
import datetime
import joblib
import matplotlib.pyplot as plt
import math
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
ax[1].set_xlabel('Price per Unit')


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
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
rlest_df['month'] = rlest_df.apply(lambda row: months[int(round(math.modf(row.transaction_date)[0], 3)*11)], axis=1)
rlest_df['year'] = rlest_df.apply(lambda row: str(math.modf(row.transaction_date)[1]), axis=1)
rlest_df.query('year==2012.0')['A']
#bar chart for cat field


from bokeh.palettes import Spectral5
from bokeh.plotting import figure, show
from bokeh.sampledata.autompg import autompg_clean as df
from bokeh.transform import factor_cmap

df.cyl = df.cyl.astype(str)
df.yr = df.yr.astype(str)

group = rlest_df.groupby(['year', 'month'])

index_cmap = factor_cmap('yr_mnth', palette=Spectral5, factors=sorted(rlest_df.year.unique()), end=1)

p = figure(width=800, height=300, title="Counts of Transaction Dates by Month and Year",
           x_range=group, toolbar_location=None, tooltips=[("Counts", "@transaction_date"), ("Year, Month", "@yr_mnth")])

p.vbar(x='yr_mnth', top='transaction_date', width=1, source=group,
       line_color="white", fill_color=index_cmap)

p.y_range.start = 0
p.x_range.range_padding = 0.05
p.xgrid.grid_line_color = None
p.xaxis.axis_label = "Manufacturer grouped by # Cylinders"
p.xaxis.major_label_orientation = 1.2
p.outline_line_color = None

show(p)



