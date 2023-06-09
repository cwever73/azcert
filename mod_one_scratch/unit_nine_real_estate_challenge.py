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
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure, show
from bokeh.transform import factor_cmap
from collections import Counter
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, export_text


#BYO Scripts
# from unit_one_flight_challenge import stats as u1fc_stats
#jk, gotta clean up so it doesnt execute every line in there...

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


def stats(data):
    '''Given a list of numbers, return mean, median and mode of data'''
    if not isinstance(data, list):
        print('Not a valid datatype. Please input data in list form.')
    else:
        #sort for median
        data.sort()
        #calc mean
        data_avg = sum(data)/len(data)
        #calc median
        if len(data) % 2 == 0:
            data_med = (data[int(len(data)/2)] + data[int(len(data)/2) - 1])/2
        else:
            data_med = data[int(len(data)/2)]
        #calc mode
        data_md = [tpl[0] for tpl in Counter(data).most_common() if tpl[1] == Counter(data).most_common()[0][1]]
        
        return {'Mean': data_avg, 'Median': data_med, 'Mode': data_md, 
                'Min': min(data), 'Max': max(data)}

ppu_stats = stats(list(rlest_df.price_per_unit))

#get rid of outliers
q01 = rlest_df.price_per_unit.quantile(0.01)
q99 = rlest_df.price_per_unit.quantile(0.99)
# Get the variable to examine
rlest_df = rlest_df[(rlest_df.price_per_unit>q01) & (rlest_df.price_per_unit<q99)]

label = rlest_df['price_per_unit']
#plot after outliers are removed
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
mnths = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
rlest_df['month'] = rlest_df.apply(lambda row: mnths[int(round(math.modf(row.transaction_date)[0], 3)*11)], axis=1)
rlest_df['year'] = rlest_df.apply(lambda row: str(math.modf(row.transaction_date)[1]), axis=1)
td_cnts = {'2012': dict(rlest_df.query("year == '2012.0'")['month'].value_counts()),
           '2013': dict(rlest_df.query("year == '2013.0'")['month'].value_counts())}

#bar chart for cat field
plt_data = {}
yrs = ['2012', '2013']
plt_data['month'] = mnths
for yr in yrs:
    plt_data.setdefault(yr, [])
    for mnth in mnths:
        try:
            plt_data[yr].append(td_cnts[yr][mnth])
        except:
            plt_data[yr].append(0)

palette = ["#c9d9d3", "#718dbf"]

x = [(mnth, yr) for mnth in mnths for yr in yrs ]
counts = sum(zip(plt_data['2012'], plt_data['2013']), ()) # like an hstack
source = ColumnDataSource(data=dict(x=x, counts=counts))

p = figure(x_range=FactorRange(*x), height=350, title="Transaction Counts by Year and Month",
           toolbar_location=None, tools="")

p.vbar(x='x', top='counts', width=0.9, source=source, line_color="white",
       fill_color=factor_cmap('x', palette=palette, factors=yrs, start=1, end=2))

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None
show(p)

#looks like we got data from 2012 Aug to 2013 July
for fld in flds:
    if fld not in ['price_per_unit', 'transaction_date']:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.gca()
        ftr = rlest_df[fld]
        lbl = rlest_df['price_per_unit']
        corr = ftr.corr(lbl)
        plt.scatter(x=ftr, y=lbl)
        plt.xlabel(fld)
        plt.ylabel('Price Per Unit')
        ax.set_title('Price per Unit vs ' + fld + '- correlation: ' + str(corr))
plt.show()



#so, look at lat, lon, convinence stores, and travel dist (all > |.5| for corr)
#and month
rlest_df['month_num_eq'] = rlest_df.apply(lambda row: tuple(mnths).index(row.month)+1, axis=1)
obs_x = rlest_df[['transit_distance', 'local_convenience_stores', 'latitude',
                 'longitude','month_num_eq']].values
obs_y = rlest_df['price_per_unit'].values

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(obs_x, obs_y, test_size=0.30, random_state=0)
print ('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))

model = LinearRegression().fit(X_train, y_train)
print(model)


#predict using test set
predictions = model.predict(X_test)
np.set_printoptions(suppress=True)
print('Predicted labels: ', np.round(predictions)[:10])
print('Actual labels   : ' ,y_test[:10])


#compare predicted to observed with plot
def obs_v_pred_plt(tst, preds):
    fig = plt.figure(figsize=(10,10))
    plt.scatter(tst, preds)
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Daily Bike Share Predictions')
    # overlay the regression line
    z = np.polyfit(tst, preds, 1)
    print('Z: ', z)
    p = np.poly1d(z)
    print(p)
    plt.plot(tst,p(tst), color='magenta')
    fig.show()
    
obs_v_pred_plt(y_test, predictions)

#calc loss with libraries
####TODO: should do this by hand at some point (maybe in a challegne?)
def errrs_vrbs(tst, preds):
    print("MSE:", mean_squared_error(tst, preds))
    print("RMSE:", np.sqrt(mean_squared_error(tst, preds)))
    print("R2:", r2_score(tst, preds))

errrs_vrbs(y_test, predictions)

#not horribe at +-9 price per unit points off with each prediciton, but
#r2 is only just above 50%


#let's try Lasso
mdl_lsso = Lasso().fit(X_train, y_train)
lsso_preds = mdl_lsso.predict(X_test)
errrs_vrbs(y_test, lsso_preds)
obs_v_pred_plt(y_test, lsso_preds)



#try decision tree
mdl_dt0 = DecisionTreeRegressor().fit(X_train, y_train)
tree = export_text(mdl_dt0)
print(tree)
dt0_preds = mdl_dt0.predict(X_test)
errrs_vrbs(y_test, dt0_preds)
obs_v_pred_plt(y_test, dt0_preds)


#try ensemble
alg = GradientBoostingRegressor()

obs_x = rlest_df[['transit_distance', 'local_convenience_stores', 'latitude',
                 'longitude', 'month_num_eq', 'year']].values
obs_y = rlest_df['price_per_unit'].values

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(obs_x, obs_y, test_size=0.30, random_state=0)
print ('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))

# Try these hyperparameter values
params = {
 'learning_rate': [0.07, 0.08, 0.09, 0.1, 0.5, 1.0],
 'n_estimators' : [30, 40, 50, 100, 150]}


# Find the best hyperparameter combination to optimize the R2 metric
score = make_scorer(r2_score)
gridsearch = GridSearchCV(alg, params, scoring=score, cv=3, return_train_score=True)
gridsearch.fit(X_train, y_train)
print("Best parameter combination:", gridsearch.best_params_, "\n")

# Get the best model
mdl_grad_reg = gridsearch.best_estimator_
print(mdl_grad_reg.get_params())

#Now do some preprocessing to the raw data
num_fts = [0,1,2,3]
num_trnsfrmr = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Define preprocessing for categorical features (encode them)
cat_fts = [4,5]
cat_trnsfrmr = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_trnsfrmr, num_fts),
        ('cat', cat_trnsfrmr, cat_fts)])

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', GradientBoostingRegressor())])


grad_reg_mdl = pipeline.fit(X_train, (y_train))
grad_reg_preds = grad_reg_mdl.predict(X_test)
errrs_vrbs(y_test, grad_reg_preds)
obs_v_pred_plt(y_test, grad_reg_preds)



#try random forest
alg = RandomForestRegressor()

obs_x = rlest_df[['transit_distance', 'local_convenience_stores', 'latitude',
                 'longitude', 'month_num_eq', 'year']].values
obs_y = rlest_df['price_per_unit'].values

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(obs_x, obs_y, test_size=0.30, random_state=0)
print ('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))

# Try these hyperparameter values
params = {
 'max_depth': [4,5,6,7,8,9,10, 11, 12, 13],
 'n_estimators' : [30, 40, 50, 100, 150, 160, 170, 180, 190]}


# Find the best hyperparameter combination to optimize the R2 metric

score = make_scorer(r2_score)
gridsearch = GridSearchCV(alg, params, scoring=score, cv=3, return_train_score=True)
gridsearch.fit(X_train, y_train)
print("Best parameter combination:", gridsearch.best_params_, "\n")

# Get the best model
mdl_grad_reg = gridsearch.best_estimator_
print(mdl_grad_reg.get_params())

#Now do some preprocessing to the raw data
num_fts = [0,1,2,3]
num_trnsfrmr = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Define preprocessing for categorical features (encode them)
cat_fts = [4,5]
cat_trnsfrmr = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_trnsfrmr, num_fts),
        ('cat', cat_trnsfrmr, cat_fts)])

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', RandomForestRegressor())])


grad_reg_mdl = pipeline.fit(X_train, (y_train))
grad_reg_preds = grad_reg_mdl.predict(X_test)
errrs_vrbs(y_test, grad_reg_preds)
obs_v_pred_plt(y_test, grad_reg_preds)



