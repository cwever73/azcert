#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:03:43 2023

@author: cmwever73

Unit 2 of Module: Train and Evaluate Regression Models
"""
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



#assumes youre in repo
flnm = 'data/bikes.csv'

with open(flnm) as f:
    bike_data = f.readlines()

hdrs = bike_data[0].split(',')
#clean up '\n' too
hdrs = [hdr.replace('\n', '') for hdr in hdrs]
#list of dictionaries holding flights
bikes = [{hdrs[indx]:data.replace('\n', '') for indx, data in enumerate(flght.split(','))} for flght in bike_data[1:]]
#now have list of dictionaries holding flight data

bike_data = pd.read_csv(flnm)
bike_data.head()

bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day
bike_data.head(32)


numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
bike_data[numeric_features + ['rentals']].describe()

#plot to look at summary stats:
label = bike_data['rentals']

# Create a figure for 2 subplots (2 rows, 1 column)
fig, ax = plt.subplots(2, 1, figsize = (9,12))

#histogram   
ax[0].hist(label, bins=100)
ax[0].set_ylabel('Frequency')


# Add stat lines
ax[0].axvline(label.mean(), color='magenta', linestyle='dashed', linewidth=2)
ax[0].axvline(label.median(), color='cyan', linestyle='dashed', linewidth=2)

# boxplot   
ax[1].boxplot(label, vert=False)
ax[1].set_xlabel('Rentals')

# Add a title to the Figure
fig.suptitle('Rental Distribution')
fig.show()

#plot hists for numeric data
for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = bike_data[col]
    feature.hist(bins=100, ax = ax)
    ax.axvline(feature.mean(), color='magenta', linestyle='dashed', linewidth=2)
    ax.axvline(feature.median(), color='cyan', linestyle='dashed', linewidth=2)
    ax.set_title(col)
plt.show()


#plot bar charts for cat data
categorical_features = ['season','mnth','holiday','weekday','workingday','weathersit', 'day']

for col in categorical_features:
    counts = bike_data[col].value_counts().sort_index()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    counts.plot.bar(ax = ax, color='steelblue')
    ax.set_title(col + ' counts')
    ax.set_xlabel(col) 
    ax.set_ylabel("Frequency")
plt.show()

#look thru for correlation between numeric features and label (num of rentals)
for col in numeric_features:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = bike_data[col]
    label = bike_data['rentals']
    correlation = feature.corr(label)
    plt.scatter(x=feature, y=label)
    plt.xlabel(col)
    plt.ylabel('Bike Rentals')
    ax.set_title('rentals vs ' + col + '- correlation: ' + str(correlation))
plt.show()



#After looking at charts -- pick out features that impact rental numbers
# separate features and labels
obs_x = bike_data[['season','mnth', 'holiday','weekday','workingday','weathersit','temp', 'atemp', 'hum', 'windspeed']].values
obs_y = bike_data['rentals'].values
print('Features:',obs_x[:10], '\nLabels:', obs_y[:10], sep='\n')

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(obs_x, obs_y, test_size=0.30, random_state=0)
print ('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))

# Fit a linear regression model on the training set
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
    p = np.poly1d(z)
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

#not great +- 500 rentals and just over 50% on R**2 measure

##################END UNIT 2 -- START UNIT 5 #################################

#try another linearu regressor using Lasso model
####TODO: learn how Lasso model works
model = Lasso().fit(X_train, y_train)
print (model, "\n")

# Evaluate the model using the test data
predictions = model.predict(X_test)
errrs_vrbs(y_test, predictions)

# Plot predicted vs actual
obs_v_pred_plt(y_test, predictions)

#Decision Tree try time!

####TODO: Learn Tree Regressor
model = DecisionTreeRegressor().fit(X_train, y_train)
print (model, "\n")

# Visualize the model tree
tree = export_text(model)
print(tree)

#great. now what.

# Evaluate the model using the test data
predictions = model.predict(X_test)
errrs_vrbs(y_test, predictions)

# Plot predicted vs actual
obs_v_pred_plt(y_test, predictions)

#not better. time to try endemble.

####TODO: Learn how Random Forest works
model = RandomForestRegressor().fit(X_train, y_train)
print (model, "\n")

# Evaluate the model using the test data
predictions = model.predict(X_test)
errrs_vrbs(y_test, predictions)

#plot
obs_v_pred_plt(y_test, predictions)

#ok ... gettting better.

# Fit a lasso model on the training set
model = GradientBoostingRegressor().fit(X_train, y_train)
print (model, "\n")

# Evaluate the model using the test data
predictions = model.predict(X_test)
errrs_vrbs(y_test, predictions)

#plot
obs_v_pred_plt(y_test, predictions)

#a wee bit better.


##################END UNIT 2 -- START UNIT 7 #################################

# Use a Gradient Boosting algorithm
alg = GradientBoostingRegressor()

# Try these hyperparameter values
params = {
 'learning_rate': [0.1, 0.5, 1.0],
 'n_estimators' : [50, 100, 150]
 }

# Find the best hyperparameter combination to optimize the R2 metric
score = make_scorer(r2_score)
gridsearch = GridSearchCV(alg, params, scoring=score, cv=3, return_train_score=True)
gridsearch.fit(X_train, y_train)
print("Best parameter combination:", gridsearch.best_params_, "\n")

# Get the best model
model=gridsearch.best_estimator_
print(model, "\n")

# Evaluate the model using the test data
predictions = model.predict(X_test)
errrs_vrbs(y_test, predictions)

#plot
obs_v_pred_plt(y_test, predictions)

#Now do some preprocessing to the raw data

# Define preprocessing for numeric columns (scale them)
numeric_features = [6,7,8,9]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Define preprocessing for categorical features (encode them)
categorical_features = [0,1,2,3,4,5]
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', GradientBoostingRegressor())])


# fit the pipeline to train a linear regression model on the training set
model = pipeline.fit(X_train, (y_train))
print (model)


