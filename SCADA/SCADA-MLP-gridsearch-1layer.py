#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_percentage_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
import math

data = pd.read_csv('SCADA-FINAL-1daybiogas-lags.csv')
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%Y-%m-%d %H:%M:%S')
data.set_index('Timestamp', inplace=True)

data.drop(columns=['day_sin', 'day_cos', 'D2_TEMPERATURE', 'Q-influent_MGD'
                  ], inplace=True)

cv=TimeSeriesSplit(gap = 300)


# In[5]:


from sklearn.preprocessing import RobustScaler, StandardScaler

x_train, x_test, y_train, y_test = train_test_split(
    data.drop('Forecast', axis=1), data['Forecast'], test_size=0.2,
    shuffle=False)

# Initialize scaler object
scaler = StandardScaler()

# Fit the scaler on the training dataset, but we have to exclude the sine and cosine terms
scaler.fit(x_train)

# Transform the training and testing dataset
x_train_s = pd.DataFrame(scaler.transform(x_train), index=x_train.index)
x_test_s = pd.DataFrame(scaler.transform(x_test), index=x_test.index)

# Add the column names back
x_train_s.columns = x_train.columns
x_test_s.columns = x_test.columns


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
random_seed=1
# Initialize Model
model_ann = MLPRegressor(max_iter=200,
                          early_stopping=True,
                          random_state=random_seed,
                          solver='adam',
                          learning_rate='invscaling',
                         )
# Create a 'grid' of ANN model fitting parameters to explore
param_grid = {
    'hidden_layer_sizes': [ (10,), (20,), (30,), (40,), (50,), (60,), (70,)],
    'alpha': [0.1, 1, 10, 25, 50, 75, 100, 200, 300, 400, 500],
    'learning_rate': ['invscaling', 'adaptive'],
    'solver': ['sgd', 'adam']
}

grid_search = GridSearchCV(model_ann,
                           param_grid,
                           cv=cv,
                           scoring='neg_mean_absolute_percentage_error',
                           n_jobs=-1
                           )

# Conduct a 'grid search' for the 'best' model
grid_search.fit(x_train_s.values, y_train)

# Print the best results
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", -grid_search.best_score_)

best_model = grid_search.best_estimator_
y_train_ann = best_model.predict(x_train_s.values)
y_test_ann = best_model.predict(x_test_s.values)


# In[15]:


mse_train_ann = mean_squared_error(y_train, y_train_ann)
mape_train_ann = mean_absolute_percentage_error(y_train, y_train_ann)
r2_train_ann = r2_score(y_train, y_train_ann)
rmse_train = math.sqrt(mse_train_ann)
mse_test_ann = mean_squared_error(y_test, y_test_ann)
mape_test_ann = mean_absolute_percentage_error(y_test, y_test_ann)
r2_test_ann = r2_score(y_test, y_test_ann)
rmse_test = math.sqrt(mse_test_ann)

# Print training and testing error
print(f"Training Mean Squared Error (MSE): {round(mse_train_ann,3)}")
print(f"Training Root Mean Squared Error (MSE): {round(rmse_train,3)}")
print(f"Training R-squared (R2) Score: {round(r2_train_ann,3)}")
print(f"Training Mean Absolute Percentage Error (MAPE): {round(mape_train_ann,3)}")
print(f"Testing Mean Squared Error (MSE): {round(mse_test_ann,3)}")
print(f"Testing Root Mean Squared Error (MSE): {round(rmse_test,3)}")
print(f"Testing R-squared (R2) Score: {round(r2_test_ann,3)}")
print(f"Testing Mean Absolute Percentage Error (MAPE): {round(mape_test_ann,3)}")

def adjusted_r2(r2, n, k):
    return 1 - ((1 - r2) * (n - 1)) / (n - k - 1)

n = len(x_test_s)  # Number of samples in test data
k = len(x_train_s.columns) # Number of predictors in the model

# Calculate Adjusted R-squared
adjusted_r2_value = adjusted_r2(r2_test_ann, n, k)
print("Adjusted R-squared:", round(adjusted_r2_value,3))
print("Number of variables:",k)
TeEI = (mape_test_ann*mape_test_ann*rmse_test)/adjusted_r2_value
print("TeEI:", round(TeEI,3))

