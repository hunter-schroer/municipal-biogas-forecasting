#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tpot
import pandas as pd
import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split


# In[2]:


features = pd.read_csv('SCADA-FINAL-1daybiogas-lags.csv')

features['Timestamp'] = pd.to_datetime(features['Timestamp'], format='%Y-%m-%d %H:%M:%S')
features.set_index('Timestamp', inplace=True)


# In[3]:


labels = np.array(features['Forecast'])
features = features.drop(['Forecast','day_sin', 'day_cos', 'D2_TEMPERATURE', 'Q-influent_MGD'], axis = 1)
feature_list = list(features.columns)

print('The shape of our features is:', features.shape)
features = np.array(features)


# In[4]:


x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)


# In[ ]:


cv = TimeSeriesSplit(n_splits=5, gap = 300)
model = TPOTRegressor(generations=500, population_size=100, scoring='neg_mean_squared_error', cv=cv, verbosity=2, random_state=1, n_jobs=-1)
# perform the search
model.fit(x_train, y_train)
# export the best model
model.export('SCADA-tpot_biogas_best_model-updated090723.py')


# In[ ]:




