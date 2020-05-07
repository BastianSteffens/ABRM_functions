#!/usr/bin/env python
# coding: utf-8

# ### Objectives of this notebook:
# - display output from ABRM
# - perform analysis on output. eg. feature importance and model selection

# In[1]:


import numpy as np
import pandas as pd
import plotly.io as pio
import ABRM_functions

pio.renderers.default = "notebook"


# ### Import data
# - one df with FD performance of each particle  
# - one df with associated particle position  
# - one dicct with initial setup that would allow reproduction of results

# In[2]:


dataset = ["2020_04_29_15_49"]
misfit_tolerance = 0.1

df_performance,df_position,setup_all, FD_targets = ABRM_functions.read_data(dataset)

print("Number of models and parameters:")
display(df_position.shape)
print("Number of particles:")
display(df_position.particle_no.max()+1)
print("Number of Iterations:")
display(df_position.iteration.max()+1)
display(df_performance.head())
display(df_position.head())


# ### Plot performance

# In[3]:


ABRM_functions.plot_performance(df_performance,df_position,FD_targets,setup_all,dataset,misfit_tolerance)


# ### Boxplots parameters
# explore if ranges of parameters need potential modifications

# In[4]:


ABRM_functions.plot_box(df = df_position,setup_all = setup_all, dataset = dataset)


# ### Histogram Paramters

# In[5]:


ABRM_functions.plot_hist(df = df_position,setup_all = setup_all, dataset = dataset,misfit_tolerance = None)


# ### Histograms for best models parameters

# In[6]:


ABRM_functions.plot_hist(df = df_position,setup_all = setup_all, dataset = dataset,misfit_tolerance = misfit_tolerance)


# ### Cluster best models with UMAP and HDBSCAN

# In[8]:


df_best = ABRM_functions.best_model_selection_UMAP_HDBSCAN(df = df_position,dataset =dataset,setup_all = setup_all,
                                                           n_neighbors= 5, min_cluster_size=5, misfit_tolerance = misfit_tolerance,use_UMAP = True)


# ### Build best performing models for flow simulation

# In[9]:


best_models = ABRM_functions.save_best_clustered_models(df_best = df_best, datasets = dataset)


# ### Feature importance - model explainability

# In[9]:


import shap
from sklearn.ensemble import RandomForestRegressor


# In[10]:


columns = setup_all[dataset[0]]["columns"]
X_train = df_position[columns]
Y_train = df_position.LC


# In[11]:


# load JS visualization code to notebook
shap.initjs()


# In[12]:


model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
model.fit(X_train, Y_train)


# In[13]:


shap_values = shap.TreeExplainer(model).shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")


# In[14]:


# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[196,:], X_train.iloc[196,:])


# In[15]:


# visualize the training set predictions
shap.force_plot(explainer.expected_value, shap_values, X_train)


# In[16]:


# create a dependence plot to show the effect of a single feature across the whole dataset
shap.dependence_plot("MatrixpermY", shap_values, X_train)


# In[17]:


shap.summary_plot(shap_values, X_train)


# In[18]:


shap.summary_plot(shap_values, X_train, plot_type="bar")


# ### Feature importance

# In[10]:


from sklearn.preprocessing import scale
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
# Keep this here in mind!
#  This is because the feature importance method of random forest favors features that have high cardinality. In our dataset, age had 55 unique values, and this caused the algorithm to think that it was the most important feature.
from matplotlib import pyplot as plt


# In[21]:


columns = setup_all[dataset[0]]["columns"]
trainDataSet_X = df_position[columns]
trainDataSet_Y = df_position.LC


# In[22]:


trainDataSet_X_scaled = scale(trainDataSet_X)


# In[23]:


X, y = shuffle(trainDataSet_X_scaled, trainDataSet_Y, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]


# In[24]:


# Fit regression model
params = {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)


# In[25]:


# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')


# In[26]:


# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplots(figsize= (10,20))

plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, trainDataSet_X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[27]:


#import required packages
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')

rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()


# In[ ]:




