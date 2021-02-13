'''
This script will apply decision trees and random forests to the 
mortality_rt_data.csv using scikit-learn
'''

import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# read in data and prep for tree
mortality = pd.read_csv(str(Path(os.path.split(__file__)[0]).parents[0] /
    'data/') + '/mortality_rt_data.csv')
# clean up missing and summary values
mortality_xy = mortality.loc[mortality['mortality_rt'].notna(), :]
mortality_xy = mortality_xy[(mortality_xy.state!='Overall') &
    (mortality_xy.gender!='Overall') & (mortality_xy.race!='Overall')]
# separate y data
y_raw = mortality_xy.loc[:, 'y']
#create binary y for gini
y_bin = (y_raw >= 0.005).astype('int64')
# separate x data
x = mortality_xy.loc[:, ['state', 'gender', 'race', 'Y_lat', 'X_lon']]
# split to training and test
x_train, x_test, y_train, y_test = train_test_split(x, y_raw, random_state=75)

# one hot encoding for categorical vars
column_trans = make_column_transformer(
    (OneHotEncoder(), ['state', 'gender', 'race']),
    remainder='passthrough'
    )

# create regressor
tree = DecisionTreeRegressor()

# create pipeline to preprocess data and fit regressor
pipe = make_pipeline(column_trans, tree)
print(cross_val_score(pipe, x_train, y_train, cv=10, scoring='r2').mean())

# mortality_xy = 



# # fit against data
# tree_fit = tree.fit(x_train, y_train) 
# # make predictions
# y_pred = tree_fit.predict(x_test)
# # view confusion confusion_matrix
# print(confusion_matrix(y_test, y_pred))