''' 
This script calculates Attribute Selection Measures for data set.
I am not going to go into the recursion to build a full tree, but
will expand upon this using the quick functions in scikit-learn.

Steps
### 1. split x and y data to train and test sets
### 2. calculate and compare Attribute Selection Measures (ASM) of using each x-var as next node in tree
#### a. This includes information gain (IG), Gini Index, and Gain Ratio
##### i. IG: select attribute with largest disparity in y_obs... lowest entropy
###### *. entropy=0 when a split contains only 1 y value, entropy=1 when split has equal # of y value outcomes
###### **. entropy = -sum(p(x)*log(p(x)))
###### ***. IG = entropy(y) - avg_entropy(children) ... entropy(y) = entropy(parent node)
##### ii. Gini works with categorical target variable “Success” or “Failure”, only binary splits
###### *. High Gini = high homogeneity = more likely to use as next node
###### **. Gini for all branches of all remaining attributes = (p^2 + (1-p)^2)
###### ***. Weight each branch's Gini by probability of occurence (ratio: # X=a / (# X=a + # X=b))
'''
## For regression, y_pred = avg of y_obs w/in leaf node
## For classification, y_pred = max occurring y_obs in node?
# scaling unnecessary
import pandas as pd
import numpy as np
import os
from pathlib import Path

def gini_idx(target_col, y):
    # Gini for all branches of all remaining attributes = (p^2 + (1-p)^2)
    y_red = y[y.index.isin(target_col.index)]
    val, counts = np.unique(y_red, return_counts=True)
    p = (counts[0] / sum(counts))

    return p ** 2 + (1 - p) ** 2

def calc_gini(target_col, y, split_path=1):
    # get gini index for single branch
    left = target_col[:split_path]
    right = target_col[split_path:]

    # Weight each branch's Gini by probability of occurence (ratio: # X=a / (# X=a + # X=b))
    left_gini = gini_idx(left, y)
    wtd_left_gini = (len(left) / len(target_col)) * left_gini
    right_gini = gini_idx(right, y)
    wtd_right_gini = (len(right) / len(target_col)) * right_gini
    
    return wtd_left_gini + wtd_right_gini

def gini(target_col, y, max_tries=10):
    # find optimal split of x-var to max IG, min entropy
    best_gini = 1
    best_split_pt = 1
    for split_pt in np.arange(
        start = 1,
        stop = len(target_col),
        step = len(target_col)/max_tries,
        dtype = int
        ):
        new_gini = calc_gini(target_col, y, split_pt)
        # we want highest gini, most homogeneous split
        if new_gini > best_gini:
            best_gini = new_gini
            best_split_pt = split_pt
    
    return best_gini, best_split_pt

def entropy(target_col, y):
    # get entropy of single branch
    # entropy = -sum(p(x)*log(p(x)
    y_red = y[y.index.isin(target_col.index)]
    val, count = np.unique(y_red, return_counts=True)
    entropy = -1 * sum(
        [count[a] / sum(count) * np.log(count[a] / sum(count)) for a in range(len(val))]
        )

    return entropy

def calc_ig(target_col, y, split_path=1):
    # IG for binary split
    left = target_col[:split_path]
    right = target_col[split_path:]

    left_entropy = entropy(left, y)
    right_entropy = entropy(right, y)

    parent_entropy = entropy(target_col, y)

    return parent_entropy - (left_entropy + right_entropy) / 2

def info_gain(target_col, y, max_tries=10):
    # find optimal split of x-var to max IG, min entropy
    best_ig = 1
    best_split_pt = 1
    for split_pt in np.arange(
        start = 1,
        stop = len(target_col),
        step = len(target_col) / max_tries,
        dtype = int
        ):
        new_ig = calc_ig(target_col, y, split_pt)
        # we want highest info gain... lowest entropy (randomness)
        if new_ig > best_ig:
            best_ig = new_ig
            best_split_pt = split_pt
    
    return best_ig, best_split_pt

def calc_sse(target_col, y_col):
    y_red = y_col[y_col.index.isin(target_col.index)]
    y_mean = y_red.mean()
    sse_val = sum([(y_mean - y_val) ** 2 for y_val in y_red])
    
    return sse_val
    
def sse_split(target_col, y_col, split_pt):
    left = target_col[:split_pt]
    right = target_col[split_pt:]

    left_sse = calc_sse(left, y_col)
    right_sse = calc_sse(right, y_col)
    
    sse = left_sse + right_sse
    
    return sse
    
def sse(target_col, y, max_tries=100):
    # find optimal split of x-var to max IG, min entropy
    best_sse = np.inf
    best_split_pt = 1
    for split_pt in np.arange(
        start = 1,
        stop = len(target_col),
        step = len(target_col) / max_tries,
        dtype = int
        ):
        new_sse = sse_split(target_col, y, split_pt)
        # we want highest info gain... lowest entropy (randomness)
        if new_sse < best_sse:
            best_sse = new_sse
            best_split_pt = split_pt
    
    return best_sse, best_split_pt

def get_ASM(target_col, y, type='ig', max_tries=100):
    
    if type == 'ig':
        # calculate Information Gain attribute selection measure
        ig_val, split_pt = info_gain(target_col, y, max_tries=max_tries)
        return ig_val, split_pt
    elif type == 'gini':
        # calculate Gini attribute selection measure
        gini_val, split_pt = gini(target_col, y, max_tries=max_tries)
        return gini_val, split_pt
    elif type == 'sse':
        # calculate Gini attribute selection measure
        sse_val, split_pt = sse(target_col, y, max_tries=max_tries)
        return sse_val, split_pt
    else:
        raise Exception('Please enter ig or gini for ASM type')

if __name__ == '__main__':

    # read in data and prep for tree
    data = pd.read_csv(str(Path(os.path.split(__file__)[0]).parents[0] / 
        'data/') + '/mortality_rt_data.csv')
    data.dropna(subset=['mortality_rt'], inplace=True)
    data = data[(data.state!='Overall') & (data.gender!='Overall') &
                (data.race!='Overall')]
    y_raw = data['y'].copy()
    #create binary y for gini
    y_bin = (y_raw >= 0.005).astype('int64')
    
    x = data[['state', 'gender', 'race']]
    tot = data[['state', 'gender', 'race', 'mortality_rt']]
    #split into train and test
    y_type='cont'
    if y_type == 'bin':
        y_train = y_bin.sample(frac=0.75)
        y_test = y_bin.iloc[~y_bin.index.isin(y_train.index)]
    else:
        y_train = y_raw.sample(frac=0.75)
        y_test = y_raw.iloc[~y_raw.index.isin(y_train.index)]
    x_train = x.iloc[x.index.isin(y_train.index)]
    x_test = x.iloc[~x.index.isin(y_train.index)]
    
    tot_train = tot.iloc[tot.index.isin(y_train.index)]
    tot_test = tot.iloc[~tot.index.isin(y_train.index)]
    
    # test columns by ASM
    best_asm = np.inf
    best_split_pt = 1
    best_target = []
    x_train_pop = x_train.copy()
    tree_saver = []
    while len(x_train_pop) > 0:
        for (target_nm, target_col) in x_train_pop.iteritems():
            target = target_col.copy().sort_values()
            asm, split_pt = get_ASM(target, y_train, type='sse', max_tries=1000)
            print(target_nm, asm, split_pt)
            if asm < best_asm:
                best_target = target_nm
                best_asm = asm
                best_split_pt = split_pt
                #best_split_val = target[split_pt]
        x_train_pop.pop(best_target)
        tree_saver.append('best target: {}, {}, {}'.format(
            best_target, best_asm, best_split_pt)) #, best_split_val
    
    
    
    