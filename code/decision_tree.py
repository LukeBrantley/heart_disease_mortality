## Steps
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

## For regression, y_pred = avg of y_obs w/in leaf node
## For classification, y_pred = max occurring y_obs in node?
# scaling unnecessary
import pandas as pd
import numpy as np
import os

def gini_idx(target_col):
    # Gini for all branches of all remaining attributes = (p^2 + (1-p)^2)
    val, counts = np.unique(target_col, return_counts=True)
    p = (counts[0] / sum(counts))

    return p ** 2 + (1 - p) ** 2

def calc_gini(target_col, split_path=1):
    # get gini index for single branch
    left = target_col[:split_path]
    right = target_col[split_path:]

    # Weight each branch's Gini by probability of occurence (ratio: # X=a / (# X=a + # X=b))
    left_gini = gini_idx(left)
    wtd_left_gini = (len(left) / len(target_col)) * left_gini
    right_gini = gini_idx(right)
    wtd_right_gini = (len(right) / len(target_col)) * right_gini
    
    return wtd_left_gini + wtd_right_gini

def gini(target_col, max_tries=10):
    # find optimal split of x-var to max IG, min entropy
    best_gini = 1
    best_split_pt = 1
    for split_pt in np.arange(
        start = 1,
        stop = len(target_col),
        step = len(target_col)/max_tries,
        dtype = int
        ):
        new_gini = calc_gini(target_col, split_pt)
        # we want highest gini, most homogeneous split
        if new_gini > best_gini:
            best_gini = new_gini
            best_split_pt = split_pt
    
    return best_gini, best_split_pt

def entropy(target_col):
    # get entropy of single branch
    # entropy = -sum(p(x)*log(p(x)
    val, count = np.unique(target_col, return_counts=True)
    entropy = -1 * sum(
        [count[a] / sum(count) * np.log(count[a] / sum(count)) for a in range(len(val))]
        )

    return entropy

def calc_ig(target_col, split_path=1):
    # IG for binary split
    left = target_col[:split_path]
    right = target_col[split_path:]

    left_entropy = entropy(left)
    right_entropy = entropy(right)

    parent_entropy = entropy(target_col)

    return parent_entropy - (left_entropy + right_entropy) / 2

def info_gain(target_col, max_tries=10):
    # find optimal split of x-var to max IG, min entropy
    best_ig = 1
    best_split_pt = 1
    for split_pt in np.arange(
        start = 1,
        stop = len(target_col),
        step = len(target_col)/max_tries,
        dtype = int
        ):
        new_ig = calc_ig(target_col,split_pt)
        # we want highest info gain... lowest entropy (randomness)
        if new_ig > best_ig:
            best_ig = new_ig
            best_split_pt = split_pt
    
    return best_ig, best_split_pt

def get_ASM(target_col,type='ig'):
    
    if type == 'ig':
        # calculate Information Gain attribute selection measure
        ig_val, split_pt = info_gain(target_col, max_tries=10)
        return ig_val, split_pt
    elif type == 'gini':
        # calculate Gini attribute selection measure
        gini_val, split_pt = gini(target_col, max_tries=10)
        return gini_val, split_pt
    else:
        raise Exception('Please enter ig or gini for ASM type')

if __name__ == '__main__':

    # read in data and prep for tree
    data = pd.read_csv(os.path.split(__file__)[0] + '\clean_data.csv')
    y = data['mortality_rt']
    x = data[['state', 'gender', 'race']]

    # test columns by ASM
    best_asm = 0
    best_split_pt = 1
    best_target = []
    for (target_nm, target_col) in x.iteritems():
        target = target_col.copy().sort_values()
        asm, split_pt = get_ASM(target,type='gini')
        if asm > best_asm:
            best_target = target_nm
            best_asm = asm
            best_split_pt = split_pt
        print(target_nm, best_asm, best_split_pt)
    
    print('best target: {}, {}, {}'.format(best_target, best_asm, best_split_pt))