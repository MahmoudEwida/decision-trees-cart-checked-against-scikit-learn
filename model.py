"""
Decision Trees: CART, Checked Against Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - gini
import numpy as np
def gini(y):
    # TODO: 1 - sum(p_k^2); 0.0 for an empty array.
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    
    return float(1.0 - np.sum(probabilities ** 2))

# Step 2 - best_split
def best_split(X, y, min_samples_leaf=1):
    n_samples, n_features = X.shape
    if n_samples <= 1:
        return None, None, 0.0
    
    parent_impurity = gini(y)
    if parent_impurity == 0.0:
        return None, None, 0.0

    best_feature = None
    best_threshold = None
    best_gain = 0.0

    for feature_idx in range(n_features):
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)
        
        if len(unique_values) <= 1:
            continue
            
        thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
        
        for threshold in thresholds:
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf:
                continue
            
            weight_left = len(y_left) / n_samples
            weight_right = len(y_right) / n_samples
            child_impurity = (weight_left * gini(y_left)) + (weight_right * gini(y_right))
            
            gain = parent_impurity - child_impurity
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, (float(best_threshold) if best_threshold is not None else None), best_gain

# Step 3 - grow_tree
def grow_tree(X, y, max_depth=2, min_samples_leaf=1, depth=0):
    # TODO: recursive CART with nested dicts; leaves hold the majority class.
    n = len(y)

    classes,counts = np.unique(y, return_counts = True)
    majority_class = int(classes[np.argmax(counts)])



    leaf_node = {
        'leaf' : True,
        'value' : majority_class,
        'n': int(n)
    }

    if len(classes) <= 1 or depth>=max_depth:
        return leaf_node


    feature, threshold, gain = best_split(X, y, min_samples_leaf=min_samples_leaf)

    if feature is None or gain <= 0.0 : 
        return leaf_node

    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask

    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]

    
    if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf:
        return leaf_node
    
    left_subtree = grow_tree(
        X_left, y_left, 
        max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf, 
        depth=depth + 1
    )
    right_subtree = grow_tree(
        X_right, y_right, 
        max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf, 
        depth=depth + 1
    )

    return {
        "leaf": False,
        "feature": int(feature),
        "threshold": float(threshold),
        "n": int(n),
        "left": left_subtree,
        "right": right_subtree
    }

# Step 4 - predict_tree
import numpy as np

def predict_one(tree, x):
    node = tree
    while not node["leaf"]:
        if x[node["feature"]] <= node["threshold"]:
            node = node["left"]    
        else:
            node = node["right"]   
            

    return node["value"]

def predict_tree(tree, X):
    X = np.asarray(X)
    predictions = [predict_one(tree, row) for row in X]
    return np.array(predictions, dtype=int)

# Step 5 - fit_sklearn_tree
from sklearn.tree import DecisionTreeClassifier

def fit_sklearn_tree(X, y, max_depth=2, min_samples_leaf=1, random_state=42):
    clf = DecisionTreeClassifier(
        max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf, 
        random_state=random_state
    )
    
    clf.fit(X, y)
    
    return clf

# Step 6 - sklearn_splits
def sklearn_splits(clf):
    # TODO: [(feature, threshold)] for internal nodes in node-index order, thresholds rounded to 3 decimals.
    tree = clf.tree_
    splits   = []

    for i in range(tree.node_count):
        if tree.children_left[i] != -1:
            feature = int(tree.feature[i])
            threshold = round(float(tree.threshold[i]),3)
            splits.append((feature,threshold))
    return splits

# Step 7 - compare_trees
import numpy as np

def extract_my_splits(node):
    splits = []

    def traverse(curr):
        if curr is None or curr["leaf"]:
            return
        splits.append((int(curr["feature"]), round(float(curr["threshold"]), 3)))
        traverse(curr["left"])
        traverse(curr["right"])

    traverse(node)
    return splits

def compare_trees(X, y, max_depth=2):
    my_tree = grow_tree(X, y, max_depth=max_depth)

    sk_clf = fit_sklearn_tree(X, y, max_depth=max_depth, random_state=42)

    my_splits = extract_my_splits(my_tree)
    sk_splits = sklearn_splits(sk_clf)

    same_splits = (my_splits == sk_splits)

    my_preds = predict_tree(my_tree, X)
    sk_preds = sk_clf.predict(X)
    agreement = float(np.mean(my_preds == sk_preds))

    return {
        'same_splits': bool(same_splits),
        'agreement': float(agreement),
        'my_splits': my_splits
    }

# Step 8 - moons_data
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def moons_data(n_samples=300, noise=0.25, random_state=42, test_size=0.3):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

# Step 9 - overfit_vs_regularized
from sklearn.tree import DecisionTreeClassifier

def overfit_vs_regularized(X_train, X_test, y_train, y_test, min_samples_leaf=5):
    clf_free = DecisionTreeClassifier(random_state=42)
    clf_free.fit(X_train, y_train)
    
    clf_reg = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=42)
    clf_reg.fit(X_train, y_train)
    
    return {
        'free': {
            'train_acc': float(clf_free.score(X_train, y_train)),
            'test_acc': float(clf_free.score(X_test, y_test)),
            'leaves': int(clf_free.get_n_leaves())
        },
        'regularized': {
            'train_acc': float(clf_reg.score(X_train, y_train)),
            'test_acc': float(clf_reg.score(X_test, y_test)),
            'leaves': int(clf_reg.get_n_leaves())
        }
    }

# Step 10 - rotation_sensitivity
import numpy as np

def rotation_sensitivity(X_train, X_test, y_train, y_test, degrees=45.0, min_samples_leaf=5):
    clf_orig = fit_sklearn_tree(X_train, y_train, max_depth=None, min_samples_leaf=min_samples_leaf)
    original_acc = float(clf_orig.score(X_test, y_test))
    
    theta = np.radians(degrees)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    X_train_rot = X_train @ R.T
    X_test_rot = X_test @ R.T
    
    clf_rot = fit_sklearn_tree(X_train_rot, y_train, max_depth=None, min_samples_leaf=min_samples_leaf)
    rotated_acc = float(clf_rot.score(X_test_rot, y_test))
    
    drop = round(original_acc - rotated_acc, 4)
    
    return {
        "original_acc": original_acc,
        "rotated_acc": rotated_acc,
        "drop": drop
    }

# Step 11 - regression_tree
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def regression_tree(X, y, max_depth=2, random_state=42):
    dtr = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    dtr.fit(X, y)
    
    preds = dtr.predict(X)

    
    train_mse = float(mean_squared_error(y, preds))
    
    return {
        "model": dtr,
        "n_distinct_predictions": int(len(np.unique(preds))),
        "train_mse": float(mean_squared_error(y, preds))
    }

# Step 12 - tree_rules
from sklearn.tree import export_text

def tree_rules(clf, feature_names):
    return export_text(clf, feature_names=list(feature_names))

# Step 13 - save_and_reload_tree
import joblib

def save_and_reload_tree(clf, path):
    joblib.dump(clf, path)
    
    reloaded_clf = joblib.load(path)
    
    return reloaded_clf

# Step 14 - predict_species
import numpy as np

def predict_species(clf, measurements, target_names):
    preds = clf.predict(measurements)
    
    target_names = np.asarray(target_names)
    return target_names[preds].tolist()

