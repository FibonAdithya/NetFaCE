This is a read me file to explain the changes that we've made to code and how to use it.

# Classification
This script provides utilities for preprocessing, training, evaluating, and visualizing machine learning models to classify chordal graphs based on graph metrics.

## Overview
The script includes the following functionalities:

- Preprocessing: Clean and prepare graph data for machine learning models.
- Model Training: Train and evaluate various classifiers, including Logistic Regression, K-Nearest Neighbors, SVM, Decision Trees, Random Forest, and XGBoost.
- Model Comparison: Compare models based on their performance metrics.
- Visualization: Explore feature importance, logistic regression coefficients, and decision tree structures.

## Usage Examples

### 1. Preproccessing
Prepares the data by handling missing values, scaling features, and splitting into training and test sets.

```python
from classification import preprocessing

# Dataset with features and chordality labels
X_train, X_test, y_train, y_test = preprocessing(dataset, split=0.3)
```

### 2. Training and Evaluating Models
```python
import classification

#Logistic Regression
lr = classification.logistic_regression(X_train, X_test, y_train, y_test)

#K-nearest neighbour
knn = K_nearest_neighbour(X_train, X_test, y_train, y_test)

#Support Vector Machine
svm = support_vector_machine(X_train, X_test, y_train, y_test)

#Decision Tree
dt = decision_tree_classifier(X_train, X_test, y_train, y_test)

#Random Forest
rf = random_forrest(X_train, X_test, y_train, y_test)

#XGBoost
xgb = xgboost_classifier(X_train, X_test, y_train, y_test)
```

### 3. Comparing Models
Use the compare_models function to train all classifiers and view their performance:
```python
from classification import compare_models

results = compare_models(X_train, X_test, y_train, y_test)
print(results)
```

### 4. Visualizing Feature Importance and Coefficients
```python
import classification
from script import plot_feature_importance
import matplotlib.pyplot as plt
#Feature Importance (Random Forest & XGBoost)
rf_importance, xgb_importance = classification.plot_feature_importance(X_train, y_train)

#Logistic Regression Coefficients
coef_df = classification.plot_logistic_coefficients(X_train, y_train)

#Decision Tree Visualization
tree_clf = classification.visualize_decision_tree(X_train, y_train, max_depth=3)

plt.show()
```