import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

def preprocessing(dataset, cols = None, split = 0.3):

    # Create DataFrame and add target variable
    df = pd.DataFrame(dataset[0])
    df["Chordal"] = dataset[1]
    
    # Handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    df = pd.DataFrame(
        imputer.fit_transform(df),
        columns=df.columns
    )
    
    if cols is None:
        cols = ['num_vertices', 'num_edges', 'max_degree', 'min_degree', 
                'mean_degree', 'average_clustering', 'global_clustering', 
                'density', 'diameter', 'radius']
    
    # Scale features
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    
    # Split features and target
    X = df.drop('Chordal', axis=1)
    y = df['Chordal']
    
    return train_test_split(X, y, test_size=split, random_state=7)

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")



def logistic_regression(X_train, X_test, y_train, y_test, print = True):
    lr_clf = LogisticRegression(solver='liblinear')
    lr_clf.fit(X_train, y_train)

    if print:
        print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
        print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)

def K_nearest_neighbour(X_train, X_test, y_train, y_test, print = True):
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)

    if print:
        print_score(knn_clf, X_train, y_train, X_test, y_test, train=True)
        print_score(knn_clf, X_train, y_train, X_test, y_test, train=False)

def support_vector_machine(X_train, X_test, y_train, y_test, print = True):
    svm_clf = SVC(kernel='rbf', gamma=0.1, C=1.0)
    svm_clf.fit(X_train, y_train)

    if print:
        print_score(svm_clf, X_train, y_train, X_test, y_test, train=True)
        print_score(svm_clf, X_train, y_train, X_test, y_test, train=False)

def decision_tree_classifier(X_train, X_test, y_train, y_test, print = True):
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)

    if print:
        print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
        print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)

def random_forrest(X_train, X_test, y_train, y_test, print = True):
    rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf_clf.fit(X_train, y_train)

    if print:
        print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
        print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)

def xgboost_classifier(X_train, X_test, y_train, y_test, print = True):
    xgb_clf = XGBClassifier(use_label_encoder=False)
    xgb_clf.fit(X_train, y_train)

    if print:
        print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)
        print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)

def compare_models(X_train, X_test, y_train, y_test):
    """
    Run multiple classifiers and compare their performance
    """
    models = {
        'Logistic Regression': logistic_regression,
        'KNN': K_nearest_neighbour,
        'SVM': support_vector_machine,
        'Decision Tree': decision_tree_classifier,
        'Random Forest': random_forrest,
        'XGBoost': xgboost_classifier
    }
    
    results = []
    
    for name, model_func in models.items():
        # Run model without printing
        model_func(X_train, X_test, y_train, y_test, print=False)
        
        # Get classifier instance
        if name == 'Logistic Regression':
            clf = LogisticRegression(solver='liblinear')
        elif name == 'KNN':
            clf = KNeighborsClassifier()
        elif name == 'SVM':
            clf = SVC(kernel='rbf', gamma=0.1, C=1.0)
        elif name == 'Decision Tree':
            clf = DecisionTreeClassifier(random_state=42)
        elif name == 'Random Forest':
            clf = RandomForestClassifier(n_estimators=1000, random_state=42)
        else:  # XGBoost
            clf = XGBClassifier(use_label_encoder=False)
            
        clf.fit(X_train, y_train)
        
        # Get scores
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred) * 100
        test_acc = accuracy_score(y_test, test_pred) * 100
        
        results.append({
            'Model': name,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'Difference': train_acc - test_acc
        })
    
    return pd.DataFrame(results).round(2)


def plot_feature_importance(X_train, y_train):
    # Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf.fit(X_train, y_train)
    
    # XGBoost Feature Importance
    xgb = XGBClassifier(use_label_encoder=False)
    xgb.fit(X_train, y_train)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Random Forest plot
    importances_rf = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=importances_rf, ax=ax1)
    ax1.set_title('Random Forest Feature Importance')
    
    # XGBoost plot
    importances_xgb = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=importances_xgb, ax=ax2)
    ax2.set_title('XGBoost Feature Importance')
    
    plt.tight_layout()
    return importances_rf, importances_xgb

def plot_logistic_coefficients(X_train, y_train):
    # Logistic Regression Coefficients
    lr = LogisticRegression(solver='liblinear')
    lr.fit(X_train, y_train)
    
    plt.figure(figsize=(10, 6))
    coef_df = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': lr.coef_[0]
    }).sort_values('coefficient', ascending=False)
    
    sns.barplot(x='coefficient', y='feature', data=coef_df)
    plt.title('Logistic Regression Coefficients')
    return coef_df

def visualize_decision_tree(X_train, y_train, max_depth=3):
    from sklearn.tree import plot_tree
    """
    Create and visualize a decision tree classifier
    max_depth parameter limits tree depth for better visualization
    """
    # Create and train a decision tree
    tree_clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    tree_clf.fit(X_train, y_train)
    
    # Set up the figure with a larger size
    plt.figure(figsize=(20,10))
    
    # Plot the tree
    plot_tree(tree_clf, 
              feature_names=X_train.columns,
              class_names=['Not Chordal', 'Chordal'],
              filled=True,
              rounded=True,
              fontsize=10)
    
    plt.title("Decision Tree for Chordal Graph Classification")
    
    return tree_clf