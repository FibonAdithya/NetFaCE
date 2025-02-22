import dtreeviz.trees
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


def preprocessing(dataset, cols = None, split = 0.3):
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split

    # Create DataFrame and add target variable
    if type(dataset) is tuple:
        df = pd.DataFrame(dataset[0])
        df["Chordal"] = dataset[1]
    else:
        df = dataset
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
    if X_train is None:
        return f"ERROR no test data for X was provided"
    elif y_test is None:
        return f"ERROR no test data for y was provided"
    
    print(f"{clf} \n ====================================================")
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    else:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")



def logistic_regression(X_train, y_train, X_test = None, y_test = None, print = False):
    from sklearn.linear_model import LogisticRegression

    lr_clf = LogisticRegression(solver='liblinear')
    lr_clf.fit(X_train, y_train)

    if print:
        print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
        print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)
    
    return lr_clf

def K_nearest_neighbour(X_train, y_train, X_test = None, y_test = None,  print = False):
    from sklearn.neighbors import KNeighborsClassifier

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)

    if print:
        print_score(knn_clf, X_train, y_train, X_test, y_test, train=True)
        print_score(knn_clf, X_train, y_train, X_test, y_test, train=False)
    
    return knn_clf

def support_vector_machine(X_train, y_train, X_test = None, y_test = None,  print = False):
    from sklearn.svm import SVC

    svm_clf = SVC(kernel='rbf', gamma=0.1, C=1.0)
    svm_clf.fit(X_train, y_train)

    if print:
        print_score(svm_clf, X_train, y_train, X_test, y_test, train=True)
        print_score(svm_clf, X_train, y_train, X_test, y_test, train=False)
    
    return svm_clf

def decision_tree(X_train, y_train, X_test = None, y_test = None,  print = False):
    from sklearn.tree import DecisionTreeClassifier

    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)

    if print:
        print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
        print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)
    
    return tree_clf

def random_forrest(X_train, y_train, X_test = None, y_test = None,  print = False):
    from sklearn.ensemble import RandomForestClassifier

    rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf_clf.fit(X_train, y_train)

    if print:
        print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
        print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
    
    return rf_clf

def xgboost(X_train, y_train, X_test = None, y_test = None,  print = False):
    from xgboost import XGBClassifier

    xgb_clf = XGBClassifier(use_label_encoder=False)
    xgb_clf.fit(X_train, y_train)

    if print:
        print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)
        print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)
    
    return xgb_clf

def compare_models(X_train, y_train, X_test, y_test, models = None):
    """
    Run multiple classifiers and compare their performance
    """
    results = []
    if models is None:
        models = {
            'Logistic Regression': logistic_regression,
            'KNN': K_nearest_neighbour,
            'SVM': support_vector_machine,
            'Decision Tree': decision_tree,
            'Random Forest': random_forrest,
            'XGBoost': xgboost
        }
        
        for name, model_func in models.items():
            # Run model
            clf = model_func(X_train, y_train)
            
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
    else:
        for model in models:
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_acc = accuracy_score(y_train, train_pred) * 100
            test_acc = accuracy_score(y_test, test_pred) * 100

            results.append({
                'Model': f"{model}",
                'Train Accuracy': train_acc,
                'Test Accuracy': test_acc,
                'Difference': train_acc - test_acc
            })
    
    return pd.DataFrame(results).round(2)


def plot_feature_importance(X_train, y_train, rf=None, xgb=None):
    # Random Forest Feature Importance
    if rf is None:
        rf = random_forrest(X_train, y_train)
    
    # XGBoost Feature Importance
    if xgb is None:
        xgb = xgboost(X_train, y_train)
    
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

def plot_logistic_coefficients(X_train, y_train, lr=None):
    # Logistic Regression Coefficients
    if lr is None:
        lr = logistic_regression(X_train, y_train)
    
    plt.figure(figsize=(10, 6))
    coef_df = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': lr.coef_[0]
    }).sort_values('coefficient', ascending=False)
    
    sns.barplot(x='coefficient', y='feature', data=coef_df)
    plt.title('Logistic Regression Coefficients')
    return coef_df

def visualize_decision_tree(X_train, y_train, max_depth=3, tree_clf = None):
    from sklearn.tree import plot_tree
    from supertree import SuperTree
    """
    Create and visualize a decision tree classifier
    max_depth parameter limits tree depth for better visualization
    """
    if tree_clf is None:
        # Create and train a decision tree
        tree_clf = decision_tree(X_train, y_train)
    
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

