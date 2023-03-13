"""
Team:
Yezi Liu (yliu58@uw.edu)
Xiaokai Xu (xiaokx@uw.edu)
Russell Liu (hl0206@uw.edu)

Description:
This file uses the loan data in CSV format from the LendingClub to
build decision tree model and random forest model and analyze the
random forest's future performance based on 2 types of errors.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm


def encode_purpose_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes the 'purpose' column of the input DataFrame and returns the
    resulting DataFrame with the one-hot encoded columns concatenated.
    """
    purpose_dummies = pd.get_dummies(data['purpose'], prefix='purpose')
    return pd.concat([data, purpose_dummies], axis=1)


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,
                                            pd.DataFrame, pd.DataFrame]:
    """
    Take in the loan data as a DataFrame and returns a tuple containing the
    training and testing sets for both features and labels, split in a ratio of
    80:20.
    """
    load_data = encode_purpose_column(data)
    # Split the data into training (80%) and testing (20%) sets
    train, test = train_test_split(load_data, test_size=0.2, random_state=42)
    # Define the X and y variables for ML model
    X_train, y_train = train.drop(['purpose', 'credit.policy'],
                                  axis=1), train['credit.policy']
    X_test, y_test = test.drop(['purpose', 'credit.policy'],
                               axis=1), test['credit.policy']
    return X_train, y_train, X_test, y_test


def train_decision_tree(X_train: pd.DataFrame, y_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_test: pd.DataFrame) -> tuple[DecisionTreeClassifier,
                                                       int, float, float]:
    """
    Take in the loan_data as a DataFrame and train a decision tree
    classifier on the training set, selects the best hyperparameters
    using grid search cross-validation, and returns the number of
    the best hyperparameter and accuracy score for the model
    on the both train and test set.
    """
    dt = DecisionTreeClassifier(random_state=42)
    # Set up a decision tree classifier and a range of hyperparameters to test
    params = {'max_depth': [6, 8, 10, 12, 14, 16]}
    clf = GridSearchCV(dt, params, cv=5)
    clf.fit(X_train, y_train)
    best_depth = clf.best_params_['max_depth']
    val_score = clf.best_score_
    dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    return dt, best_depth, test_score, val_score


def plot_tree(model: DecisionTreeClassifier, features: pd.DataFrame,
              labels: pd.DataFrame) -> None:
    """
    visualizes a Decision Tree using Graphviz, and saves it as
    'decision_tree'. It returns None.
    """
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=list(features.columns),
                               class_names=list(map(str, labels.unique())),
                               impurity=False,
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render("decision_tree")


def random_forest_classifier(X_train: pd.DataFrame,
                             y_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_test: pd.DataFrame
                             ) -> tuple[RandomForestClassifier,
                                        int, int, float, float]:
    """
    Trains a random forest classifier on the training data using grid search
    cross-validation to find the best hyperparameters. Returns the trained
    model, best hyperparameters, validation score, and test score. The
    function takes less than 10 minutes to run.
    """
    # Set up a random forest classifier and a range of hyperparameters to test
    rf = RandomForestClassifier(random_state=42)
    params = {'n_estimators': [150, 200, 250, 300],
              'max_depth': [15, 20, 25, 30]}
    # Use grid search cross-validation to find the best hyperparameters
    clf = GridSearchCV(rf, params, cv=5)
    clf.fit(X_train, y_train)
    # Get the best hyperparameters and the corresponding accuracy score on
    # the validation set
    best_n_estimators = clf.best_params_['n_estimators']
    best_max_depth = clf.best_params_['max_depth']
    val_score = clf.best_score_
    rf = RandomForestClassifier(n_estimators=best_n_estimators,
                                max_depth=best_max_depth, random_state=42)
    rf.fit(X_train, y_train)
    # Use the trained model to predict on the test set and calculate the
    # accuracy score
    y_pred = rf.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    return rf, best_n_estimators, best_max_depth, val_score, test_score


def confusion_matrix_plotting(model: RandomForestClassifier, X: pd.DataFrame,
                              y_true: pd.DataFrame) -> None:
    """
    Take in parameters of trained random forest classifier, features of the
    test data, and true labels of the test data to generates a confusion matrix
    plot for a given model and test data, and saves it as 'confusion_matrix'.
    It returns None.
    """
    y_pred = model.predict(X)

    cm_data = cm(y_true, y_pred)
    cm_data = cm_data.ravel()[::-1].reshape(cm_data.shape)
    plt.figure(figsize=(8, 8), dpi=80)
    sns.heatmap(cm_data, annot=True, fmt=".0f",
                xticklabels=['Pred Positive', 'Pred Negative'],
                yticklabels=['Actual Positive', 'Actual Negative'])
    plt.savefig('confusion_matrix.png')


def main():
    data = pd.read_csv("loan_data.csv")

    X_train, y_train, X_test, y_test = split_data(data)

    dt, best_depth, test_score, val_score = train_decision_tree(X_train,
                                                                y_train,
                                                                X_test, y_test)
    print("Decision Tree - Best depth: {}".format(best_depth))
    print("Decision Tree - Validation score: {:.3f}".format(val_score))
    print("Decision Tree - Test score: {:.3f}".format(test_score))
    plot_tree(dt, X_train, y_train)

    rf, best_n_estimators, best_max_depth,\
        val_score, test_score = random_forest_classifier(X_train, y_train,
                                                         X_test, y_test)
    print("Random Forest - Best n_estimators: {}, Best max_depth: {}"
          .format(best_n_estimators, best_max_depth))
    print("Random Forest - Validation score: {:.3f}".format(val_score))
    print("Random Forest - Test score: {:.3f}".format(test_score))

    confusion_matrix_plotting(rf, X_test, y_test)


if __name__ == '__main__':
    main()
