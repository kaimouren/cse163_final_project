# The Effect of Borrower Identities on Loan Application in the P2P Market
## Team
Yezi Liu (yliu58@uw.edu):smile:

Xiaokai Xu (xiaokx@uw.edu):grinning:

Russell Liu (hl0206@uw.edu):smiley:
## Requirements
Python 3.7 or later
pandas
matplotlib
seaborn
scikit-learn
graphviz
## Introduction
This project examines the impact of borrower identities on loan applications in the P2P (peer-to-peer) market. The code is divided into three parts: Visualization, Statistical Method, and Machine Learnine. To downloan the dataset, we can go to website https://www.kaggle.com/datasets/itssuru/loan-data. We are using Pycharm as our integrated development environment.
### Visualization
This file contains code to visualize loan data and create logistic regression curves for various variables in the dataset. The data is read in from a CSV file, and the visualizations and plots are created using libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Statsmodels. Thus it might be helpful to import packages with the code
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sns.set()
```
The code reads in the loan data from a CSV file named "loan_data.csv". It then prints the head of the data, the shape, and a summary table with descriptive statistics for each variable. Below are the discription of each function.

1. `visualize_plot_1(data: pd.DataFrame) -> None`: Takes in the loan data as a DataFrame, plots a pie chart to display different loan purposes, and saves it as 'plot1.png'.

2. `visualize_plot_2(data: pd.DataFrame) -> None`: Takes in the loan data as a DataFrame, plots a histogram to display values from the 'credit.policy' column, and saves it as 'plot2.png'.

3. `fit_logistic_regression(X: pd.DataFrame, y: pd.Series)`: Takes in 2 parameters: X is a DataFrame and y is a Series. It returns the fitted logistic regression model.

4. `plot_hist(data: pd.DataFrame, x_column: str) -> None`: Takes in the data as a DataFrame and x_column as a string, plots a histogram for x_column in the data, and saves it with the column name.

5. `plot_logistic_regression(data: pd.DataFrame, x_column: str) -> None`: Takes in the loan data as a DataFrame and the x_column as a string, and plots the logistic regression curve with the input x_column and the 'credit.policy' as y column. It saves the plot with the x_column name.

Finally, the `logistic_regression` function calls the `plot_logistic_regression` function and the `plot_hist` function to create logistic regression curves and histograms for various variables in the data. 

To run the reproduce our result, you can just click the run button. The resulting plots will be saved as .png files in the same directory as the code. All other information are shown in Terminal.

### Statistical Method
The statistical part contains the code using the loan data in CSV format to perform logistic regression and LASSO regression and display their respective results with the use of packages pandas, statsmodels, and sklearn. So You should import these package with the code that:
```
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso
```
Both regression techniques are aim to find the relationship between dependent variable which here is credit.policy column and independent variables which are other columns in the dataset.  Below are the discription of each function.

`encode_purpose_column(data: pd.DataFrame) -> pd.DataFrame`: Takes in the loan data as a DataFrame and one-hot encodes the 'purpose' column. Returns the resulting DataFrame with the one-hot encoded columns concatenated.

`logistic_regression(data: pd.DataFrame) -> None`: Takes in the loan_data as a DataFrame, and perform a logistic regression analysis on the given dataset and print a summary of the results.

`lasso_regression(data: pd.DataFrame) -> float`: Takes in the loan data as a DataFrame, and perform a LASSO regression analysis on the given dataset and print a summary of the results.

To reproduce our results, You can directly run our code. The resulting plots will be saved as .png files in the same directory as the code. The coefficients results of logistic regression and LASSO regression are shown in Terminal
### Machine Learning
The machine learning part contains the code to perform decision tree and random forest analysis. Decision trees are a powerful tool for classification and prediction, while random forests are an ensemble learning method that uses multiple decision trees to improve the accuracy of the prediction. We use the several packages in this part: pandas, matplotlib, seaborn, and graphviz. To import these packages, you can run the code that；
···
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
···
Specifically, Graphviz requires some more steps to set up. You can follow the instruction from its offical website(https://www.computerhope.com/issues/ch000549.htm). Then, below is the description of each function:

`encode_purpose_column(data: pd.DataFrame) -> pd.DataFrame`: Takes in a Pandas DataFrame containing loan data. Returns a new DataFrame with one-hot encoded columns of the 'purpose' column concatenated to the input DataFrame.

`split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]`: Takes in a Pandas DataFrame containing loan data, splits the data into training and testing sets in a ratio of 8:2. Returns a tuple of the training and testing sets for both the features and labels.

`train_decision_tree(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> tuple[DecisionTreeClassifier, int, float, float]`: Takes in the training and testing sets for both the features and labels, trains a decision tree classifier on the training set, selects the best hyperparameters using grid search cross-validation. Returns the trained decision tree model, the number of the best hyperparameter, and the accuracy scores of the model on both the training and testing sets.

`plot_tree(model: DecisionTreeClassifier, features: pd.DataFrame, labels: pd.DataFrame) -> None`: Takes in a trained decision tree model, the features DataFrame, and the labels DataFrame, and visualizes the decision tree using Graphviz.

`random_forest_classifier(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> tuple[RandomForestClassifier, int, int, float, float]`: Takes in the training and testing sets for both the features and labels, trains a random forest classifier on the training set, selects the best hyperparameters using grid search cross-validation. Returns the trained random forest model, the best hyperparameters, and the accuracy scores of the model on both the validation and testing sets.

`confusion_matrix(y_true: Any, y_pred: Any) -> pd.DataFrame`: Takes in the true labels and predicted labels. Returns a confusion matrix showing the number of true positives, false positives, true negatives, and false negatives in a tabular form as a Pandas DataFrame.

Similarly, You can directly run our code and get the same result as we did. The resulting plots will be saved as .png files in the same directory as the code. The terminal will show the information including best hyperparameters of decisiontree and randomforest with the corresponding validation scores and accuracy scores.
