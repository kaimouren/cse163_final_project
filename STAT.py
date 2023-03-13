"""
Team:
Yezi Liu (yliu58@uw.edu)
Xiaokai Xu (xiaokx@uw.edu)
Russell Liu (hl0206@uw.edu)

Description:
This file uses the loan data in CSV format from the LendingClub to
perform logistic regression and LASSO regression and display their
respective results.
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso


def encode_purpose_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in the loan data as a DataFrame and one-hot encodes the 'purpose'
    column. It returns the resulting DataFrame with the one-hot encoded columns
    concatenated.
    """
    purpose_dummies = pd.get_dummies(data['purpose'], prefix='purpose')
    return pd.concat([data, purpose_dummies], axis=1)


def logistic_regression(data: pd.DataFrame) -> None:
    """
    Take in the loan_data as a DataFrame, and perform a logistic
    regression analysis on the given dataset and print a summary
    of the results.
    """
    loan_data_encoded = encode_purpose_column(data)
    # Define the X and y variables for logistic regression
    y = loan_data_encoded['credit.policy']
    X = loan_data_encoded.drop(['purpose', 'credit.policy'], axis=1)
    # Fit a logistic regression model using statsmodels
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    print(result.summary2())


def lasso_regression(data: pd.DataFrame) -> float:
    """
    Take in the loan data as a DataFrame, and perform a LASSO
    regression analysis on the given dataset and print a summary
    of the results.
    """
    X = encode_purpose_column(data).drop(['purpose', 'credit.policy'], axis=1)
    y = encode_purpose_column(data)['credit.policy']

    # an alpha value of 0.1, which controls the strength of the penalty term
    lasso = Lasso(alpha=0.1)
    # Train the model on the training set
    lasso.fit(X, y)
    # Create a DataFrame to store the coefficients and their
    # corresponding feature names
    coef_df = pd.DataFrame({'Features': X.columns,
                            'Coefficients': lasso.coef_})
    # Sort the DataFrame by absolute value of the coefficients
    # in descending order
    coef_df = coef_df.reindex(coef_df['Coefficients'].abs()
                              .sort_values(ascending=False).index)
    # Print the sorted coefficients
    print("Coefficient of Each Feature")
    print(coef_df)


def main():
    data = pd.read_csv("loan_data.csv")
    print('Logistic Regression Result')
    print()
    print()
    logistic_regression(data)
    print('LASSO Regression Result')
    lasso_regression(data)


if __name__ == '__main__':
    main()
