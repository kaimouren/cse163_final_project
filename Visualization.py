"""
Team:
Yezi Liu (yliu58@uw.edu)
Xiaokai Xu (xiaokx@uw.edu)
Russell Liu (hl0206@uw.edu)

Description:
This file uses the loan data in CSV format from the LendingClub to create
data visualizations for displaying interesting facts in this dataset as
well as main logistic regression curves for various variables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sns.set()


def visualize_plot_1(data: pd.DataFrame) -> None:
    """
    Takes in the loan data as a DataFrame, plots a pie chart
    to display different loan purposes, and saves it as
    'plot1.png'. It returns None.
    """
    purpose_counts = data['purpose'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(purpose_counts.values, labels=purpose_counts.index,
            autopct='%1.1f%%')
    plt.title("Loan Purposes")
    plt.savefig('plot1.png')
    plt.close()


def visualize_plot_2(data: pd.DataFrame) -> None:
    """
    Takes in the loan data as a DataFrame, plots a histogram to
    display values from the 'credit.policy' column, and saves
    it as 'plot2.png'. It returns None.
    """
    sns.catplot(x='credit.policy', kind='count', color='b', data=data)
    plt.title('1 represent that the customer meets the ' +
              'borrowing criteria, and 0 otherwise.')
    plt.savefig('plot2.png')
    plt.close()


def fit_logistic_regression(X: pd.DataFrame, y: pd.Series):
    """
    Takes in 2 parameters: X is a DataFrame and y is a Series. It
    returns the fitted logistic regression model.
    """
    logit_model = sm.Logit(y, X)
    return logit_model.fit()


def plot_hist(data: pd.DataFrame, x_column: str) -> None:
    """
    Takes in the data as a DataFrame and x_column as
    a string, plots a histogram for x_column in the data, and saves
    it with the column name. It returns None.
    """
    sns.histplot(data[x_column], kde=True, stat='density', color='grey')
    plt.xlabel(x_column)
    plt.ylabel('Density')
    plt.savefig(f'hist{x_column}.png')
    plt.close()


def plot_logistic_regression(data: pd.DataFrame, x_column: str) -> None:
    """
    Takes in the loan data as a DataFrame and the x_column as a string,
    and plots the logistic regression curve with the input x_column and the
    'credit.policy' as y column. It saves the plot with the x_column name
    and returns None.
    """
    y = data['credit.policy']
    X = sm.add_constant(data[[x_column]])
    result = fit_logistic_regression(X, y)

    # Define the range of x values for the plot
    x_range = np.linspace(data[x_column].min(), data[x_column].max(), 100)

    # Calculate the predicted probabilities for the range of x values
    X_plot = sm.add_constant(x_range)  # check the size
    y_plot = result.predict(X_plot)

    # Plot the logistic regression curve
    plt.scatter(data[x_column], y)
    plt.plot(x_range, y_plot, color='red')
    plt.xlabel(x_column)
    plt.ylabel('Probability of Meeting Borrowing Criteria')
    plt.savefig(f'{x_column}.png')
    plt.close()


def logistic_regression(data: pd.DataFrame) -> None:
    """
    Takes in the loan data as a DataFrame, plots the
    respective logistic regression curves with corresponding x columns of
    'fico', 'inq.last.6mths', 'log.annual.inc', 'installment' and y column of
    'credit.policy', and plots the respective histograms for these 4 x columns
    in the data. It returns None.
    """
    # Fit logistic regression models and plot curves for various variables
    plot_logistic_regression(data, 'fico')
    plot_logistic_regression(data, 'inq.last.6mths')
    plot_logistic_regression(data, 'log.annual.inc')
    plot_logistic_regression(data, 'installment')

    plot_hist(data, 'fico')
    plot_hist(data, 'inq.last.6mths')
    plot_hist(data, 'log.annual.inc')
    plot_hist(data, 'installment')


def main():
    data = pd.read_csv("loan_data.csv")
    print(data.head())
    print()
    print()
    print(data.shape)
    print()
    print()
    summary_table = data.describe()
    summary_table = summary_table.T.reset_index().rename(columns={'index':
                                                                  'variable'})
    summary_table = summary_table[['variable', 'count', 'mean',
                                   'std', 'min', '25%', '50%', '75%', 'max']]
    print(summary_table)
    visualize_plot_1(data)
    visualize_plot_2(data)
    logistic_regression(data)


if __name__ == '__main__':
    main()
