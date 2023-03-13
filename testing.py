import pandas as pd
from ML import split_data
from ML import encode_purpose_column


def test_length(data: pd.DataFrame, tolerance) -> None:
    '''
    Print the size of the dataset and check the length of X and y in train and test sets.
    '''
    load_data = encode_purpose_column(data)
    print("Size of the Dataset:")
    print(data.shape)
    print("Test One-hot coded:")
    print("The number of purpose:")
    print(len(data['purpose'].unique()))
    print("The number of encoded dataset")
    print(load_data.shape)
    X_train, y_train, X_test, y_test = split_data(load_data)
    print("Check the length of X and y in train_set: ", len(X_train) == len(y_train))
    print("Check the length of X and y in test_set: ", len(X_test) == len(y_test))
    train_portion = len(X_train)/len(data)
    test_portion = len(X_test)/len(data)
    print(train_portion)
    print("Check the length of train_set")
    print(abs(train_portion - 0.8) <= tolerance)
    print("Check the length of test_set")
    print(abs(test_portion - 0.2) <= tolerance)


def main():
    data = pd.read_csv("loan_data.csv")
    test_length(data, 0.001)


if __name__ == '__main__':
    main()